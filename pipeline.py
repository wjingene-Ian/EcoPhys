"""Portable adapters for the three existing engines, sharing inputs and run records."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent
for engine in ['basic', 'handoff', 'chenma']:
    sys.path.insert(0, str(ROOT / 'engines' / engine))
import rmt_analysis as ra
import run_pipeline as cm
import build_global as bg


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def configure_plots() -> None:
    for path in ['/System/Library/Fonts/Hiragino Sans GB.ttc',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
                         'savefig.facecolor': 'white'})


def save(fig, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{name}.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def metadata_for(dataset: str, columns) -> pd.DataFrame | None:
    path = ROOT / 'data' / 'metadata' / f'{dataset}_industry_mapping.csv'
    if not path.exists():
        return None
    meta = pd.read_csv(path).set_index('ticker').reindex(columns)
    if meta[['industry_l1', 'industry_l2']].isna().any().any():
        raise ValueError('行业标签未覆盖全部输入股票。')
    return meta


def basic(panel, source, output, title):
    # Feed the already validated/selected panel into the original numerical engine.
    original = bg.read_prices
    bg.read_prices = lambda _: (panel, pd.DataFrame({'ticker': panel.columns}))
    try:
        bg.configure_plots()
        summary = bg.run_group(source, output, label=title)
    finally:
        bg.read_prices = original
    metrics = pd.read_csv(output / 'full_period_metrics.csv')
    return {'assets': summary['assets'], 'annual_days': 252,
            'covariance_returns': 'simple', 'initial_build_turnover': 1,
            'portfolio_metrics': metrics.to_dict('records')}


def audit(panel, returns, output, dataset):
    import analyze_hs300 as ah
    tables, figures = ah.ensure_dirs(output)
    if dataset == 'hs300':
        ah.DATASET_DIRS = {'price': 'prices', 'return': 'archived_returns', 'zscore': 'archived_zscores'}
        _, audit_table, extras = ah.audit_and_load(ROOT / 'data' / 'hs300')
        audit_table = audit_table[audit_table.ticker.isin(panel.columns)]
        extras['excluded'].to_csv(tables / '02_excluded_stocks.csv', index=False)
        extras['coverage'].to_csv(tables / '03_universe_coverage.csv')
        write_json(output / 'archive_reconstruction.json', extras['summary'])
    else:
        rows = []
        for ticker in returns:
            values = returns[ticker].to_numpy()
            zero = np.isclose(values, 0, atol=1e-14)
            rows.append({'ticker': ticker, 'price_observations': len(panel),
                         'return_observations': len(returns), 'zero_return_days': int(zero.sum()),
                         'max_consecutive_zero_returns': ah.longest_true_run(zero),
                         'membership_start': None, 'membership_end': None})
        audit_table = pd.DataFrame(rows)
        extras = {'coverage': pd.DataFrame({'constant_analysis_universe': len(panel.columns)}, index=returns.index)}
    audit_table.to_csv(tables / '01_data_audit_by_ticker.csv', index=False)
    metrics = ah.compute_metrics(returns, 60, 120, 5)
    stock = ah.compute_stock_summary(returns, audit_table)
    metrics.to_csv(tables / '04_market_daily_metrics.csv')
    stock.to_csv(tables / '05_stock_summary.csv', index=False)
    ah.build_event_candidates(metrics).to_csv(tables / '06_event_candidates.csv')
    returns.to_csv(tables / '07_log_return_matrix.csv')
    ah.compute_annual_summary(metrics, 60, 120).to_csv(tables / '08_annual_summary.csv', index=False)
    anomalies = ah.find_individual_return_anomalies(returns)
    anomalies.to_csv(tables / '09_individual_return_anomalies.csv', index=False)
    ah.save_figures(metrics, returns, extras['coverage'], figures, 60, 120)
    return {'assets': len(panel.columns), 'annual_days': 242, 'anomaly_records': len(anomalies),
            'three_archive_comparison': dataset == 'hs300',
            'global_membership_history': 'not provided'}


def rmt(returns, output, args):
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    n = returns.shape[1]
    result = ra.rmt_decomposition(returns.to_numpy())
    eigen, _, _ = ra.full_sample_tables(returns, result, tables)
    summary = ra.matrix_summary(returns.to_numpy(), 'full_selected_universe')
    pd.DataFrame([summary]).to_csv(tables / '01_full_sample_summary.csv', index=False)
    ra.save_core_figures(returns, result, eigen, figures)
    counts = sorted(set([k for k in [30, 50, 100, 150, 200, 250] if k < n] + [n]))
    counts_table = ra.stock_count_sensitivity(returns, counts, repeats=3 if args.quick else 30)
    counts_table.to_csv(tables / '06_stock_count_sensitivity.csv', index=False)
    counts_table.groupby('N').agg(
        repetitions=('N', 'size'), lambda_max_fraction_mean=('lambda_max_fraction', 'mean'),
        lambda_max_fraction_std=('lambda_max_fraction', 'std'),
        high_outlier_count_mean=('high_outlier_count', 'mean')).fillna(0).to_csv(tables / '06b_stock_count_summary.csv')
    windows = ra.window_length_sensitivity(returns, list(returns.columns))
    windows.to_csv(tables / '07_window_length_sensitivity.csv', index=False)
    regimes = ra.regime_sensitivity(returns, list(returns.columns))
    regimes.to_csv(tables / '08_market_regime_sensitivity.csv', index=False)
    # This preserves the handoff engine's log-return estimation and 242-day annualization.
    solver = ('projected' if n > 100 else 'slsqp') if args.solver == 'auto' else args.solver
    original_solver = ra.long_only_minimum_variance
    if solver == 'projected':
        from build_extension import optimize
        ra.long_only_minimum_variance = lambda cov, max_weight, initial=None: optimize(
            cov, initial, cap=max_weight, ridge_fraction=1e-6)
    try:
        daily, performance, rebalances, weights = ra.portfolio_backtest(
            returns, list(returns.columns), 504, 21, .05, 10)
    finally:
        ra.long_only_minimum_variance = original_solver
    daily.to_csv(tables / '09_portfolio_daily_returns.csv')
    performance.to_csv(tables / '10_portfolio_performance.csv', index=False)
    rebalances.to_csv(tables / '11_portfolio_rebalances.csv', index=False)
    weights.to_csv(tables / '12_latest_portfolio_weights.csv', index=False)
    ra.save_portfolio_figures(daily, performance, weights, figures)
    rng = np.random.default_rng(20261109)
    delta = []
    a, b = daily.raw_gmv.to_numpy(), daily.rmt_clipped_gmv.to_numpy()
    for _ in range(args.bootstrap_reps):
        starts = rng.integers(0, len(a), size=int(np.ceil(len(a) / 21)))
        ix = np.concatenate([(s + np.arange(21)) % len(a) for s in starts])[:len(a)]
        delta.append(float((b[ix].mean()/b[ix].std(ddof=1)-a[ix].mean()/a[ix].std(ddof=1))*np.sqrt(242)))
    pd.DataFrame({'delta_sharpe_clipped_minus_raw': delta}).to_csv(tables / 'bootstrap_delta_sharpe.csv', index=False)
    stress = []
    for name, start, end in [('2022下跌年', '2022-01-01', '2022-12-31'),
                              ('2024急涨急跌', '2024-09-24', '2024-10-31'),
                              ('2025四月冲击', '2025-04-01', '2025-04-30')]:
        for strategy in daily:
            values = daily.loc[start:end, strategy]
            if len(values) < 2:
                continue
            wealth = np.r_[1., (1+values).cumprod().to_numpy()]
            stress.append({'period': name, 'strategy': strategy, 'days': len(values),
                           'total_return': wealth[-1]-1, 'sharpe': values.mean()/values.std(ddof=1)*np.sqrt(242),
                           'drawdown': (wealth/np.maximum.accumulate(wealth)-1).min()})
    pd.DataFrame(stress).to_csv(tables / 'stress_metrics.csv', index=False)
    matched = []
    for end in ['2020-12-31', '2022-12-31', '2024-12-31', '2026-09-28']:
        for length in [120, 242, 484]:
            sub = returns.loc[:end].tail(length)
            if len(sub) == length:
                matched.append({'end': end, 'window': length, **ra.matrix_summary(sub.to_numpy(), end)})
    pd.DataFrame(matched).to_csv(tables / 'matched_window_sensitivity.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(delta, bins=30, color='#2D6688')
    ax.axvline(0, color='#C65E46', linestyle='--')
    ax.set(xlabel='Sharpe difference: clipped - raw', ylabel='Bootstrap frequency')
    save(fig, figures, '13_sharpe_difference_bootstrap')
    return {'assets': n, 'annual_days': 242, 'covariance_returns': 'log', 'initial_build_turnover': 0, 'solver': solver,
            'bootstrap_repetitions': args.bootstrap_reps,
            'sharpe_difference_ci95': np.quantile(delta, [.025, .975]).tolist(),
            'portfolio_metrics': performance.to_dict('records')}


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    return np.minimum(q, 1)


def eta(vector, labels):
    denominator = np.sum((vector-vector.mean())**2)
    if denominator <= 1e-20:
        return 0.
    return sum((labels == g).sum()*(vector[labels == g].mean()-vector.mean())**2
               for g in np.unique(labels))/denominator


def structure(returns, output, args, dataset, meta):
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    x = returns.to_numpy(); n = x.shape[1]
    z = ra.standardize(x); result = ra.rmt_decomposition(x)
    vals, vectors = result.eigenvalues[::-1], result.eigenvectors[:, ::-1].copy()
    for j in range(n):
        vectors[:, j] = ra.orient_vector(vectors[:, j])
    ac = []
    for lag in range(1, 61):
        ac.append({'lag': lag,
                   'return_median': np.median([np.corrcoef(z[:-lag, i], z[lag:, i])[0, 1] for i in range(n)]),
                   'abs_return_median': np.median([np.corrcoef(abs(z[:-lag, i]), abs(z[lag:, i]))[0, 1] for i in range(n)])})
    ac = pd.DataFrame(ac); ac.to_csv(tables / 'autocorrelation.csv', index=False)
    tails = pd.DataFrame([{'sigma': s, 'empirical': np.mean(abs(z) > s), 'gaussian': 2*stats.norm.sf(s)} for s in [2, 3, 4, 5]])
    tails.to_csv(tables / 'tail_probabilities.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    den, edges = np.histogram(z, bins=np.linspace(-8, 8, 100), density=True)
    centers = (edges[1:]+edges[:-1])/2
    ax.plot(centers, den, label='Standardized stock returns')
    ax.plot(centers, stats.norm.pdf(centers), '--', label='N(0,1)')
    ax.set(yscale='log', ylim=(1e-5, 1), xlabel='z', ylabel='Density'); ax.legend()
    save(fig, figures, '01_fat_tails')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ac.lag, ac.return_median, label='Returns')
    ax.plot(ac.lag, ac.abs_return_median, label='Absolute returns'); ax.axhline(0, color='grey')
    ax.set(xlabel='Lag (trading days)', ylabel='Median stock ACF'); ax.legend()
    save(fig, figures, '02_volatility_clustering')
    half = len(x)//2
    first, last = ra.rmt_decomposition(x[:half]), ra.rmt_decomposition(x[half:])
    va, vb = first.eigenvectors[:, ::-1], last.eigenvectors[:, ::-1]
    overlap = abs(va.T@vb)
    split = pd.DataFrame([{'train_rank': j+1, 'best_test_rank': int(overlap[j].argmax()+1),
                           'best_overlap': overlap[j].max(),
                           'test_variance': va[:, j]@last.correlation@va[:, j]} for j in range(min(n, 25))])
    split.to_csv(tables / 'chronological_mode_reliability.csv', index=False)
    rng = np.random.default_rng(20261110)
    null = [ra.rmt_decomposition(np.column_stack([rng.permutation(x[:, i]) for i in range(n)])).eigenvalues[-1]
            for _ in range(args.null_reps)]
    pd.DataFrame({'lambda_max': null}).to_csv(tables / 'shuffle_control.csv', index=False)
    meta_summary = {'assets': n, 'pooled_excess_kurtosis': stats.kurtosis(z.ravel(), fisher=True, bias=False),
                    'market_equal_weight_proxy_correlation': np.corrcoef(z@vectors[:, 0], z.mean(axis=1))[0, 1],
                    'industry_analysis': 'unavailable' if meta is None else 'complete'}
    # Retrospective data-quality diagnostic, never used to choose trading assets.
    import analyze_hs300 as ah
    suspect = [i for i in range(n) if ah.longest_true_run(np.isclose(x[:, i], 0, atol=1e-14)) >= 6
               or (abs(np.expm1(x[:, i])) > .25).any()]
    clean = [i for i in range(n) if i not in suspect]
    quality = [ra.matrix_summary(x, 'baseline')]
    if len(clean) >= 2:
        quality.append(ra.matrix_summary(x[:, clean], 'retrospective_clean'))
    pd.DataFrame(quality).to_csv(tables / 'data_quality_sensitivity.csv', index=False)
    pd.DataFrame({'ticker': returns.columns, 'retrospective_flag': [i in suspect for i in range(n)]}).to_csv(
        tables / 'data_quality_flags.csv', index=False)
    meta_summary['data_quality_note'] = 'full-sample diagnostic only; not used for portfolio selection'
    if meta is not None:
        labels = meta.industry_l1.to_numpy()
        permutations = [rng.permutation(labels) for _ in range(args.permutations)]
        mode_rows, sector_rows = [], []
        for j in range(n):
            obs = eta(vectors[:, j], labels)
            null_eta = np.array([eta(vectors[:, j], g) for g in permutations])
            mode_rows.append({'rank': j+1, 'eigenvalue': vals[j], 'eta2': obs,
                              'p': (1+(null_eta >= obs).sum())/(args.permutations+1),
                              'within_mp': result.lambda_minus <= vals[j] <= result.lambda_plus})
            for g in np.unique(labels):
                ix = labels == g
                sector_rows.append({'rank': j+1, 'industry': g, 'n': int(ix.sum()),
                                    'mean_loading': np.sqrt(n)*vectors[ix, j].mean(),
                                    'energy_fraction': (vectors[ix, j]**2).sum(),
                                    'energy_enrichment': n*(vectors[ix, j]**2).mean()})
        modes = pd.DataFrame(mode_rows); modes['q'] = bh(modes.p)
        modes.to_csv(tables / 'all_mode_information.csv', index=False)
        sectors = pd.DataFrame(sector_rows); sectors.to_csv(tables / 'all_mode_industry_loadings.csv', index=False)
        meta_summary['within_mp_industry_q05_count'] = int(modes.query('within_mp and q < .05').shape[0])
        grid = sectors.query('rank <= 10').pivot(index='rank', columns='industry', values='mean_loading')
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(grid, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
        ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=60, ha='right', fontsize=8)
        ax.set_yticks(range(len(grid)), grid.index); ax.set_ylabel('Mode rank')
        fig.colorbar(im, ax=ax, label='Mean sqrt(N) loading')
        save(fig, figures, '03_mode_industry_heatmap')
        palette = dict(zip(sorted(set(labels)), plt.get_cmap('tab20')(np.linspace(0, 1, len(set(labels))))))
        order = np.argsort(labels)
        fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
        for j, ax in enumerate(axes.flat):
            ax.bar(range(n), np.sqrt(n)*vectors[order, j], color=[palette[v] for v in labels[order]], width=1)
            ax.set_title(f'Mode {j+1}, lambda={vals[j]:.3f}', fontsize=10)
        save(fig, figures, '04_top10_industry_colored_loadings')
    benchmark = ROOT / 'data' / 'metadata' / f'{dataset}_index_response.json'
    if benchmark.exists():
        response = json.loads(benchmark.read_text())
        frame = pd.DataFrame([row.split(',') for row in response['response']['data']['klines']])
        price = pd.Series(frame[2].astype(float).to_numpy(), index=pd.to_datetime(frame[0]))
        index_return = np.log(price).diff().dropna()
        comparison = pd.DataFrame({'pc_standardized_returns': z@vectors[:, 0],
                                    'course_raw_return_factor': x@vectors[:, 0],
                                    'equal_weight_standardized_proxy': z.mean(axis=1)}, index=returns.index)
        comparison['index_log_return'] = index_return
        comparison = comparison.dropna()
        comparison.to_csv(tables / 'official_index_comparison.csv')
        meta_summary['official_index_pc_correlation'] = comparison.pc_standardized_returns.corr(comparison.index_log_return)
        meta_summary['official_index_days'] = len(comparison)
        meta_summary['official_index_source'] = response['url']
    else:
        meta_summary['official_index_comparison'] = 'unavailable: no local benchmark data'
    return meta_summary


def transform(x, method, h):
    frame = pd.DataFrame(x)
    if method == 'daily': return x
    if method == 'overlap': return frame.rolling(h).sum().dropna().to_numpy()
    if method == 'nonoverlap': return x[:len(x)//h*h].reshape(-1, h, x.shape[1]).sum(axis=1)
    if method == 'return_detrend': return (frame-frame.rolling(h, center=True).mean()).dropna().to_numpy()
    if method == 'profile_dma':
        profile = (frame-frame.mean()).cumsum()
        return (profile-profile.rolling(h, center=True).mean()).dropna().to_numpy()
    raise ValueError(method)


def gram(x, center=True):
    x = x-x.mean(axis=0) if center else x
    energy = np.sqrt((x*x).sum(axis=0))
    if (energy <= 1e-12).any(): raise ValueError('变换后的序列能量为零。')
    normalized = x/energy
    return normalized.T@normalized


def scales(returns, output, args):
    output.mkdir(parents=True, exist_ok=True)
    x = returns.to_numpy(); n = x.shape[1]
    ref = ra.rmt_decomposition(x).eigenvectors[:, -1]
    rng = np.random.default_rng(20261110)
    rows, matrices = [], {}
    specs = [('daily', 1)]+[(m, h) for m in ['overlap', 'nonoverlap', 'return_detrend', 'profile_dma'] for h in [5, 21, 63]]
    for method, h in specs:
        processed = transform(x, method, h)
        C = gram(processed, center=method != 'profile_dma')
        eig, vec = np.linalg.eigh(C)
        null_max, null_third = [], []
        for _ in range(args.null_reps):
            shuffled = np.column_stack([rng.permutation(x[:, i]) for i in range(n)])
            e = np.linalg.eigvalsh(gram(transform(shuffled, method, h), center=method != 'profile_dma'))
            null_max.append(e[-1]); null_third.append(e[-3])
        rows.append({'method': method, 'scale': h, 'T': len(processed), 'rank': np.linalg.matrix_rank(C, tol=1e-9),
                     'market_fraction': eig[-1]/n, 'lambda1': eig[-1], 'lambda3': eig[-3],
                     'avg_correlation': C[np.triu_indices(n, 1)].mean(),
                     'naive_mp_upper': (1+np.sqrt(n/len(processed)))**2,
                     'null_max_q95': np.quantile(null_max, .95), 'null_third_q95': np.quantile(null_third, .95),
                     'market_vector_overlap': abs(vec[:, -1]@ref), 'eigen_min': eig[0]})
        matrices[f'{method}_{h}'] = C
        print(f'  scales: {method} {h}', flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(output / 'scale_dma_comparison.csv', index=False)
    np.savez_compressed(output / 'scale_matrices.npz', **matrices)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    labels = [f'{r.method}\n{r.scale}d' for r in frame.itertuples()]
    ax.plot(range(len(frame)), frame.lambda1, 'o-', label='Observed maximum eigenvalue')
    ax.plot(range(len(frame)), frame.null_max_q95, 'o--', label='Matched shuffled 95% quantile')
    ax.set_xticks(range(len(frame)), labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Eigenvalue'); ax.legend()
    save(fig, output / 'figures', '01_scale_comparison')
    return {'assets': n, 'matrices': len(rows), 'null_repetitions': args.null_reps,
            'centered_filters': 'structure only; never enter portfolio backtest'}


def industry(returns, output, args, dataset):
    path = ROOT / 'data' / 'metadata' / f'{dataset}_industry_raw.csv'
    if not path.exists():
        return {'status': 'skipped', 'reason': '未提供本市场的行业标签，不能套用沪深行业。'}
    import industry_structure_analysis as isa
    output.mkdir(parents=True, exist_ok=True)
    csv = output / 'log_returns.csv'; returns.to_csv(csv)
    original = isa.parse_args
    from argparse import Namespace
    isa.parse_args = lambda: Namespace(returns=csv, metadata=path, output=output, permutations=args.permutations)
    try:
        isa.main()
    finally:
        isa.parse_args = original
    # Industry-centroid association tree from the same selected universe.
    meta = isa.prepare_metadata(path, list(returns.columns))
    group = ra.correlation_components(ra.rmt_decomposition(returns.to_numpy()))['group']
    names, weights = [], []
    for (l1, l2), sub in meta.groupby(['industry_l1', 'industry_l2']):
        w = np.zeros(returns.shape[1]); w[returns.columns.get_indexer(sub.index)] = 1/len(sub)
        weights.append(w); names.append(f'{l1} / {l2} ({len(sub)})')
    W = np.asarray(weights); C = ra.normalize_to_correlation(W@group@W.T)
    distance = np.sqrt(np.maximum(0, 2*(1-C))); np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method='average', optimal_ordering=True)
    fig, ax = plt.subplots(figsize=(12, max(8, len(names)*.25)))
    dendrogram(tree, orientation='left', labels=names, leaf_font_size=8, ax=ax)
    ax.set_xlabel('Group-mode correlation distance')
    save(fig, output / 'figures', 'second_level_industry_tree')
    return {'assets': returns.shape[1], 'industry_tree_leaves': len(names), 'permutations': args.permutations}


def modes(panel, output, args, source):
    import mode_structure_extension as ms
    import residual_and_risk_extension as rr
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    n = panel.shape[1]; logp = np.log(panel.to_numpy())
    original_modes = cm.MODES.copy()
    cm.MODES = {name: i for name, i in original_modes.items() if -n <= i < n}
    rows, eigenrows, details = [], [], {}
    try:
        for tau in range(1, args.tau_max+1):
            data, eig, detail = cm.analyze(logp, tau, 'observed', 'full', tau in [1, args.tau_max])
            rows.extend(data); details[tau] = detail
            eigenrows.extend({'tau_days': tau, 'rank_ascending': j+1, 'eigenvalue': e} for j, e in enumerate(eig))
        rng = np.random.default_rng(2018)
        controls = sorted(set(t for t in [1, 20, 60, 120, 250, args.tau_max] if t <= args.tau_max))
        for j in range(args.null_reps):
            # Common date permutation preserves contemporaneous cross-stock dependence.
            shuffled = cm.permute_daily_returns(logp, rng)
            for tau in controls:
                data, _, _ = cm.analyze(shuffled, tau, f'common_date_shuffle_{j+1}', 'full')
                rows.extend(data)
    finally:
        cm.MODES = original_modes
    frame = pd.DataFrame(rows); frame.to_csv(tables / 'mode_metrics.csv', index=False)
    pd.DataFrame(eigenrows).to_csv(tables / 'eigenvalues.csv', index=False)
    for tau in [1, args.tau_max]:
        detail = details[tau]
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        for name in ['largest', 'second_largest']:
            d = detail[name]
            axes[0].plot(panel.index[tau:], d['time']*np.sqrt(len(d['time'])), label=name, alpha=.8)
            axes[1].loglog(d['frequency'], d['power'], label=name, alpha=.8)
        axes[0].set(title=f'Tau={tau}: standardized KL time components', ylabel='Own standard deviation')
        axes[1].set(xlabel='Frequency (1/trading day)', ylabel='Power')
        for ax in axes: ax.legend()
        save(fig, figures, f'01_time_and_spectrum_tau{tau}')
    observed = frame.query("source == 'observed'")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name, data in observed.groupby('mode'):
        axes[0].plot(data.tau_days, data.variance_share, label=name)
        axes[1].plot(data.tau_days, data.alpha, label=name)
    axes[0].set_ylabel('Variance share'); axes[1].set(xlabel='Tau (trading days)', ylabel='Spectral slope')
    axes[0].legend(); save(fig, figures, '02_tau_scan')
    cm.make_tau1_covariance_figure(panel, figures / '07_covariance_tau1.png', tables / 'covariance_tau1_order.json')
    cm.make_market_removed_covariance_figure(panel, tables / 'covariance_tau1_order.json', figures / '08_market_removed_covariance_tau1.png')
    cm.make_market_mode_tau1_time_figure(panel, figures / '09_market_mode_tau1_time.png')
    cm.make_tau1_top10_time_modes(panel, figures / '10_tau1_top10_time_modes.png',
                                tables / 'tau1_top10_time_modes.csv', tables / 'tau1_top10_eigenmodes.csv')
    cfg = json.loads((ROOT / 'engines/chenma/original_config.json').read_text())
    cfg['input_dir'] = str(source)
    cfg['output_dir'] = str(output)
    cfg['selected_stocks_file'] = str(output.parent / 'selected_stocks.csv')
    split_date = pd.Timestamp(cfg['split_date'])
    if min((panel.index < split_date).sum(), (panel.index >= split_date).sum()) < 100:
        cfg['split_date'] = str(panel.index[len(panel)//2].date())
    cfg['mode_structure']['bootstrap_repetitions'] = 20 if args.quick else 200
    cfg['residual_group']['permutation_repetitions'] = args.permutations
    cfg['risk_estimation']['fold_bootstrap_repetitions'] = 20 if args.quick else 1000
    write_json(output / 'config.json', cfg)
    ms.run_extension(panel, output, cfg)
    rr.run_extension(panel, output, cfg, cm.spectral_block_order)
    return {'assets': n, 'tau_max': args.tau_max, 'available_modes': list(details[1]),
            'split_date': cfg['split_date'], 'null_repetitions': args.null_reps,
            'common_date_shuffle': 'preserves cross-stock same-day dependence; tests ordering effects'}


def run_dataset(dataset, spec, prices, args, root):
    panel, manifest = cm.load_panel(prices)
    total_assets = panel.shape[1]
    if args.assets:
        if args.assets > total_assets: raise ValueError(f'请求{args.assets}只，但仅有{total_assets}只股票。')
        if dataset == 'hs300' and args.assets == 100:
            selection = pd.read_csv(ROOT / 'data/metadata/hs300_legacy_selection.csv')
            columns = selection.loc[selection.in_fixed_100, 'ticker'].tolist()
        else:
            order = np.random.default_rng(20261006).permutation(total_assets)
            columns = panel.columns[order[:args.assets]].tolist()
        panel = panel[columns]
    n = panel.shape[1]
    if n < 20 or len(panel) < 527:
        raise ValueError('当前统一分析至少需要20只股票及527个共同价格日。')
    profile = f'sample{n}' if args.assets else 'full'
    if args.quick: profile += '_quick'
    output = (args.output.expanduser().resolve() if args.output else root / 'outputs') / dataset / profile
    output.mkdir(parents=True, exist_ok=True)
    returns = np.log(panel).diff().iloc[1:]; returns.index.name = 'Date'
    returns.to_csv(output / 'log_returns.csv')
    manifest['included'] = manifest.file.map(lambda f: Path(f).stem in panel.columns)
    manifest.to_csv(output / 'input_manifest.csv', index=False)
    pd.DataFrame({'ticker': panel.columns}).to_csv(output / 'selected_stocks.csv', index=False)
    metadata = metadata_for(dataset, panel.columns)
    stages = ['basic', 'audit', 'rmt', 'structure', 'industry', 'scales', 'modes'] if args.analysis == 'all' else [args.analysis]
    previous = {}
    manifest_path = output / 'run_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text()).get('stages', {})
    record = {'dataset': dataset, 'name_zh': spec['name_zh'], 'source': str(prices), 'source_assets': total_assets,
              'selected_assets': n, 'price_dates': len(panel), 'price_start': str(panel.index[0].date()),
              'price_end': str(panel.index[-1].date()), 'profile': profile,
              'arguments': vars(args), 'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'stages': previous}
    history_path = output / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'_'+args.analysis+'.json')
    record['code_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [root / 'run.py', root / 'pipeline.py', *sorted((root / 'engines').rglob('*.py'))]}
    write_json(output / 'run_manifest.json', record)
    print(f"\n{dataset}｜{spec['name_zh']}｜{n}/{total_assets}只｜{len(returns)}个收益日", flush=True)
    configure_plots()
    for stage in stages:
        destination = output / stage; destination.mkdir(parents=True, exist_ok=True)
        print(f'开始 {stage} → {destination}', flush=True)
        start = time.monotonic()
        try:
            with (destination / 'run.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                if stage == 'basic': info = basic(panel, prices, destination, f'{dataset.upper()} {spec["name_zh"]}')
                elif stage == 'audit': info = audit(panel, returns, destination, dataset)
                elif stage == 'rmt': info = rmt(returns, destination, args)
                elif stage == 'structure': info = structure(returns, destination, args, dataset, metadata)
                elif stage == 'industry': info = industry(returns, destination, args, dataset)
                elif stage == 'scales': info = scales(returns, destination, args)
                elif stage == 'modes': info = modes(panel, destination, args, prices)
            info = {'status': 'complete', **info, 'elapsed_seconds': round(time.monotonic()-start, 2),
                    'parameters': vars(args), 'code_sha256': record['code_sha256']}
            record['stages'][stage] = info
            write_json(destination / 'summary.json', info)
            print(f"{stage}: {info['status']} ({info['elapsed_seconds']}秒)", flush=True)
        except Exception as exc:
            record['stages'][stage] = {'status': 'failed', 'error': str(exc)}
            write_json(output / 'run_manifest.json', record)
            write_json(history_path, record)
            raise
        finally:
            plt.close('all')
        write_json(output / 'run_manifest.json', record)
        write_json(history_path, record)
    print(f'完成，结果：{output}', flush=True)
