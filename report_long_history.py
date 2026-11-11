#!/usr/bin/env python3
"""Summarize a completed seven-stage run, add long-history checks, and package it.

Usage: python report_long_history.py hs300_long100
The source prices and the seven original engines' outputs are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from PIL import Image
from pipeline import configure_plots, ra, save, write_json


def table(frame, columns=None, digits=3):
    frame = frame if columns is None else frame[columns]
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return '' if pd.isna(value) else f'{value:.{digits}f}'
        return str(value).replace('|', '/')
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)]])


def performance(values, annual_days=242):
    values = np.asarray(values, float)
    wealth = np.r_[1., np.cumprod(1+values)]
    return {'days': len(values), 'total_return': wealth[-1]-1,
            'sharpe_rf0': values.mean()/values.std(ddof=1)*np.sqrt(annual_days),
            'max_drawdown': (wealth/np.maximum.accumulate(wealth)-1).min()}


def annual_extensions(x, daily, destination):
    annual, exclusions, matched, strategy_rows = [], [], [], []
    for year, sample in x.groupby(x.index.year):
        valid = sample.std(ddof=1) > 1e-12
        for ticker in valid.index[~valid]:
            exclusions.append({'year': year, 'ticker': ticker, 'reason': 'zero annual return variance'})
        if valid.sum() >= 2:
            row = ra.matrix_summary(sample.loc[:, valid].to_numpy(), str(year))
            row.update(year=year, start=str(sample.index[0].date()), end=str(sample.index[-1].date()),
                       omitted_constant_assets=int((~valid).sum()))
            annual.append(row)
        end = sample.index[-1]
        for length in [120, 242, 484]:
            sub = x.loc[:end].tail(length)
            if len(sub) == length and (sub.std(ddof=1) > 1e-12).all():
                row = ra.matrix_summary(sub.to_numpy(), f'{year}_{length}')
                row.update(end=str(end.date()), window=length)
                matched.append(row)
    for year, sample in daily.groupby(daily.index.year):
        for strategy in sample:
            if len(sample) > 1:
                strategy_rows.append({'year': year, 'strategy': strategy,
                                      **performance(sample[strategy])})
    annual = pd.DataFrame(annual)
    annual.to_csv(destination / 'annual_rmt_2002_onward.csv', index=False)
    pd.DataFrame(exclusions, columns=['year', 'ticker', 'reason']).to_csv(
        destination / 'annual_zero_variance_exclusions.csv', index=False)
    pd.DataFrame(matched).to_csv(destination / 'all_year_end_matched_windows.csv', index=False)
    strategy_rows = pd.DataFrame(strategy_rows)
    strategy_rows.to_csv(destination / 'annual_portfolio_performance.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(annual.year, annual.lambda_max_fraction*100, 'o-', label='Leading mode')
    axes[0].plot(annual.year, annual.avg_off_diagonal_correlation*100, 's--', label='Mean pair correlation')
    axes[0].set(ylabel='Percent', title='Full historical coverage: yearly cross-stock structure')
    axes[0].legend()
    axes[1].bar(annual.year, annual.high_outlier_count)
    axes[1].set(xlabel='Year (first/last years may be partial)', ylabel='Modes above yearly MP upper edge')
    save(fig, destination / 'figures', '01_annual_market_structure')
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for strategy in ['raw_gmv', 'rmt_clipped_gmv']:
        sub = strategy_rows[strategy_rows.strategy == strategy]
        axes[0].plot(sub.year, sub.total_return*100, 'o-', label=strategy)
        axes[1].plot(sub.year, sub.max_drawdown*100, 'o-', label=strategy)
    axes[0].axhline(0, color='grey', lw=.7)
    axes[0].set(ylabel='Calendar-year return (%)', title='Portfolio outcomes in every available backtest year')
    axes[1].set(xlabel='Year (first/last years may be partial)', ylabel='Within-year maximum drawdown (%)')
    axes[0].legend()
    save(fig, destination / 'figures', '02_annual_portfolio_outcomes')
    return annual, strategy_rows


def matched_shift_null(x, existing_metrics, destination, repetitions):
    """Circularly shift each stock's tau-return independently, retaining its own ACF.

    These are pointwise extreme-eigenvalue reference intervals, not simultaneous
    confidence bounds over all 250 horizons, and not an MP analytic density.
    """
    rng = np.random.default_rng(20261110)
    raw = x.to_numpy()
    profile = np.vstack([np.zeros(raw.shape[1]), raw.cumsum(axis=0)])
    summaries, replicates = [], []
    max_tau = int(existing_metrics.tau_days.max())
    for tau in range(1, max_tau+1):
        returns = profile[tau:]-profile[:-tau]
        z = (returns-returns.mean(axis=0))/returns.std(axis=0, ddof=0)
        t, n = z.shape
        eig = np.linalg.eigvalsh(z.T@z/t)
        lower, upper = [], []
        base = np.arange(t)[:, None]
        for repeat in range(repetitions):
            shift = rng.integers(0, t, n)
            shifted = z[(base+shift) % t, np.arange(n)]
            vals = np.linalg.eigvalsh(shifted.T@shifted/t)
            lower.append(vals[0]); upper.append(vals[-1])
            replicates.append({'tau': tau, 'replicate': repeat+1,
                               'lambda_min': vals[0], 'lambda_max': vals[-1]})
        lo, hi = np.quantile(lower, .025), np.quantile(upper, .975)
        summaries.append({'tau': tau, 'observations': t, 'null_lower_q025': lo,
                          'null_upper_q975': hi, 'observed_min': eig[0], 'observed_max': eig[-1],
                          'below': int((eig < lo).sum()), 'inside': int(((eig >= lo)&(eig <= hi)).sum()),
                          'above': int((eig > hi).sum())})
        if tau % 25 == 0:
            print(f'  independent circular shifts: tau {tau}/{max_tau}, {repetitions} replicates each', flush=True)
    frame = pd.DataFrame(summaries)
    frame.to_csv(destination / 'circular_shift_envelope.csv', index=False)
    pd.DataFrame(replicates).to_csv(destination / 'circular_shift_null_extremes.csv', index=False)
    plot_shift_reference(frame, destination, repetitions)
    return frame


def plot_shift_reference(frame, destination, repetitions):
    observed = pd.read_csv(destination.parent / 'modes/tables/eigenvalues.csv')
    bounds = frame.set_index('tau')
    observed['classification'] = np.where(
        observed.eigenvalue > observed.tau_days.map(bounds.null_upper_q975), 'above',
        np.where(observed.eigenvalue < observed.tau_days.map(bounds.null_lower_q025), 'below', 'inside'))
    observed.to_csv(destination / 'circular_shift_observed_classification.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    for name, color in [('inside', '#aab5bf'), ('below', '#245894'), ('above', '#CC6646')]:
        sub = observed[observed.classification == name]
        axes[0].scatter(sub.tau_days, sub.eigenvalue, s=1, color=color, label=name)
    axes[0].fill_between(frame.tau, frame.null_lower_q025, frame.null_upper_q975, color='#187968', alpha=.1)
    axes[0].plot(frame.tau, frame.null_lower_q025, color='#187968', lw=2)
    axes[0].plot(frame.tau, frame.null_upper_q975, color='#187968', lw=2)
    axes[0].set(yscale='log', ylabel='Eigenvalue', title=f'Overlapping returns: pointwise circular-shift reference ({repetitions} replicates)')
    # Decimal log-axis labels avoid missing superscript-minus glyphs in CJK fonts.
    from matplotlib.ticker import FuncFormatter
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    axes[0].legend(ncol=3)
    for name, color in [('below', '#245894'), ('above', '#CC6646')]:
        axes[1].plot(frame.tau, frame[name], color=color, label=name)
    axes[1].set(xlabel='Return horizon tau (trading observations)', ylabel='Observed modes outside')
    axes[1].legend()
    save(fig, destination / 'figures', '03_circular_shift_null_envelope')
    return frame


def longer_tau_time_modes(x, destination):
    raw = x.to_numpy(); n = x.shape[1]
    profile = np.vstack([np.zeros(n), raw.cumsum(axis=0)])
    summary = []
    z1 = (raw-raw.mean(axis=0))/raw.std(axis=0, ddof=0)
    _, v1 = np.linalg.eigh(z1.T@z1/len(z1))
    v1 = v1[:, ::-1]
    for tau in [1, 5, 20, 60, 120, 250]:
        piece = profile[tau:]-profile[:-tau]
        z = (piece-piece.mean(axis=0))/piece.std(axis=0, ddof=0)
        vals, vectors = np.linalg.eigh(z.T@z/len(z))
        vals, vectors = vals[::-1], vectors[:, ::-1]
        scores = z@vectors[:, :10]/np.sqrt(vals[:10])
        for j in range(10):
            if (j == 0 and scores[:, j]@z.mean(axis=1) < 0) or (
                    j > 0 and vectors[np.argmax(abs(vectors[:, j])), j] < 0):
                scores[:, j] *= -1; vectors[:, j] *= -1
            summary.append({'tau': tau, 'rank': j+1, 'eigenvalue': vals[j], 'variance_share': vals[j]/n,
                            'same_rank_loading_overlap_with_tau1': abs(vectors[:, j]@v1[:, j]),
                            'best_tau1_rank': int(np.argmax(abs(v1.T@vectors[:, j]))+1),
                            'best_tau1_overlap': np.max(abs(v1.T@vectors[:, j]))})
        dates = x.index[tau-1:]
        pd.DataFrame(scores, index=dates, columns=[f'mode_{j+1:02d}' for j in range(10)]).to_csv(
            destination / f'tau{tau}_top10_time_modes.csv', index_label='Date')
        pd.DataFrame(vectors[:, :10], index=x.columns,
                     columns=[f'mode_{j+1:02d}' for j in range(10)]).to_csv(
                         destination / f'tau{tau}_top10_loadings.csv', index_label='ticker')
        fig, axes = plt.subplots(10, 1, figsize=(16, 20), sharex=True, sharey=True)
        limit = np.ceil(np.max(abs(scores)))
        for j, ax in enumerate(axes):
            ax.plot(dates, scores[:, j], color='#245894', lw=.45)
            ax.axhline(0, color='grey', lw=.5)
            ax.set(ylim=(-limit, limit), ylabel=f'Mode {j+1}',
                   title=f'lambda={vals[j]:.3f}; variance share={vals[j]/n:.1%}')
            ax.title.set_fontsize(9)
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(4))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.suptitle(f'Tau={tau}: top 10 KL time modes, each in its own standard deviations', fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, .98))
        save(fig, destination / 'figures', f'04_tau{tau}_top10_time_modes')
    frame = pd.DataFrame(summary); frame.to_csv(destination / 'longer_tau_top10_summary.csv', index=False)
    return frame


def export_matrices_and_check(x, output, destination, manifest):
    n, t = x.shape[1], len(x)
    result = ra.rmt_decomposition(x.to_numpy())
    components = ra.correlation_components(result)
    covariance = np.cov(x.to_numpy(), rowvar=False)
    matrices = {'correlation': result.correlation, 'daily_log_covariance': covariance,
                'market_removed_standardized_covariance': result.correlation-components['market'], **components}
    vol = x.std(ddof=1).to_numpy()
    matrices['market_removed_daily_log_covariance'] = matrices['market_removed_standardized_covariance']*vol[:, None]*vol[None, :]
    matrices['rmt_clipped_correlation'] = ra.clipped_correlation(result)
    matrices['rmt_clipped_daily_log_covariance'] = matrices['rmt_clipped_correlation']*vol[:, None]*vol[None, :]
    for name, matrix in matrices.items():
        pd.DataFrame(matrix, index=x.columns, columns=x.columns).to_csv(destination / f'{name}.csv', index_label='ticker')
    np.savez_compressed(destination / 'full_sample_matrices.npz', **matrices,
                        tickers=np.asarray(x.columns, dtype=str))
    eig, vectors = result.eigenvalues[::-1], result.eigenvectors[:, ::-1].copy()
    reference = pd.read_csv(output / 'modes/tables/tau1_top10_loadings.csv').set_index('ticker').reindex(x.columns)
    for j in range(n):
        sign = vectors[:, j]@reference[f'mode_{j+1:02d}'].to_numpy() if j < 10 else vectors[np.argmax(abs(vectors[:, j])), j]
        if sign < 0:
            vectors[:, j] *= -1
    z = (x.to_numpy()-x.to_numpy().mean(axis=0))/x.to_numpy().std(axis=0, ddof=0)
    scores = z@vectors/np.sqrt(eig)
    labels = [f'mode_{j+1:02d}' for j in range(n)]
    pd.DataFrame(vectors, index=x.columns, columns=labels).to_csv(destination / 'tau1_all_loadings.csv', index_label='ticker')
    pd.DataFrame(scores, index=x.index, columns=labels).to_csv(destination / 'tau1_all_time_modes.csv.gz', index_label='Date', compression='gzip')
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)
    check('all_seven_stages_complete', all(v['status'] == 'complete' for v in manifest['stages'].values()) and len(manifest['stages']) == 7)
    check('strictly_ordered_unique_dates', x.index.is_unique and x.index.is_monotonic_increasing)
    check('finite_input_returns', np.isfinite(x.to_numpy()).all())
    check('no_full_universe_flat_rows', not (x.abs() <= 1e-14).all(axis=1).any())
    check('correlation_trace_equals_N', np.isclose(eig.sum(), n, atol=1e-8))
    check('correlation_diagonal_is_one', np.allclose(np.diag(result.correlation), 1))
    check('correlation_positive_definite', eig[-1] > 0)
    check('matrix_components_reconstruct_C', np.allclose(components['market']+components['group']+components['noise'], result.correlation, atol=1e-9))
    check('all_time_modes_mean_zero', np.max(abs(scores.mean(axis=0))) < 1e-10)
    check('all_time_modes_orthonormal_covariance', np.allclose(scores.T@scores/t, np.eye(n), atol=1e-8))
    expected = pd.read_csv(output / 'basic/eigenvalues.csv').eigenvalue.to_numpy()
    check('spectrum_agrees_between_basic_and_rmt', np.allclose(eig, expected, atol=1e-9))
    spectral = pd.read_csv(output / 'modes/tables/eigenvalues.csv')
    check('all_250_tau_spectra_present', set(spectral.tau_days) == set(range(1, 251)) and len(spectral) == 250*n)
    check('all_tau_trace_equals_N', np.allclose(spectral.groupby('tau_days').eigenvalue.sum(), n, atol=1e-7))
    top = pd.read_csv(output / 'modes/tables/tau1_top10_time_modes.csv').filter(regex='^mode_').to_numpy()
    check('all_mode_export_matches_existing_top10', np.allclose(scores[:, :10], top, atol=1e-8))
    weights = pd.read_csv(output / 'rmt/tables/12_latest_portfolio_weights.csv')
    check('portfolio_weights_sum_one', np.allclose(weights.groupby('strategy').weight.sum(), 1))
    check('portfolio_weights_within_cap', weights.weight.min() >= -1e-10 and weights.weight.max() <= .05+1e-8)
    rebalance = pd.read_csv(output / 'rmt/tables/11_portfolio_rebalances.csv')
    check('all_rebalances_respect_weight_cap', rebalance.maximum_weight.max() <= .05+1e-8)
    risk = pd.read_csv(output / 'modes/tables/risk_forecast_folds.csv')
    check('risk_training_precedes_testing', (pd.to_datetime(risk.train_end_date) < pd.to_datetime(risk.test_start_date)).all())
    for window, sample in risk[risk.method == 'sample'].groupby('window_days'):
        sample = sample.sort_values('test_start_index')
        check(f'risk_test_folds_nonoverlap_{window}', (np.diff(sample.test_start_index) >= sample.test_days.iloc[0]).all())
    check('risk_scores_finite', np.isfinite(risk.select_dtypes(include='number')).all().all())
    scale = np.load(output / 'scales/scale_matrices.npz')
    check('all_scale_matrices_psd', all(np.linalg.eigvalsh(scale[key]).min() >= -1e-9 for key in scale.files))
    code_changes = [name for name, digest in manifest['code_sha256'].items()
                    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest]
    check('engine_code_matches_run_manifest', not code_changes)
    return checks


def long_rolling_figures(output, destination):
    rolling = pd.read_csv(output / 'modes/tables/rolling_mode_structure.csv', parse_dates=['end_date'])
    groups = pd.read_csv(output / 'modes/tables/residual_group_rolling_metrics.csv', parse_dates=['end_date'])
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    specs = [('market_variance_share', 'Market-mode variance share'),
             ('market_raw_daily_volatility_percent', 'Equal-weight daily log volatility (%)'),
             ('market_overlap_disjoint', 'Market loading overlap: disjoint past'),
             ('top10_subspace_overlap_disjoint', 'Top-10 subspace overlap: disjoint past')]
    for window, color in [(500, '#245894'), (750, '#db713c')]:
        sub = rolling[rolling.window_days == window]
        for ax, (column, title) in zip(axes, specs):
            ax.plot(sub.end_date, sub[column], color=color, label=f'{window} days', lw=1.1)
            ax.set_ylabel(title, fontsize=9)
        sub = groups[groups.window_days == window]
        axes[-1].plot(sub.end_date, sub.ari_vs_disjoint_window, color=color, label=f'{window} days', lw=1.1)
    axes[-1].set_ylabel('Residual-group ARI: disjoint past', fontsize=9)
    axes[0].legend(ncol=2)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(4))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.suptitle('Rolling structure across the complete available history; dynamic vertical ranges', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    save(fig, destination / 'figures', '05_complete_history_rolling_structure')
    risk = pd.read_csv(output / 'modes/tables/risk_forecast_folds.csv', parse_dates=['test_end_date'])
    fig, ax = plt.subplots(figsize=(14, 5))
    sub = risk[risk.window_days == 500]
    reference = sub[sub.method == 'sample']
    ax.plot(reference.test_end_date, reference.equal_weight_realized_vol_percent, color='black', lw=1.2, label='20-day realized')
    for method in ['sample', 'shrink_20pct', 'clip_top10']:
        piece = sub[sub.method == method]
        ax.plot(piece.test_end_date, piece.equal_weight_forecast_vol_percent, lw=.9, label=method)
    ax.set(ylabel='Daily volatility (%)', title='Complete history: out-of-sample equal-weight risk forecasts, 500-day training')
    ax.xaxis.set_major_locator(mdates.YearLocator(4)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(ncol=4)
    save(fig, destination / 'figures', '06_complete_history_risk_forecast')
    summary = pd.read_csv(output / 'modes/tables/risk_method_summary.csv')
    methods = summary.method.drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for window, offset, color in [(500, -.2, '#245894'), (750, .2, '#db713c')]:
        piece = summary[summary.window_days == window].set_index('method').reindex(methods)
        locations = np.arange(len(methods))+offset
        values = piece.gaussian_score_improvement_vs_sample
        errors = np.maximum(0, np.vstack([values-piece.score_improvement_ci025,
                                          piece.score_improvement_ci975-values]))
        axes[0].bar(locations, values, width=.38, yerr=errors, capsize=3, color=color, label=f'{window}-day train')
        axes[1].bar(locations, piece.portfolio_vol_mae_pp, width=.38, color=color, label=f'{window}-day train')
    axes[0].axhline(0, color='grey', lw=.7)
    axes[0].set(ylabel='Sample score - method score', title='Gaussian covariance-score improvement (higher is better)')
    axes[1].set(ylabel='Percentage points of daily volatility', title='Portfolio volatility MAE (lower is better)')
    for ax in axes:
        ax.set_xticks(range(len(methods)), methods, rotation=35, ha='right')
        ax.legend()
    fig.tight_layout()
    # Replace the legacy fixed-range drawing, whose 0.7 pp ceiling clips this run.
    save(fig, output / 'modes/figures', '17_risk_method_comparison')


def correct_generic_price_notes(output, data_summary):
    if 'adjusted close' not in data_summary.get('prices', '').lower():
        return
    replacements = {
        '原始收盘价的复权和成分历史规则未由输入数据说明。':
        '本次输入为 Yahoo 拆股/分红调整收盘价，保留原始响应和上市日期核验；未提供动态成分历史。',
        '输入价格是否复权、停牌与成分历史规则均未获确认。':
        '本次价格为 Yahoo 拆股/分红调整收盘价；已剔除全体零成交且价格不变的供应商日期，单股停牌记录仍保留，动态成分历史未提供。'}
    for name in ['mode_structure_report.md', 'residual_and_risk_report.md']:
        path = output / 'modes' / name
        text = path.read_text()
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text)


def make_report(dataset, output, destination, manifest, x, annual, annual_portfolio, envelope, larger_tau, checks):
    read = lambda path: pd.read_csv(output/path)
    meta = lambda path: json.loads((output/path).read_text())
    matrix = read('rmt/tables/01_full_sample_summary.csv').iloc[0]
    eig = read('rmt/tables/02_full_sample_eigenvalues.csv')
    perf = read('rmt/tables/10_portfolio_performance.csv')
    basic = read('basic/full_period_metrics.csv')
    rmt = meta('rmt/summary.json')
    structure = meta('structure/summary.json')
    industry = meta('industry/rmt_tables/19_industry_alignment_tests.json')
    modes = meta('modes/mode_structure_metadata.json')
    residual = meta('modes/residual_and_risk_metadata.json')
    cfg = meta('modes/config.json')
    risk = read('modes/tables/risk_method_summary.csv')
    tails = read('structure/tables/tail_probabilities.csv')
    acf = read('structure/tables/autocorrelation.csv')
    scales = read('scales/scale_dma_comparison.csv')
    reliability = read('structure/tables/chronological_mode_reliability.csv')
    top10 = read('modes/tables/tau1_top10_loading_summary.csv')
    group_tests = read('modes/tables/residual_group_holdout_validation.csv')
    rolling = read('modes/tables/rolling_mode_structure.csv')
    flags = read('structure/tables/data_quality_flags.csv')
    benchmark = read('structure/tables/official_index_comparison.csv')
    common = read('modes/tables/mode_metrics.csv')
    observed = common[common.source == 'observed']
    selected_tau = [1, 5, 20, 60, 120, 250]
    scan = observed[observed.tau_days.isin(selected_tau)].copy()
    null = common[common.source != 'observed'].groupby(['tau_days', 'mode']).agg(
        null_alpha_median=('alpha', 'median'), null_alpha_q025=('alpha', lambda v: v.quantile(.025)),
        null_alpha_q975=('alpha', lambda v: v.quantile(.975)))
    comparison = scan.merge(null, on=['tau_days', 'mode'], how='left')
    comparison.to_csv(destination / 'observed_and_common_shuffle_spectral_comparison.csv', index=False)
    peak = rolling.loc[rolling.market_variance_share.idxmax()]
    low = rolling.loc[rolling.market_variance_share.idxmin()]
    early = int((x.index < cfg['split_date']).sum()); late = len(x)-early
    n = x.shape[1]
    p = perf.set_index('strategy')
    sharpe_delta = p.loc['rmt_clipped_gmv', 'sharpe_rf0']-p.loc['raw_gmv', 'sharpe_rf0']
    ci = rmt['sharpe_difference_ci95']
    report = [f'# {dataset}：长历史 {n} 股完整分析', '',
              '## 1. 本次分析口径与完成情况', '',
              f'- 数据区间：价格 {manifest["price_start"]} 至 {manifest["price_end"]}；收益 {x.index[0].date()} 至 {x.index[-1].date()}。',
              f'- 使用全部 {n} 只选定股票；原始共同价格记录 {manifest["source_price_dates"]} 行，剔除 {manifest["excluded_nontrading_dates"]} 个全体零成交且复权价格不变的供应商日期后，得到 {manifest["price_dates"]} 个价格日、{len(x)} 个收益日。清单见 [excluded_nontrading_dates.csv](excluded_nontrading_dates.csv)。',
              '- 选股来自原有固定 288 股研究池，以较早的有效上市后历史选出 100 股；它不是逐期真实沪深300成分股组合。价格是 Yahoo 拆股/分红调整收盘价；未前向填充、未拼接不同供应商价格，原始响应与上市日期核验保存在 data/hs300_long100/。',
              '- 七个模块均正式完成：basic、audit、rmt、structure、industry、scales、modes。主 τ 扫描为 1–250，无 HF1MA 平滑。scales 另保留仓库既有的中心化去趋势/DMA结构对照；这些变换不进入回测。',
              '- 随机重复：主随机对照100次；行业和群组标签置换999次；组合差异块重采样2000次；模态移动块重采样200次；风险比较块重采样1000次。',
              '- 全部图为 PNG；数值表为 CSV/CSV.GZ，矩阵同时导出 NPZ；运行参数、耗时和引擎代码哈希见 [run_manifest.json](run_manifest.json)。', '',
              '## 2. 市场模态与随机矩阵基线', '',
              f'标准化日对数收益相关矩阵的最大特征值为 **{matrix.lambda_max:.4f}**，解释 **{matrix.lambda_max_fraction:.2%}** 方差；平均非对角相关系数为 **{matrix.avg_off_diagonal_correlation:.4f}**。第一模态的有效参与股票数 PR={matrix.largest_mode_standard_PR:.2f}/{n}，说明载荷广泛分布在股票中。课程 Γ=3PR，不能把 Γ 当作股票数。', '',
              f'MP 参照区间为 **[{matrix.lambda_minus:.4f}, {matrix.lambda_plus:.4f}]**；高于上界 **{int(matrix.high_outlier_count)}** 个，低于下界 **{int(matrix.low_outlier_count)}** 个，上方模态合计解释 **{matrix.high_outlier_strength_fraction:.2%}** 方差。这是独立、同方差基线的偏离，不能把上方模态数直接解释为行业数，也不能断言区间内全部为噪声。', '',
              table(eig.head(10), ['rank_descending', 'eigenvalue', 'strength_fraction', 'standard_participation_ratio']), '',
              f'逐股独立打乱的100次日频最大特征值第95百分位为 {scales.iloc[0].null_max_q95:.4f}，远低于观测最大特征值。补充目录提供相关/协方差、市场/组/其余部分、去市场矩阵和谱去噪矩阵，以及全部100个日频载荷与时间分量。相关矩阵分量在归一化之前满足市场+组+其余=原矩阵；分别重新归一化后不能再相加。', '',
              '![经验谱与MP基线](basic/figures/01_ranked_spectrum.png)', '',
              '## 3. 尾部、波动聚集与数据质量', '',
              f'合并标准化收益的超额峰度为 **{structure["pooled_excess_kurtosis"]:.2f}**。尾部概率如下：', '',
              table(tails.assign(empirical=tails.empirical.map(lambda v: f'{v:.6g}'),
                                 gaussian=tails.gaussian.map(lambda v: f'{v:.6g}'))), '',
              f'逐股自相关的中位数：滞后1日，收益为 {acf.iloc[0].return_median:.4f}，绝对收益为 {acf.iloc[0].abs_return_median:.4f}；滞后20日，绝对收益为 {acf.loc[acf.lag == 20, "abs_return_median"].iloc[0]:.4f}。绝对收益更持久，支持波动聚集的描述，不等同于收益方向可预测。', '',
              f'收益异常记录和单股零收益连续段见 audit/tables。仓库“出现连续6个零收益日或单日简单收益绝对值>25%即标记”的全样本回顾规则标记 **{int(flags.retrospective_flag.sum())}/{n}** 股；在约25年的历史上，这个阈值过于严苛。若剔除后不足2股，data_quality_sensitivity.csv 只保留 baseline，不能声称完成了清洁股票子集的稳健性比较。该规则没有用于回测选股。', '',
              '## 4. 行业信息与分组', '',
              f'按超过MP上界、扣除第一市场模态后的 group 矩阵做平均连接层次聚类，轮廓系数选择 **{industry["selected_cluster_count"]} 组**，silhouette={industry["selected_silhouette"]:.3f}。它与另一个残差谱聚类中预先指定的8组是不同实验。行业标签对应本项目保存的2026年快照，不能当作2002年以来的历史行业标签。', '',
              f'- 一级行业对齐：purity={industry["purity"]:.3f}，ARI={industry["adjusted_rand_index"]:.3f}，NMI={industry["normalized_mutual_information"]:.3f}。',
              f'- 一级行业组内−组间 group 相关差={industry["l1_within_minus_between"]:.3f}，标签置换 p={industry["l1_permutation_p_one_sided"]:.3f}。',
              f'- 同二级行业相对“同一级但不同二级”的差={industry["l2_same_minus_same_l1_different_l2"]:.3f}，p={industry["l2_permutation_p_one_sided"]:.3f}。',
              f'- 对所有{n}个载荷做行业解释度置换并校正多重比较后，MP区间内仍有 **{structure["within_mp_industry_q05_count"]}** 个模态满足 q<0.05。这说明固定MP边界不是行业信息的完整筛选器。', '',
              '![行业对齐统计](industry/rmt_figures/14_industry_alignment_statistics.png)', '',
              '## 5. 模态的时间稳定性与去市场后的群组', '',
              f'按 {cfg["split_date"]} 分割：早期{early}日、晚期{late}日。市场载荷绝对内积为 **{modes["market_early_late_overlap"]:.3f}**，前十维子空间重合度为 **{modes["top10_early_late_subspace_overlap"]:.3f}**。第一模态较稳定，后续模态会换序和混合；例如第二模态在晚期最相似的模态排名为{int(top10.iloc[1].early_late_best_late_rank)}，绝对内积{top10.iloc[1].early_late_best_abs_overlap:.3f}。另一个等长前后半样本实验中，市场载荷重合度为{reliability.iloc[0].best_overlap:.3f}；它的分割时间与2023留出实验不同。', '',
              f'500/750日滚动、每20日更新，共{len(rolling)}个窗口；市场方差份额从{low.market_variance_share:.1%}到{peak.market_variance_share:.1%}。最大值出现在以{peak.end_date}结束的{int(peak.window_days)}日窗口，最小值出现在以{low.end_date}结束的{int(low.window_days)}日窗口。这里是整个窗口的统计量，不能归因于结束当天。', '',
              f'固定8组残差谱聚类：早晚独立分组 ARI=**{residual["early_late_group_ari"]:.3f}**；早期分组用于晚期数据，组内−组间差=**{residual["early_groups_on_late_contrast"]:.4f}**，保持组大小的标签置换 p=**{residual["early_groups_on_late_permutation_p"]:.3f}**。组内联系在留出期仍存在，同时组员结构也发生变化。此处“残差相关”数值实际为 C−λ₁v₁v₁ᵀ，即以原股票标准差为单位的残差协方差，没有再次把残差对角元归一化为1。', '',
              table(group_tests), '',
              '![完整历史滚动结构](supplement/figures/05_complete_history_rolling_structure.png)', '',
              '## 6. τ=1–250、多尺度与功率谱', '',
              '主扫描在每个τ独立构造重叠τ日对数收益并逐股标准化。时间坐标 aₖ(t)=vₖᵀz(t)/√λₖ，均值0、时间方差1；纵轴单位为该模态自身的全样本标准差。单位L2的论文时间分量为 bₖ=aₖ/√T。特征向量正负号本来任意，第一模态方向与等权同涨同跌对齐。', '',
              '100股可计算最大、第二大和第44小特征值对应分量；**第144小模态不存在**，没有替换成其他模态。全部τ的100个特征值保存在 modes/tables/eigenvalues.csv；补充导出τ=1、5、20、60、120、250的前十个时间模态和载荷。', '',
              table(larger_tau[larger_tau['rank'] == 1], ['tau', 'eigenvalue', 'variance_share', 'same_rank_loading_overlap_with_tau1']), '',
              'τ变大时，相邻重叠收益共享更多日收益；时间曲线变平滑、低频功率提高可由这种构造本身产生，不能直接解释为确定性增强。这里保留三种不同对照：', '',
              '1. **共同日期置换**：同一排列用于所有股票，保留日频同时相关，破坏日期顺序；检验时间顺序的影响。对应谱斜率的观测/随机中位数和区间见 supplement/observed_and_common_shuffle_spectral_comparison.csv。',
              '2. **逐股独立打乱**：scales 模块重新应用同一尺度变换后比较，破坏单股日期依赖和跨股同步。',
              f'3. **逐股独立循环平移**：补充对照在每个τ保留各股该τ收益的边际分布和循环自相关，同时打散跨股日期对齐；每τ {int(len(pd.read_csv(destination / "circular_shift_null_extremes.csv"))/len(envelope))}次。下界为随机最小特征值的2.5%分位，上界为随机最大特征值的97.5%分位，是逐τ极值参考区间，不是解析MP上下界、排序特征值逐一置信区间，也没有对250次τ比较做同时覆盖保证。', '',
              table(envelope[envelope.tau.isin(selected_tau)], ['tau', 'null_lower_q025', 'null_upper_q975', 'observed_max', 'above', 'below']), '',
              '![与保持单股时间结构的随机对照比较](supplement/figures/03_circular_shift_null_envelope.png)', '',
              '在scales的63日非重叠收益中，只有约95个窗口，矩阵秩低于100；零特征值是样本数量限制，不能当作额外的经济信息。中心DMA为回顾性的结构对照。谱斜率使用原仓库相对频带做对数线性拟合；拟合值、R²与全部随机样本均已保存，没有把拟合当作幂律或可预测性的证明。', '',
              '## 7. 最小方差组合：全部样本外回测', '',
              '两个原项目的回测口径均保留，必须分别比较同一模块内部结果：basic使用简单收益估计、252日年化、首次建仓计成本；rmt使用对数收益估计、242日年化、首次建仓不计成本。两者均504日训练、21日再平衡、仅做多、目标单股权重上限5%、换手成本10bps，持有期间权重随收益漂移。等权再平衡是基准，未将股票组合误称为可直接买入的ETF。', '',
              '本节数值表的收益、波动和回撤以小数表示，例如0.200为20.0%；Sharpe无量纲。', '',
              '**basic（原基础复现口径）：**', '', table(basic), '',
              '**rmt（原handoff口径）：**', '',
              table(perf, ['strategy', 'start_date', 'end_date', 'annualized_geometric_return_net',
                           'annualized_volatility', 'sharpe_rf0', 'max_drawdown', 'average_rebalance_turnover']), '',
              f'RMT clip相对原样本GMV的 Sharpe 差为 **{sharpe_delta:+.5f}**；21日配对循环块bootstrap的95%描述区间为 **[{ci[0]:+.4f}, {ci[1]:+.4f}]**。区间包含0，不能据此声称Sharpe稳定改善。波动和回撤变化应结合数值表观察；去噪没有消除系统性暴跌。', '',
              '![组合净值与回撤](rmt/figures/09_portfolio_backtest.png)', '',
              '原仓库预选2022/2024/2025压力窗口和按高波动/下跌日期的条件比较均保留。为覆盖更早历史，补充逐年组合收益、Sharpe和年内最大回撤，包含2008、2015等年份；年内回撤从该年初净值1重置，不等于跨年累计最大回撤。', '',
              table(annual_portfolio[(annual_portfolio.year.isin([2008, 2015, 2022])) & annual_portfolio.strategy.isin(['raw_gmv', 'rmt_clipped_gmv'])]), '',
              '## 8. 样本外协方差预测：201个固定组合、六种估计器', '',
              '使用全体等权1个、随机10股等权100个、随机30股等权100个组合；各方法使用同一组固定权重。以过去500/750日训练，预测随后20日，测试块不重叠。比较原样本、对角、20%/50%对角收缩、保留前1/10模态的谱截断。Gaussian伪负对数评分、波动MAE和方差QLIKE均为越小越好，校准比理想值约1。', '',
              table(risk, ['window_days', 'method', 'n_folds', 'gaussian_pseudo_log_score_per_stock',
                           'portfolio_vol_mae_pp', 'portfolio_vol_bias_pp', 'median_realized_to_forecast_variance']), '',
              '固定20%收缩在两个训练窗口中均降低平均波动预测MAE；误差改善的块bootstrap区间见 risk_method_summary.csv。只保留对角项明显低估共同风险；谱截断可改善Gaussian评分，但不意味着波动预测误差同比例改善。多个方法与指标同时比较属于探索性检验，区间没有做全方法多重校正，不能从中挑选最高分再宣称样本外最优。', '',
              '## 9. 得到什么、哪些结果不可获得', '',
              '- 市场共同波动广泛、跨时段较稳定；行业/群组结构可辨认，但中间模态的身份和分组会变化。',
              '- 大τ的平滑和低频集中必须与匹配的随机对照比较，不能仅凭功率谱斜率推断确定性。',
              '- 风险估计改善与投资表现改善是两个问题；本样本没有显示RMT clip相对原样本GMV的明确Sharpe优势。',
              '- 全部样本是事后固定长历史股票池，存在存活者与长历史筛选偏差；所得高历史回报不代表真实动态指数或可实现的历史投资收益。单股停牌、供应商复权和长期公司变化仍会影响结果。',
              f'- 实际沪深300指数比较仅覆盖本地基准交集：{benchmark.iloc[0, 0]} 至 {benchmark.iloc[-1, 0]}，{structure["official_index_days"]}日，第一模态与指数日收益相关系数{structure["official_index_pc_correlation"]:.3f}；不能扩展为全25年的指数验证。',
              '- 没有第144小模态；没有这100股的独立历史return/zscore三套档案，audit中三档案交叉重建不适用；没有逐期成分或历史行业标签，不能做无存活者偏差的历史指数复现。',
              '- 库中部分旧报告与事件标签针对原短区间；本次以此汇总、实际数值表、完整年份补充图和运行清单为准。scipy对当前NumPy版本发出兼容范围警告；实际七模块均完成，并通过下述数值一致性校验，软件版本保存在运行清单中。', '',
              f'## 10. 校验与复跑', '',
              f'通过 **{len(checks)} 项**结果校验：矩阵迹/对角/正定性、分量重构、全部100个时间分量正交性、与原前10分量一致性、250个τ完整性、两引擎经验谱一致、回测权重约束、训练先于测试、测试块不重叠、尺度矩阵半正定、代码哈希及PNG可读取性。详见 [supplement/validation.json](supplement/validation.json)。', '',
              '在项目根目录运行：', '', '```bash', 'python run.py hs300_long100 --analysis all',
              'python report_long_history.py hs300_long100', 'python verify_data.py', '```', '',
              '第一条复跑七模块；第二条生成长样本补充实验、这份报告、图表目录、文件哈希清单和完整压缩包；第三条验证原始数据未被修改。正式重复次数保持默认。', '',
              '## 11. 全部结果入口', '',
              '- [离线图表与数据目录](index.html)：按模块浏览全部PNG，点击对应CSV/NPZ/报告。',
              '- [文件清单与SHA-256](artifact_manifest.csv)：列出本次结果文件；清单自身、打包日志和压缩包不纳入递归哈希。',
              '- [长历史完整年份RMT](supplement/annual_rmt_2002_onward.csv)、[逐年组合结果](supplement/annual_portfolio_performance.csv)、[所有年末匹配窗口](supplement/all_year_end_matched_windows.csv)。',
              '- supplement/full_sample_matrices.npz：完整矩阵；supplement/tau1_all_time_modes.csv.gz：全部100个日频时间分量；supplement/tau1_all_loadings.csv：对应载荷。',
              '- 上一级目录的 hs300_long100_all_results.zip：打包此 full/ 目录全部研究结果、报告、审计清单、日志和运行参数；原始数据保留在项目 data/ 中，压缩包不重复打包。', '']
    (output / '完整分析报告.md').write_text('\n'.join(report), encoding='utf-8')
    key = {'dataset': dataset, 'assets': n, 'return_days': len(x), 'excluded_vendor_nontrading_rows': manifest['excluded_nontrading_dates'],
           'lambda1': matrix.lambda_max, 'market_variance_share': matrix.lambda_max_fraction,
           'mp_lower': matrix.lambda_minus, 'mp_upper': matrix.lambda_plus,
           'above_mp_count': int(matrix.high_outlier_count), 'below_mp_count': int(matrix.low_outlier_count),
           'industry_alignment': industry, 'mode_stability': modes, 'residual_groups': residual,
           'rmt_sharpe_difference': sharpe_delta, 'rmt_sharpe_difference_ci95': ci,
           'rmt_portfolio_performance': perf.to_dict('records'), 'basic_portfolio_performance': basic.to_dict('records'),
           'risk_forecast_summary': risk.to_dict('records'), 'circular_shift_reference_selected_tau': envelope[envelope.tau.isin(selected_tau)].to_dict('records')}
    write_json(output / 'key_findings.json', key)
    return key


def build_index_and_archive(dataset, output, checks, repetitions):
    titles = {'basic': '基础复现：经验谱与两策略回测', 'audit': '数据审计与市场逐日统计',
              'rmt': 'RMT分解、敏感性与四策略回测', 'structure': '厚尾、自相关、模态行业信息',
              'industry': '行业映射与层次聚类', 'scales': '多尺度和DMA结构对照',
              'modes': 'K–L分量、稳定性、残差群组、样本外风险', 'supplement': '长历史补充：完整年份、随机对照、大τ分量与全矩阵'}
    doc = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>沪深固定池长历史100股 · 全部结果</title>',
           '<style>body{max-width:1350px;margin:35px auto;padding:0 22px;font:16px/1.7 system-ui;color:#18364b;background:#fafcfd}h1,h2{line-height:1.3}a{color:#175f8e}nav{display:flex;gap:18px;flex-wrap:wrap}details{background:white;border:1px solid #dbe5ed;border-radius:9px;margin:22px 0;padding:18px}summary{font-size:22px;font-weight:650;cursor:pointer}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:24px}figure{margin:15px 0}img{width:100%;height:auto;border:1px solid #e7ecf0}figcaption{font-size:14px;overflow-wrap:anywhere}li{overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#eef3f7;padding:16px}code{font-size:14px}</style>',
           '<h1>沪深固定股票池长历史100股 · 完整结果</h1>',
           '<p>经成交量与价格联合核验后的共同区间：2002-02-05—2026-11-09；100股，6001个日收益。全部图为PNG。</p>',
           '<nav><a href="完整分析报告.md">完整中文分析报告</a><a href="key_findings.json">主要结果JSON</a><a href="artifact_manifest.csv">文件与哈希清单</a><a href="supplement/validation.json">校验记录</a></nav>',
           '<p>点击下面的模块标题展开图表与数据。每张图可点击打开原始PNG。表格、NPZ和报告都保留可下载链接。</p>',
           '<p>这是一组固定长历史股票的研究结果；行业标签是2026年快照。MP边界、时间顺序置换与独立循环平移分别回答不同问题。回测口径、随机重复次数和局限见中文报告。</p>']
    for folder, title in titles.items():
        paths = sorted((output/folder).rglob('*'))
        images = [p for p in paths if p.suffix.lower() == '.png']
        data = [p for p in paths if p.is_file() and p.suffix.lower() != '.png']
        doc.append(f'<details open><summary>{html.escape(title)} · {len(images)}张图</summary><div class="gallery">')
        for path in images:
            rel = path.relative_to(output).as_posix(); label = html.escape(rel)
            doc.append(f'<figure><a href="{label}"><img loading="lazy" src="{label}" alt="{label}"></a><figcaption>{label}</figcaption></figure>')
        doc.append('</div><details><summary>数据表、矩阵、日志与参数</summary><ul>')
        for path in data:
            rel = html.escape(path.relative_to(output).as_posix())
            doc.append(f'<li><a href="{rel}">{rel}</a> · {path.stat().st_size/1024:.1f} KB</li>')
        doc.append('</ul></details></details>')
    doc.append('<h2>复跑</h2><pre>python run.py hs300_long100 --analysis all\npython report_long_history.py hs300_long100\npython verify_data.py</pre></html>')
    (output / 'index.html').write_text('\n'.join(doc), encoding='utf-8')
    rows = []
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name not in ['artifact_manifest.csv', 'report_generation.log']:
            rows.append({'path': str(path.relative_to(output)), 'bytes': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(output / 'artifact_manifest.csv', index=False)
    archive = output.parent / f'{dataset}_all_results.zip'
    temp = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as pack:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name != 'report_generation.log':
                pack.write(path, f'{dataset}_all_results/{path.relative_to(output)}')
    temp.replace(archive)
    with zipfile.ZipFile(archive) as pack:
        if pack.testzip() is not None:
            raise AssertionError('Archive CRC verification failed')
    print(json.dumps({'archive': str(archive), 'archive_MB': round(archive.stat().st_size/1e6, 2),
                      'result_files': len(rows)+1, 'PNG': len(list(output.rglob('*.png'))),
                      'checks_passed': len(checks)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', default='hs300_long100')
    parser.add_argument('--output', type=Path, help='Completed full/ results folder; default outputs/DATASET/full')
    parser.add_argument('--shift-reps', type=int, default=100, help='Independent circular shifts per tau, default100')
    args = parser.parse_args()
    if args.shift_reps < 1:
        parser.error('shift-reps must be positive')
    output = args.output.resolve() if args.output else ROOT/'outputs'/args.dataset/'full'
    manifest = json.loads((output/'run_manifest.json').read_text())
    required = ['basic', 'audit', 'rmt', 'structure', 'industry', 'scales', 'modes']
    if any(manifest['stages'].get(stage, {}).get('status') != 'complete' for stage in required):
        raise SystemExit('先完成七个正式分析模块；本报告不将缺失或跳过当作已完成。')
    if manifest['arguments']['quick'] or manifest['arguments']['tau_max'] != 250:
        raise SystemExit('需要正式、完整τ=1–250的结果；不打包quick预览。')
    destination = output/'supplement'; destination.mkdir(parents=True, exist_ok=True)
    configure_plots()
    x = pd.read_csv(output/'log_returns.csv', index_col=0, parse_dates=True)
    daily = pd.read_csv(output/'rmt/tables/09_portfolio_daily_returns.csv', index_col=0, parse_dates=True)
    print('Validating and exporting complete matrices and all daily time components...', flush=True)
    checks = export_matrices_and_check(x, output, destination, manifest)
    print('Adding annual historical coverage and larger-tau time components...', flush=True)
    annual, annual_portfolio = annual_extensions(x, daily, destination)
    larger_tau = longer_tau_time_modes(x, destination)
    long_rolling_figures(output, destination)
    print('Running independent circular-shift control for all 250 horizons...', flush=True)
    metrics = pd.read_csv(output/'modes/tables/mode_metrics.csv')
    envelope = matched_shift_null(x, metrics, destination, args.shift_reps)
    checks['all_shift_reference_intervals_positive_ordered'] = bool(
        (envelope.null_lower_q025 > 0).all() and (envelope.null_lower_q025 < envelope.null_upper_q975).all())
    checks['all_250_reference_counts_sum_N'] = bool((envelope[['above', 'inside', 'below']].sum(axis=1) == x.shape[1]).all())
    for path in sorted(output.rglob('*.png')):
        with Image.open(path) as picture:
            picture.verify()
    checks['all_png_readable'] = True
    data_manifest = json.loads((ROOT/'data_manifest.json').read_text())
    checks['all_source_data_sha256_unchanged'] = all(
        hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest() == row['sha256'] for row in data_manifest)
    if not all(checks.values()):
        raise AssertionError([key for key, value in checks.items() if not value])
    data_summary = json.loads((ROOT/'data'/args.dataset/'summary.json').read_text())
    correct_generic_price_notes(output, data_summary)
    write_json(destination/'validation.json', {'checks': checks, 'checked_source_files': len(data_manifest),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'utc': datetime.now(timezone.utc).isoformat(), 'shift_seed': 20261110,
                'circular_shift_repetitions_per_tau': args.shift_reps,
                'circular_shift_rule': 'independent per-stock shifts of each standardized tau-return series',
                'reference_bounds': 'pointwise q025 of minima and q975 of maxima; not simultaneous across tau'})
    make_report(args.dataset, output, destination, manifest, x, annual, annual_portfolio, envelope, larger_tau, checks)
    # Preserve exact implementation with the result bundle, without credentials or caches.
    snapshots = output/'reproduction_code'
    for path in [ROOT/'run.py', ROOT/'pipeline.py', Path(__file__), ROOT/'datasets.json', ROOT/'requirements.txt', ROOT/'README.md',
                 *sorted((ROOT/'engines').rglob('*.py'))]:
        target = snapshots/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    build_index_and_archive(args.dataset, output, checks, args.shift_reps)


if __name__ == '__main__':
    main()
