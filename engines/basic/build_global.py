"""Apply the same RMT spectrum and rolling min-variance backtest to every global basket.

Each input subdirectory is treated as one universe. Every file with valid, positive
prices on the common calendar is included; no fixed-size sample is drawn.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from build_extension import (CAP, COST, HOLD, TRAIN, N_DAYS, filtered_covariance,
                             optimize, performance, spectrum, worst_episode_drawdown)

ROOT = Path(__file__).resolve().parent
DATA = Path('/Users/wangxujing/Documents/courses/经济物理学/task/global_indices_price')
OUT = ROOT / 'global'
COLORS = {'原始协方差': '#245894', 'RMT 去噪': '#CC6646'}


def read_prices(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(folder.glob('*.pkl'))
    if not files:
        raise ValueError(f'No price files in {folder}')
    columns, rows = [], []
    for file in files:
        df = pd.read_pickle(file)
        if list(df.columns) != ['Date', 'ClosePrice']:
            raise ValueError(f'Unexpected columns in {file}')
        s = df.set_index('Date')['ClosePrice'].astype(float).sort_index()
        if not s.index.is_unique:
            raise ValueError(f'Duplicate dates in {file}')
        columns.append(s.rename(file.stem))
        rows.append({'ticker': file.stem, 'file_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                     'source_file': str(file)})
    prices = pd.concat(columns, axis=1, join='inner').sort_index()
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError(f'Invalid values on common dates in {folder}')
    return prices, pd.DataFrame(rows)


def plot_spectrum(values: np.ndarray, low: float, high: float, destination: Path,
                  label: str) -> None:
    n = len(values)
    above, below = int((values > high).sum()), int((values < low).sum())
    fig, ax = plt.subplots(figsize=(12.8, 6.5))
    ax.scatter(np.arange(1, n+1), values, s=max(9, 42-n/16),
               c=np.where(values > high, '#C65E46', '#2D6688'))
    ax.axhspan(low, high, color='#3E947A', alpha=.12, label='MP 噪声区间')
    ax.axhline(high, color='#176F60', linestyle='--', linewidth=1.5,
               label=f'MP 上界 {high:.3f}')
    ax.axhline(low, color='#176F60', linestyle=':', linewidth=1.3)
    ax.annotate(f'最大 λ = {values[0]:.2f}\n占总方差 {values[0]/n:.1%}',
                (1, values[0]), xytext=(max(5, n*.12), values[0]*.75),
                arrowprops={'arrowstyle': '->', 'color': '#C65E46'},
                color='#9B4534', fontsize=13)
    ax.set(xlabel='特征值排名（降序）', ylabel='特征值 λ（对数刻度）',
           title=f'{label}：{above} 个高于 MP 上界，{below} 个低于下界')
    ax.set_yscale('log')
    ax.set_ylim(max(1e-6, values[-1]*.8), values[0]*1.3)
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def run_group(folder: Path, output: Path | None = None, *, verbose: bool = False,
              label: str | None = None) -> dict:
    """Analyze one folder of price pickles; write results to ``output`` if provided."""
    folder = Path(folder).expanduser().resolve()
    name = label or folder.name
    prices, manifest = read_prices(folder)
    if prices.shape[1] < int(np.ceil(1 / CAP)):
        raise ValueError(f'At least {int(np.ceil(1 / CAP))} stocks are needed for the {CAP:.0%} weight cap')
    if len(prices) - 1 <= TRAIN + 21:
        raise ValueError(f'At least {TRAIN + 23} common price dates are needed for the rolling test')
    out = Path(output).expanduser().resolve() if output is not None else OUT / name
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / 'included_all.csv', index=False)
    n = prices.shape[1]
    log_returns = np.log(prices).diff().iloc[1:]
    simple_returns = prices.pct_change().iloc[1:]
    _, eigenvalues, _, low, high = spectrum(log_returns.to_numpy())
    above = int((eigenvalues > high).sum())
    below = int((eigenvalues < low).sum())
    pd.DataFrame({'rank': np.arange(1, n+1), 'eigenvalue': eigenvalues,
                  'MP_class': np.where(eigenvalues > high, 'above',
                                       np.where(eigenvalues < low, 'below', 'bulk'))}).to_csv(
                                           out / 'eigenvalues.csv', index=False)
    plot_spectrum(eigenvalues, low, high, figures / '01_ranked_spectrum.png', name.upper())

    strategies = ['原始协方差', 'RMT 去噪']
    realized = {strategy: [] for strategy in strategies}
    dates = []
    previous_drift = {strategy: None for strategy in strategies}
    previous_target = {strategy: None for strategy in strategies}
    rebalances = []
    for start in range(TRAIN, len(simple_returns), HOLD):
        stop = min(start + HOLD, len(simple_returns))
        training = simple_returns.iloc[start-TRAIN:start].to_numpy(dtype=float)
        future = simple_returns.iloc[start:stop]
        raw_cov = np.cov(training, rowvar=False, ddof=1)
        clean_cov, signals, mp_high = filtered_covariance(training)
        dates.extend(future.index.tolist())
        for strategy, cov in zip(strategies, [raw_cov, clean_cov]):
            target = optimize(cov, previous_target[strategy])
            prior = previous_drift[strategy]
            turnover = 1.0 if prior is None else .5 * np.abs(target-prior).sum()
            current = target.copy()
            block = []
            for k, returns in enumerate(future.to_numpy(dtype=float)):
                gross = float(current @ returns)
                net = (1 - turnover*COST)*(1+gross)-1 if k == 0 else gross
                block.append(net)
                current = current*(1+returns)/(1+gross)
            realized[strategy].extend(block)
            previous_drift[strategy] = current
            previous_target[strategy] = target
            rebalances.append({'date': str(future.index[0].date()), 'strategy': strategy,
                               'turnover': float(turnover), 'signal_eigenvalues': signals,
                               'mp_upper': mp_high, 'active_positions': int((target > 1e-7).sum())})
        if verbose and start % (HOLD*10) == TRAIN % (HOLD*10):
            print(name, f'{stop}/{len(simple_returns)}', flush=True)
    result = pd.DataFrame(realized, index=pd.DatetimeIndex(dates))
    result.index.name = 'Date'
    result.to_csv(out / 'portfolio_daily_returns.csv')
    pd.DataFrame(rebalances).to_csv(out / 'rebalance_summary.csv', index=False)
    full_metrics = pd.DataFrame([{'strategy': strategy, **performance(result[strategy])}
                                 for strategy in strategies])
    full_metrics.to_csv(out / 'full_period_metrics.csv', index=False)

    market = simple_returns.mean(axis=1).reindex(result.index)
    vol20 = market.rolling(20).std(ddof=1) * np.sqrt(N_DAYS)
    return21 = (1+market).rolling(21).apply(np.prod, raw=True)-1
    high_vol_threshold = float(vol20.quantile(.9))
    fall_threshold = float(return21.quantile(.1))
    flags = pd.DataFrame({'high_vol': vol20 >= high_vol_threshold,
                          'market_fall': return21 <= fall_threshold,
                          'market_vol20': vol20, 'market_return21': return21}, index=result.index)
    flags.index.name = 'Date'
    flags.to_csv(out / 'stress_dates.csv')
    stress_rows = []
    for regime, column in [('高波动', 'high_vol'), ('市场下跌', 'market_fall')]:
        mask = flags[column]
        for strategy in strategies:
            selected = result.loc[mask, strategy]
            dd, episodes, longest = worst_episode_drawdown(result[strategy], mask)
            stress_rows.append({'regime': regime, 'strategy': strategy,
                                'days': int(mask.sum()), 'episodes': episodes,
                                'longest_episode_days': longest,
                                'conditional_sharpe': float(selected.mean()/selected.std(ddof=1)*np.sqrt(N_DAYS)),
                                'worst_episode_drawdown': dd})
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(out / 'stress_metrics.csv', index=False)

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7), sharex=True,
                             gridspec_kw={'height_ratios': [1.25, .85]})
    for strategy in strategies:
        wealth = (1 + result[strategy]).cumprod()
        drawdown = wealth / np.maximum.accumulate(np.r_[1, wealth.to_numpy()])[1:] - 1
        axes[0].plot(result.index, wealth, label=strategy, linewidth=1.6, color=COLORS[strategy])
        axes[1].plot(result.index, drawdown*100, linewidth=1.3, color=COLORS[strategy])
    axes[0].set(title=f'{name.upper()} 样本外净值：每 {HOLD} 日调仓、交易成本 {COST*10000:.0f} bps', ylabel='初始净值 = 1')
    axes[1].set(xlabel='交易日期', ylabel='回撤 (%)')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / '02_backtest.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    labels, positions = ['高波动', '市场下跌'], np.arange(2)
    for i, strategy in enumerate(strategies):
        data = stress.set_index(['regime', 'strategy']).loc[[(r, strategy) for r in labels]]
        offset = (i-.5)*.33
        axes[0].bar(positions+offset, data.conditional_sharpe, width=.31,
                    label=strategy, color=COLORS[strategy])
        axes[1].bar(positions+offset, -data.worst_episode_drawdown*100, width=.31,
                    label=strategy, color=COLORS[strategy])
    axes[0].axhline(0, color='#8798A3', linewidth=.8)
    axes[0].set(ylabel='条件年化 Sharpe（无风险利率 0）', title='压力日期的收益 / 波动')
    axes[1].set(ylabel='最深单次压力段回撤 (%)', title='连续压力段内的最坏回撤')
    for ax in axes:
        ax.set_xticks(positions, labels)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / '03_stress_comparison.png', dpi=180)
    plt.close(fig)

    summary = {'group': name, 'source': str(folder), 'assets': n,
               'price_days': len(prices), 'return_days': len(simple_returns),
               'price_start': str(prices.index.min().date()),
               'backtest_start': str(result.index.min().date()),
               'backtest_end': str(result.index.max().date()),
               'backtest_days': len(result), 'rebalance_count': len(rebalances)//2,
               'mp_lower': low, 'mp_upper': high,
               'top_eigenvalue': float(eigenvalues[0]),
               'top_variance_share': float(eigenvalues[0]/n),
               'above_upper': above, 'below_lower': below,
               'train_days': TRAIN, 'rebalance_days': HOLD,
               'weight_cap': CAP, 'cost_bps': COST*10000,
               'high_vol_days': int(flags.high_vol.sum()),
               'market_fall_days': int(flags.market_fall.sum()),
               'stress_definition': 'Ex-post top decile of 20-day rolling annualized equal-weight basket volatility and bottom decile of trailing 21-day equal-weight basket return.',
               'portfolio_interpretation': 'Minimum-variance constituent basket proxy, not an actual ETF.'}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if verbose:
        print(name, 'done', summary['assets'],
              full_metrics[['strategy','sharpe','max_drawdown']].to_dict('records'), flush=True)
    return summary


def configure_plots() -> None:
    font = Path('/System/Library/Fonts/Hiragino Sans GB.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Hiragino Sans GB' if font.exists() else 'sans-serif',
                         'axes.unicode_minus': False,
                         'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_plots()
    groups = sorted(folder for folder in DATA.iterdir() if folder.is_dir())
    all_summaries = [run_group(folder, verbose=True) for folder in groups]
    (OUT / 'summary.json').write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
