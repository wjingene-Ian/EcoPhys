"""Project 3 advanced section: industry mapping and an out-of-sample RMT portfolio test.

The portfolio is a constituent-stock ETF proxy, not an exchange-traded fund.
Training uses only dates before each rebalance; stress labels are retrospective.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = Path("/Users/wangxujing/Documents/courses/经济物理学/task/data300_price")
INDUSTRY = Path("/Users/wangxujing/Documents/courses/经济物理学/task/experiment/inputs/industry_map.csv")
OUT = ROOT / "extension"
FIG = OUT / "figures"
TRAIN = 504
HOLD = 21
CAP = .05
COST = 10 / 10000
N_DAYS = 252


def get_prices() -> pd.DataFrame:
    names = pd.read_csv(ROOT / "analysis" / "included_all.csv")["ticker"].tolist()
    assert len(names) == len(set(names)) == 288
    parts = []
    for name in names:
        frame = pd.read_pickle(DATA / f"{name}.pkl")
        parts.append(frame.set_index("Date")["ClosePrice"].astype(float).rename(name))
    prices = pd.concat(parts, axis=1, join="inner").sort_index()
    assert prices.shape == (1877, 288)
    assert np.isfinite(prices.to_numpy()).all() and (prices.to_numpy() > 0).all()
    return prices


def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    t, n = x.shape
    sd = x.std(axis=0, ddof=1)
    z = (x - x.mean(axis=0)) / sd
    c = z.T @ z / (t - 1)
    c = (c + c.T) / 2
    vals, vecs = np.linalg.eigh(c)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    q = n / t
    low, high = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    return c, vals, vecs, low, high


def filtered_covariance(x: np.ndarray) -> tuple[np.ndarray, int, float]:
    c, vals, vecs, _, high = spectrum(x)
    noise = vals <= high
    clean_vals = vals.copy()
    clean_vals[noise] = vals[noise].mean()
    clean = (vecs * clean_vals) @ vecs.T
    clean = (clean + clean.T) / 2
    d = np.sqrt(np.diag(clean))
    clean /= np.outer(d, d)
    sd = x.std(axis=0, ddof=1)
    cov = clean * np.outer(sd, sd)
    assert np.linalg.eigvalsh(cov).min() > -1e-12
    return cov, int((~noise).sum()), float(high)


def project_capped_simplex(y: np.ndarray, cap: float = CAP) -> np.ndarray:
    """Euclidean projection onto nonnegative weights summing to one, each <= cap."""
    low, high = float(y.min() - cap), float(y.max())
    for _ in range(40):
        middle = (low + high) / 2
        if np.clip(y - middle, 0, cap).sum() > 1:
            low = middle
        else:
            high = middle
    return np.clip(y - (low + high) / 2, 0, cap)


def optimize(cov: np.ndarray, initial: np.ndarray | None = None, *,
             cap: float = CAP, ridge_fraction: float = 1e-7) -> np.ndarray:
    n = len(cov)
    assert n * cap >= 1
    cov = ((cov + cov.T) / 2).copy()
    cov += np.eye(n) * np.trace(cov) / n * ridge_fraction
    scale = 1 / max(np.trace(cov) / n, 1e-12)
    cov *= scale
    lipschitz = 2 * np.linalg.eigvalsh(cov)[-1]
    w = np.full(n, 1 / n) if initial is None else project_capped_simplex(initial, cap)
    y, momentum = w.copy(), 1.0
    residual = float('inf')
    for iteration in range(4001):
        next_w = project_capped_simplex(y - 2 * (cov @ y) / lipschitz, cap)
        next_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        y = next_w + (momentum - 1) / next_momentum * (next_w - w)
        w, momentum = next_w, next_momentum
        if iteration % 100 == 0:
            residual = np.max(np.abs(project_capped_simplex(w - 2 * (cov @ w) / lipschitz, cap) - w))
            if residual < 2e-6:
                break
    if residual >= 2e-6:
        raise RuntimeError(f"Projected-gradient optimizer did not converge: {residual:.3g}")
    if abs(w.sum() - 1) > 1e-7 or w.min() < -1e-7 or w.max() > cap + 1e-7:
        raise RuntimeError("Invalid optimizer weights")
    return w


def performance(r: pd.Series) -> dict:
    x = r.to_numpy(dtype=float)
    wealth = np.cumprod(1 + x)
    peak = np.maximum.accumulate(np.r_[1, wealth])[1:]
    return {
        "days": len(x), "annual_return": float(wealth[-1] ** (N_DAYS / len(x)) - 1),
        "annual_vol": float(np.std(x, ddof=1) * np.sqrt(N_DAYS)),
        "sharpe": float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(N_DAYS)),
        "max_drawdown": float(np.min(wealth / peak - 1)),
        "cumulative_return": float(wealth[-1] - 1),
    }


def worst_episode_drawdown(r: pd.Series, mask: pd.Series) -> tuple[float, int, int]:
    vals, flags = r.to_numpy(), mask.to_numpy(dtype=bool)
    best = 0.0
    run_count = max_len = 0
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < len(flags) and flags[j]:
            j += 1
        run_count += 1
        max_len = max(max_len, j - i)
        wealth = np.cumprod(1 + vals[i:j])
        peak = np.maximum.accumulate(np.r_[1, wealth])[1:]
        best = min(best, float(np.min(wealth / peak - 1)))
        i = j
    return best, run_count, max_len


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont("/System/Library/Fonts/Hiragino Sans GB.ttc")
    plt.rcParams.update({"font.family": "Hiragino Sans GB", "axes.unicode_minus": False,
                         "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    prices = get_prices()
    log_r = np.log(prices).diff().iloc[1:]
    simple_r = prices.pct_change().iloc[1:]
    names = prices.columns.tolist()

    # Industry mapping uses the full-stock eigenvector from the basic section.
    _, vals, vecs, _, _ = spectrum(log_r.to_numpy())
    v = vecs[:, 0].copy()
    if v.sum() < 0:
        v *= -1
    industry = pd.read_csv(INDUSTRY, dtype={"Ticker": str}).drop_duplicates("Ticker", keep="last")
    info = industry.set_index("Ticker").reindex(names)
    assert info["Industry"].notna().all()
    load = pd.DataFrame({"Ticker": names, "Industry": info["Industry"].to_numpy(),
                         "Name": info["Name"].to_numpy(), "Loading": v,
                         "SquaredLoading": v * v})
    load.to_csv(OUT / "market_loadings_industry.csv", index=False)
    sector = load.groupby("Industry").agg(stocks=("Ticker", "count"),
                                           loading_mass=("SquaredLoading", "sum"),
                                           mean_abs_loading=("Loading", lambda x: float(np.mean(np.abs(x)))))
    sector["stock_share"] = sector.stocks / len(names)
    sector["mass_to_stock_share"] = sector.loading_mass / sector.stock_share
    sector = sector.sort_values("loading_mass", ascending=False)
    sector.to_csv(OUT / "industry_loading_summary.csv")
    hhi = float((sector.loading_mass ** 2).sum())

    # One full-sample matrix image is descriptive only. Portfolio matrices below are rolling.
    full_corr, _, _, _, _ = spectrum(simple_r.to_numpy())
    full_clean, _, _ = filtered_covariance(simple_r.to_numpy())
    full_sd = simple_r.std(ddof=1).to_numpy()
    full_clean_corr = full_clean / np.outer(full_sd, full_sd)

    strategies = ["原始协方差", "RMT 去噪"]
    realized = {name: [] for name in strategies}
    dates = []
    last_drift = {name: None for name in strategies}
    last_target = {name: None for name in strategies}
    rebalances = []
    weight_rows = []
    for start in range(TRAIN, len(simple_r), HOLD):
        stop = min(start + HOLD, len(simple_r))
        train = simple_r.iloc[start - TRAIN:start].to_numpy(dtype=float)
        test = simple_r.iloc[start:stop]
        raw_cov = np.cov(train, rowvar=False, ddof=1)
        clean_cov, signal_count, mp_high = filtered_covariance(train)
        candidates = {strategies[0]: raw_cov, strategies[1]: clean_cov}
        dates.extend(test.index.tolist())
        for name in strategies:
            target = optimize(candidates[name], last_target[name])
            prior = last_drift[name]
            turnover = 1.0 if prior is None else 0.5 * float(np.abs(target - prior).sum())
            current = target.copy()
            block = []
            for k, r_t in enumerate(test.to_numpy(dtype=float)):
                gross = float(current @ r_t)
                net = (1 - turnover * COST) * (1 + gross) - 1 if k == 0 else gross
                block.append(net)
                current = current * (1 + r_t) / (1 + gross)
            realized[name].extend(block)
            last_drift[name] = current
            last_target[name] = target
            rebalances.append({"date": str(test.index[0].date()), "strategy": name,
                               "turnover": turnover, "signal_eigenvalues": signal_count,
                               "mp_upper": mp_high, "active_positions": int((target > 1e-7).sum()),
                               "max_weight": float(target.max())})
            for ticker, weight in zip(names, target):
                weight_rows.append({"date": str(test.index[0].date()), "strategy": name,
                                    "ticker": ticker, "weight": float(weight)})
    out = pd.DataFrame(realized, index=pd.DatetimeIndex(dates))
    assert len(out) == len(simple_r) - TRAIN and np.isfinite(out.to_numpy()).all()
    out.index.name = "Date"
    out.to_csv(OUT / "portfolio_daily_returns.csv")
    pd.DataFrame(rebalances).to_csv(OUT / "rebalance_summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUT / "portfolio_weights.csv", index=False)
    all_metrics = pd.DataFrame([{"strategy": name, **performance(out[name])} for name in strategies])
    all_metrics.to_csv(OUT / "full_period_metrics.csv", index=False)

    # Retrospective common stress labels: identical dates for both strategies.
    market = simple_r.mean(axis=1).reindex(out.index)
    vol20 = market.rolling(20).std(ddof=1) * np.sqrt(N_DAYS)
    return21 = (1 + market).rolling(21).apply(np.prod, raw=True) - 1
    vol_threshold = float(vol20.quantile(.9))
    crash_threshold = float(return21.quantile(.1))
    mask = pd.DataFrame({"high_vol": vol20 >= vol_threshold,
                         "market_fall": return21 <= crash_threshold,
                         "market_vol20": vol20, "market_return21": return21}, index=out.index)
    mask.index.name = "Date"
    mask.to_csv(OUT / "stress_dates.csv")
    stress_rows = []
    for label, col in [("高波动", "high_vol"), ("市场下跌", "market_fall")]:
        flags = mask[col]
        for name in strategies:
            selected = out.loc[flags, name]
            dd, episodes, longest = worst_episode_drawdown(out[name], flags)
            stress_rows.append({"regime": label, "strategy": name,
                                "days": int(flags.sum()), "episodes": episodes,
                                "longest_episode_days": longest,
                                "conditional_sharpe": float(selected.mean() / selected.std(ddof=1) * np.sqrt(N_DAYS)),
                                "worst_episode_drawdown": dd,
                                "worst_day": float(selected.min()),
                                "mean_daily_return": float(selected.mean())})
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(OUT / "stress_metrics.csv", index=False)

    # Figure 1: industry mass versus number of stocks. Group small sectors as Other.
    top = sector.head(10).copy()
    if len(sector) > 10:
        top.loc["其他"] = [sector.iloc[10:].stocks.sum(), sector.iloc[10:].loading_mass.sum(),
                          np.nan, sector.iloc[10:].stock_share.sum(), np.nan]
    top = top.iloc[::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12.8, 6.5))
    ax.barh(y - .2, top.stock_share * 100, height=.37, label="股票数量份额", color="#A3BBC9")
    ax.barh(y + .2, top.loading_mass * 100, height=.37, label="第一模态载荷平方份额", color="#2A7C69")
    ax.set_yticks(y, top.index)
    ax.set(xlabel="份额 (%)", title="第一特征向量的行业分布：与股票数量作对照")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "05_industry_mapping.png", dpi=180)
    plt.close(fig)

    # Figure 2: descriptive full-sample correlation cleaning.
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 6.2))
    for ax, mat, title in zip(axs, [full_corr, full_clean_corr], ["原始相关矩阵", "全样本 RMT 去噪矩阵（示意）"]):
        im = ax.imshow(mat, vmin=-.4, vmax=.8, cmap="RdBu_r", interpolation="nearest")
        ax.set(title=title, xlabel="股票序号", ylabel="股票序号")
    fig.colorbar(im, ax=axs, fraction=.026, pad=.02, label="相关系数")
    fig.subplots_adjust(left=.07, right=.88, wspace=.16, top=.88, bottom=.11)
    fig.savefig(FIG / "06_rmt_filtering.png", dpi=180)
    plt.close(fig)

    # Figure 3: actual out-of-sample net wealth and drawdown.
    fig, axs = plt.subplots(2, 1, figsize=(13.2, 7), sharex=True,
                            gridspec_kw={"height_ratios": [1.25, .85]})
    colors = {strategies[0]: "#245894", strategies[1]: "#CC6646"}
    for name in strategies:
        wealth = (1 + out[name]).cumprod()
        dd = wealth / np.maximum.accumulate(np.r_[1, wealth.to_numpy()])[1:] - 1
        axs[0].plot(out.index, wealth, label=name, linewidth=1.6, color=colors[name])
        axs[1].plot(out.index, dd * 100, label=name, linewidth=1.3, color=colors[name])
    axs[0].set(title="样本外净值：每 21 日调仓、交易成本 10 bps", ylabel="初始净值 = 1")
    axs[1].set(xlabel="交易日期", ylabel="回撤 (%)")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "07_backtest.png", dpi=180)
    plt.close(fig)

    # Figure 4: stress comparison, using the same masks for both strategies.
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.7))
    labels = ["高波动", "市场下跌"]
    positions = np.arange(2)
    for i, name in enumerate(strategies):
        d = stress.set_index(["regime", "strategy"]).loc[[(x, name) for x in labels]]
        offset = (i - .5) * .33
        axs[0].bar(positions + offset, d.conditional_sharpe, width=.31, label=name, color=colors[name])
        axs[1].bar(positions + offset, -d.worst_episode_drawdown * 100, width=.31,
                   label=name, color=colors[name])
    axs[0].axhline(0, color="#8798A3", linewidth=.8)
    axs[0].set(ylabel="条件年化 Sharpe（无风险利率 0）", title="压力日期的收益 / 波动")
    axs[1].set(ylabel="最深单次压力段回撤 (%)", title="连续压力段内的最坏回撤")
    for ax in axs:
        ax.set_xticks(positions, labels)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "08_stress_comparison.png", dpi=180)
    plt.close(fig)

    summary = {
        "assets": len(names), "return_days": len(simple_r),
        "market_mode_eigenvalue": float(vals[0]),
        "market_mode_explained_share": float(vals[0] / len(names)),
        "loading_positive_fraction": float((v > 0).mean()),
        "loading_participation": float(1 / np.sum(v ** 4)),
        "sector_count": len(sector), "sector_label_coverage": 1.0,
        "sector_loading_hhi": hhi,
        "top_sectors": sector.head(5).reset_index().to_dict("records"),
        "train_days": TRAIN, "rebalance_days": HOLD, "max_weight": CAP,
        "transaction_cost_bps": COST * 10000,
        "backtest_start": str(out.index.min().date()), "backtest_end": str(out.index.max().date()),
        "backtest_days": len(out), "rebalance_count": len(rebalances) // 2,
        "high_vol_threshold": vol_threshold, "market_fall_threshold": crash_threshold,
        "stress_definition": "Ex-post top decile of rolling 20-day annualized equal-weight market volatility, and bottom decile of rolling 21-day equal-weight market simple return, within the out-of-sample period. These labels do not affect weights.",
        "stress_drawdown_definition": "Worst max drawdown within any contiguous run of stress-flagged dates; each run begins at wealth=1.",
        "strategy_return_convention": "Simple returns; weights buy-and-hold and drift within each 21-day block; rebalance turnover costs on first day.",
        "portfolio_interpretation": "Constituent-stock minimum-variance basket, serving as an ETF proxy, not an actual ETF or ETF return.",
        "industry_source": str(INDUSTRY),
        "industry_caveat": "Industry labels were retrieved 2026-09-29 and do not represent historical time-varying classifications.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(all_metrics.to_string(index=False))
    print(stress.to_string(index=False))


if __name__ == "__main__":
    main()
