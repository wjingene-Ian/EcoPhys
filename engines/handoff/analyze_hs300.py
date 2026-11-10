#!/usr/bin/env python3
"""
沪深300 股票日频数据的初步审计与时间演化分析。

输入是 Data/沪深300 下的三组 pickle：复权价格、对数收益率、全样本 z-score。
脚本不修改原始数据，所有表格和图形都写入 Analysis_codex。

注：这批 pickle 由 pandas 3.x 写出，请使用 pandas>=3.0 读取。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


DATASET_DIRS = {
    "price": "沪深300(原始价格)",
    "return": "沪深300(收益率)",
    "zscore": "沪深300(标准化)",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_input = here.parent / "Data" / "沪深300"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=here)
    parser.add_argument("--corr-window", type=int, default=60)
    parser.add_argument("--eigen-window", type=int, default=120)
    parser.add_argument("--eigen-step", type=int, default=5)
    return parser.parse_args()


def ensure_dirs(output: Path) -> tuple[Path, Path]:
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def load_frame(path: Path, expected_column: str) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{path} is {type(frame)!r}, expected pandas.DataFrame")
    if expected_column not in frame.columns:
        raise ValueError(f"{path} lacks column {expected_column!r}: {list(frame.columns)}")
    return frame


def longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def max_drawdown_from_log_returns(values: pd.Series) -> float:
    wealth = np.exp(values.cumsum())
    drawdown = wealth / wealth.cummax().clip(lower=1.0) - 1.0
    return float(drawdown.min())


def leading_correlation_eigenvalue_fraction(window: np.ndarray, iterations: int = 35) -> float:
    """Power iteration for lambda_max(C)/N without forming an N x N matrix."""
    x = window - window.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, ddof=1, keepdims=True)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        return np.nan
    z = x / scale
    n_assets = z.shape[1]
    vector = np.full(n_assets, 1.0 / math.sqrt(n_assets))
    for _ in range(iterations):
        updated = z.T @ (z @ vector) / (z.shape[0] - 1)
        norm = np.linalg.norm(updated)
        if not np.isfinite(norm) or norm == 0:
            return np.nan
        vector = updated / norm
    eigenvalue = float(vector @ (z.T @ (z @ vector)) / (z.shape[0] - 1))
    return eigenvalue / n_assets


def rolling_collectivity(
    returns: pd.DataFrame, corr_window: int, eigen_window: int, eigen_step: int
) -> pd.DataFrame:
    values = returns.to_numpy(float)
    dates = returns.index
    avg_corr = np.full(len(dates), np.nan)
    for end in range(corr_window - 1, len(dates)):
        block = values[end - corr_window + 1 : end + 1]
        centered = block - block.mean(axis=0, keepdims=True)
        scale = centered.std(axis=0, ddof=1, keepdims=True)
        z = centered / scale
        n_assets = z.shape[1]
        total_corr = float(np.square(z.sum(axis=1)).sum() / (corr_window - 1))
        avg_corr[end] = (total_corr - n_assets) / (n_assets * (n_assets - 1))

    eigen_fraction = pd.Series(np.nan, index=dates, dtype=float)
    evaluation_ends = list(range(eigen_window - 1, len(dates), eigen_step))
    if evaluation_ends and evaluation_ends[-1] != len(dates) - 1:
        evaluation_ends.append(len(dates) - 1)
    for end in evaluation_ends:
        block = values[end - eigen_window + 1 : end + 1]
        eigen_fraction.iloc[end] = leading_correlation_eigenvalue_fraction(block)
    eigen_fraction = eigen_fraction.ffill()

    return pd.DataFrame(
        {
            f"avg_pairwise_corr_{corr_window}d": avg_corr,
            f"lambda1_fraction_{eigen_window}d": eigen_fraction,
        },
        index=dates,
    )


def audit_and_load(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dirs = {key: input_dir / name for key, name in DATASET_DIRS.items()}
    for path in dirs.values():
        if not path.is_dir():
            raise FileNotFoundError(path)

    ticker_sets = {key: {path.stem for path in folder.glob("*.pkl")} for key, folder in dirs.items()}
    if not (ticker_sets["price"] == ticker_sets["return"] == ticker_sets["zscore"]):
        raise ValueError("The three pickle folders do not contain the same ticker set")
    tickers = sorted(ticker_sets["price"])

    return_series: dict[str, pd.Series] = {}
    rows: list[dict] = []
    reference_return_index: pd.DatetimeIndex | None = None
    all_dates_aligned = True

    return_manifest = pd.read_csv(dirs["return"] / "manifest.csv", dtype={"Ticker": str})
    return_manifest = return_manifest.set_index("Ticker")

    for ticker in tickers:
        price = load_frame(dirs["price"] / f"{ticker}.pkl", "ClosePrice").copy()
        price["Date"] = pd.to_datetime(price["Date"])
        price = price.set_index("Date")["ClosePrice"].astype(float)
        ret = load_frame(dirs["return"] / f"{ticker}.pkl", "Return")["Return"].astype(float)
        zscore = load_frame(dirs["zscore"] / f"{ticker}.pkl", "ZScore")["ZScore"].astype(float)
        ret.index = pd.to_datetime(ret.index)
        zscore.index = pd.to_datetime(zscore.index)

        if reference_return_index is None:
            reference_return_index = ret.index
        else:
            all_dates_aligned = all_dates_aligned and ret.index.equals(reference_return_index)

        calculated_return = np.log(price).diff().dropna()
        common_ret = ret.index.intersection(calculated_return.index)
        return_diff = ret.loc[common_ret] - calculated_return.loc[common_ret]
        calculated_z = (ret - ret.mean()) / ret.std(ddof=0)
        z_diff = zscore - calculated_z

        manifest_row = return_manifest.loc[ticker] if ticker in return_manifest.index else pd.Series(dtype=object)
        rows.append(
            {
                "ticker": ticker,
                "price_start": price.index.min(),
                "price_end": price.index.max(),
                "price_observations": len(price),
                "return_start": ret.index.min(),
                "return_end": ret.index.max(),
                "return_observations": len(ret),
                "price_missing": int(price.isna().sum()),
                "return_missing": int(ret.isna().sum()),
                "zscore_missing": int(zscore.isna().sum()),
                "nonpositive_prices": int((price <= 0).sum()),
                "duplicate_price_dates": int(price.index.duplicated().sum()),
                "duplicate_return_dates": int(ret.index.duplicated().sum()),
                "return_matches_log_price_max_abs_error": float(return_diff.abs().max()),
                "zscore_matches_full_sample_max_abs_error": float(z_diff.abs().max()),
                "zero_return_days": int(np.isclose(ret.to_numpy(), 0.0, atol=1e-14).sum()),
                "max_consecutive_zero_returns": longest_true_run(np.isclose(ret.to_numpy(), 0.0, atol=1e-14)),
                "membership_start": manifest_row.get("MembershipStart", np.nan),
                "membership_end": manifest_row.get("MembershipEnd", np.nan),
                "price_source": manifest_row.get("PriceSource", np.nan),
            }
        )
        return_series[ticker] = ret

    audit = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    returns = pd.DataFrame(return_series).sort_index()

    excluded = pd.read_csv(dirs["return"] / "excluded_stocks.csv", dtype={"Ticker": str})
    membership_start = pd.to_datetime(return_manifest["MembershipStart"], errors="coerce")
    membership_end = pd.to_datetime(return_manifest["MembershipEnd"], errors="coerce")
    active_counts = pd.Series(
        [int(((membership_start <= date) & (membership_end >= date)).sum()) for date in returns.index],
        index=returns.index,
        name="manifest_active_members",
    )

    metadata = {
        "pandas_version": pd.__version__,
        "ticker_count": len(tickers),
        "excluded_ticker_count": int(len(excluded)),
        "excluded_by_reason": {str(k): int(v) for k, v in excluded["Reason"].value_counts().items()},
        "all_return_dates_aligned": bool(all_dates_aligned),
        "return_matrix_shape": list(returns.shape),
        "date_start": returns.index.min().date().isoformat(),
        "date_end": returns.index.max().date().isoformat(),
        "total_missing_returns": int(returns.isna().sum().sum()),
        "max_return_reconstruction_error": float(audit["return_matches_log_price_max_abs_error"].max()),
        "max_zscore_reconstruction_error": float(audit["zscore_matches_full_sample_max_abs_error"].max()),
        "manifest_active_members_min": int(active_counts.min()),
        "manifest_active_members_max": int(active_counts.max()),
        "manifest_active_members_last": int(active_counts.iloc[-1]),
    }
    coverage = active_counts.to_frame()
    coverage["constant_analysis_universe"] = len(tickers)
    return returns, audit, {"summary": metadata, "coverage": coverage, "excluded": excluded, "manifest": return_manifest}


def compute_metrics(returns: pd.DataFrame, corr_window: int, eigen_window: int, eigen_step: int) -> pd.DataFrame:
    mean_constituent_log = returns.mean(axis=1)
    market_simple = np.expm1(returns).mean(axis=1)
    market_log = np.log1p(market_simple)
    cumulative = np.exp(market_log.cumsum())
    drawdown = cumulative / cumulative.cummax().clip(lower=1.0) - 1.0
    metrics = pd.DataFrame(index=returns.index)
    metrics.index.name = "date"
    metrics["ew_market_log_return"] = market_log
    metrics["ew_market_simple_return"] = market_simple
    metrics["ew_cumulative_index"] = cumulative
    metrics["ew_drawdown"] = drawdown
    metrics["cross_section_mean_log_return"] = mean_constituent_log
    metrics["geometric_mean_cohort_index"] = np.exp(mean_constituent_log.cumsum())
    metrics["cross_section_median_log_return"] = returns.median(axis=1)
    metrics["cross_section_dispersion"] = returns.std(axis=1, ddof=1)
    metrics["breadth_positive_fraction"] = (returns > 0).mean(axis=1)
    metrics["zero_return_fraction"] = np.isclose(returns, 0.0, atol=1e-14).mean(axis=1)
    metrics["abs_return_gt_5pct_fraction"] = (np.abs(np.expm1(returns)) > 0.05).mean(axis=1)
    metrics["abs_return_gt_9pct_fraction"] = (np.abs(np.expm1(returns)) > 0.09).mean(axis=1)
    metrics["rolling_vol_20d_annualized"] = market_log.rolling(20).std(ddof=1) * np.sqrt(242)
    metrics["rolling_vol_60d_annualized"] = market_log.rolling(60).std(ddof=1) * np.sqrt(242)
    metrics["rolling_return_20d"] = np.expm1(market_log.rolling(20).sum())
    metrics["rolling_return_60d"] = np.expm1(market_log.rolling(60).sum())
    metrics = metrics.join(rolling_collectivity(returns, corr_window, eigen_window, eigen_step))
    return metrics


def compute_stock_summary(returns: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    years = len(returns) / 242.0
    rows = []
    for ticker in returns.columns:
        series = returns[ticker]
        simple_series = np.expm1(series)
        annual_return = float(np.exp(series.sum() / years) - 1.0)
        annual_vol = float(series.std(ddof=1) * np.sqrt(242))
        rows.append(
            {
                "ticker": ticker,
                "annualized_geometric_return": annual_return,
                "annualized_volatility": annual_vol,
                "sharpe_rf0": float(simple_series.mean() / simple_series.std(ddof=1) * np.sqrt(242)),
                "max_drawdown": max_drawdown_from_log_returns(series),
                "daily_skewness": float(stats.skew(series, bias=False)),
                "daily_excess_kurtosis": float(stats.kurtosis(series, fisher=True, bias=False)),
                "worst_daily_simple_return": float(np.expm1(series.min())),
                "worst_day": series.idxmin(),
                "best_daily_simple_return": float(np.expm1(series.max())),
                "best_day": series.idxmax(),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.merge(
        audit[["ticker", "zero_return_days", "max_consecutive_zero_returns", "membership_start", "membership_end"]],
        on="ticker",
        how="left",
    ).sort_values("ticker")


def build_event_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    categories = {
        "largest_market_drop": metrics["ew_market_simple_return"].nsmallest(15).index,
        "largest_market_rise": metrics["ew_market_simple_return"].nlargest(15).index,
        "largest_dispersion": metrics["cross_section_dispersion"].nlargest(15).index,
        "most_stocks_beyond_5pct": metrics["abs_return_gt_5pct_fraction"].nlargest(15).index,
        "lowest_breadth": metrics["breadth_positive_fraction"].nsmallest(10).index,
        "highest_breadth": metrics["breadth_positive_fraction"].nlargest(10).index,
    }
    flags: dict[pd.Timestamp, list[str]] = {}
    for label, dates in categories.items():
        for date in dates:
            flags.setdefault(date, []).append(label)
    candidates = metrics.loc[sorted(flags)].copy()
    candidates.insert(0, "signals", [";".join(flags[date]) for date in candidates.index])
    candidates["abs_market_return_rank"] = metrics["ew_market_simple_return"].abs().rank(
        ascending=False, method="min"
    ).loc[candidates.index]
    candidates["dispersion_rank"] = metrics["cross_section_dispersion"].rank(
        ascending=False, method="min"
    ).loc[candidates.index]
    candidates = candidates.sort_values(
        ["abs_market_return_rank", "dispersion_rank"], ascending=[True, True]
    )
    candidates.index.name = "date"
    return candidates


def compute_annual_summary(metrics: pd.DataFrame, corr_window: int, eigen_window: int) -> pd.DataFrame:
    corr_col = f"avg_pairwise_corr_{corr_window}d"
    eigen_col = f"lambda1_fraction_{eigen_window}d"
    rows = []
    for year, frame in metrics.groupby(metrics.index.year):
        log_return = frame["ew_market_log_return"]
        local_wealth = np.exp(log_return.cumsum())
        rows.append(
            {
                "year": int(year),
                "observations": int(len(frame)),
                "market_proxy_return": float(np.expm1(log_return.sum())),
                "market_proxy_annualized_volatility": float(log_return.std(ddof=1) * np.sqrt(242)),
                "within_year_max_drawdown": float((local_wealth / local_wealth.cummax().clip(lower=1.0) - 1).min()),
                "worst_daily_return": float(np.expm1(log_return.min())),
                "best_daily_return": float(np.expm1(log_return.max())),
                "mean_cross_section_dispersion": float(frame["cross_section_dispersion"].mean()),
                f"mean_{corr_col}": float(frame[corr_col].mean()),
                f"peak_{corr_col}": float(frame[corr_col].max()),
                f"mean_{eigen_col}": float(frame[eigen_col].mean()),
                f"peak_{eigen_col}": float(frame[eigen_col].max()),
            }
        )
    return pd.DataFrame(rows)


def find_individual_return_anomalies(returns: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    simple = np.expm1(returns)
    stacked = simple.stack().rename("simple_return").reset_index()
    stacked.columns = ["date", "ticker", "simple_return"]
    anomalies = stacked[stacked["simple_return"].abs() > threshold].copy()
    anomalies["log_return"] = np.log1p(anomalies["simple_return"])
    anomalies["absolute_return"] = anomalies["simple_return"].abs()
    anomalies["review_reason"] = "absolute daily return exceeds 25%; verify suspension, corporate action, or source adjustment"
    return anomalies.sort_values("absolute_return", ascending=False)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def format_time_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def save_figures(
    metrics: pd.DataFrame,
    returns: pd.DataFrame,
    coverage: pd.DataFrame,
    figures: Path,
    corr_window: int,
    eigen_window: int,
) -> None:
    set_plot_style()
    blue, red, gold, teal = "#2F5D8C", "#B23A48", "#D69E2E", "#2A7F62"

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.1, 1.2]})
    axes[0].plot(metrics.index, metrics["ew_cumulative_index"], color=blue, lw=1.8, label="Daily-rebalanced equal weight")
    axes[0].plot(
        metrics.index,
        metrics["geometric_mean_cohort_index"],
        color=gold,
        lw=1.3,
        label="Geometric mean of constituent wealth",
    )
    axes[0].set_title(f"Market proxies from {returns.shape[1]} stocks")
    axes[0].set_ylabel("Growth of 1")
    axes[0].legend(ncol=2)
    axes[1].fill_between(metrics.index, metrics["ew_drawdown"], 0, color=red, alpha=0.75)
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    colors = np.where(metrics["ew_market_simple_return"] >= 0, teal, red)
    axes[2].bar(metrics.index, metrics["ew_market_simple_return"], width=1.0, color=colors, alpha=0.75)
    axes[2].set_ylabel("Daily return")
    axes[2].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    format_time_axis(axes[2])
    fig.tight_layout()
    fig.savefig(figures / "01_market_overview.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(metrics.index, metrics["rolling_vol_20d_annualized"], label="20-day", color=red, lw=1.2)
    axes[0].plot(metrics.index, metrics["rolling_vol_60d_annualized"], label="60-day", color=blue, lw=1.5)
    axes[0].set_title("Risk and market collectivity")
    axes[0].set_ylabel("Annualized volatility")
    axes[0].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[0].legend(ncol=2)
    axes[1].plot(metrics.index, metrics[f"avg_pairwise_corr_{corr_window}d"], color=gold, lw=1.5)
    axes[1].set_ylabel(f"Average correlation\n({corr_window} days)")
    axes[2].plot(metrics.index, metrics[f"lambda1_fraction_{eigen_window}d"], color=teal, lw=1.5)
    axes[2].set_ylabel(f"Largest eigenvalue / N\n({eigen_window} days)")
    format_time_axis(axes[2])
    fig.tight_layout()
    fig.savefig(figures / "02_risk_and_collectivity.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(metrics.index, metrics["breadth_positive_fraction"], color=blue, lw=1.0)
    axes[0].axhline(0.5, color="#666666", lw=0.8, ls="--")
    axes[0].set_title("Cross-sectional breadth and dispersion")
    axes[0].set_ylabel("Positive-return fraction")
    axes[0].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].plot(metrics.index, metrics["cross_section_dispersion"], color=gold, lw=1.0)
    axes[1].set_ylabel("Daily dispersion")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
    axes[2].plot(metrics.index, metrics["abs_return_gt_5pct_fraction"], label="|r| > 5%", color=red, lw=1.0)
    axes[2].plot(metrics.index, metrics["abs_return_gt_9pct_fraction"], label="|r| > 9%", color=teal, lw=1.0)
    axes[2].set_ylabel("Extreme-return fraction")
    axes[2].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[2].legend(ncol=2)
    format_time_axis(axes[2])
    fig.tight_layout()
    fig.savefig(figures / "03_breadth_and_dispersion.png", bbox_inches="tight")
    plt.close(fig)

    market = metrics["ew_market_log_return"].dropna()
    standardized = (market - market.mean()) / market.std(ddof=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(standardized, bins=70, stat="density", color=blue, alpha=0.55, ax=axes[0])
    grid = np.linspace(-5, 5, 500)
    axes[0].plot(grid, stats.norm.pdf(grid), color=red, lw=1.8, label="Standard normal")
    axes[0].set_xlim(-7, 7)
    axes[0].set_title("Standardized market-return distribution")
    axes[0].legend()
    stats.probplot(standardized, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor(blue)
    axes[1].get_lines()[0].set_markeredgecolor(blue)
    axes[1].get_lines()[0].set_markersize(3)
    axes[1].get_lines()[1].set_color(red)
    axes[1].set_title("Normal Q-Q plot")
    fig.tight_layout()
    fig.savefig(figures / "04_market_return_distribution.png", bbox_inches="tight")
    plt.close(fig)

    monthly = np.expm1(metrics["ew_market_log_return"].resample("ME").sum()).to_frame("return")
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month
    heat = monthly.pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(
        heat,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".1%",
        linewidths=0.4,
        cbar_kws={"label": "Monthly return"},
        ax=ax,
    )
    ax.set_title("Equal-weight market proxy: monthly returns")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    fig.tight_layout()
    fig.savefig(figures / "05_monthly_return_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    if "manifest_active_members" not in coverage:
        return
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(coverage.index, coverage["constant_analysis_universe"], color=blue, lw=1.8, label="Stocks used every day")
    ax.plot(coverage.index, coverage["manifest_active_members"], color=red, lw=1.4, label="Membership intervals active")
    ax.set_title("Constant return matrix versus manifest membership dates")
    ax.set_ylabel("Number of stocks")
    ax.legend()
    format_time_axis(ax)
    fig.tight_layout()
    fig.savefig(figures / "06_universe_membership_mismatch.png", bbox_inches="tight")
    plt.close(fig)


def write_text_summary(
    output: Path,
    metrics: pd.DataFrame,
    stock_summary: pd.DataFrame,
    audit: pd.DataFrame,
    metadata: dict,
    corr_window: int,
    eigen_window: int,
) -> None:
    market = metrics["ew_market_log_return"]
    mean_constituent_log = metrics["cross_section_mean_log_return"]
    years = len(market) / 242.0
    annual_return = float(np.exp(market.sum() / years) - 1)
    annual_vol = float(market.std(ddof=1) * np.sqrt(242))
    market_simple = metrics["ew_market_simple_return"]
    sharpe_rf0 = float(market_simple.mean() / market_simple.std(ddof=1) * np.sqrt(242))
    summary = {
        **metadata,
        "market_proxy_annualized_return": annual_return,
        "market_proxy_annualized_volatility": annual_vol,
        "market_proxy_sharpe_rf0": sharpe_rf0,
        "market_proxy_max_drawdown": float(metrics["ew_drawdown"].min()),
        "market_proxy_max_drawdown_date": metrics["ew_drawdown"].idxmin().date().isoformat(),
        "geometric_mean_constituent_annualized_return": float(np.exp(mean_constituent_log.mean() * 242) - 1),
        "geometric_mean_constituent_terminal_index": float(metrics["geometric_mean_cohort_index"].iloc[-1]),
        "daily_rebalanced_ew_terminal_index": float(metrics["ew_cumulative_index"].iloc[-1]),
        "worst_market_day": metrics["ew_market_simple_return"].idxmin().date().isoformat(),
        "worst_market_day_return": float(metrics["ew_market_simple_return"].min()),
        "best_market_day": metrics["ew_market_simple_return"].idxmax().date().isoformat(),
        "best_market_day_return": float(metrics["ew_market_simple_return"].max()),
        "market_daily_skewness": float(stats.skew(market, bias=False)),
        "market_daily_excess_kurtosis": float(stats.kurtosis(market, fisher=True, bias=False)),
        f"peak_avg_pairwise_corr_{corr_window}d": float(metrics[f"avg_pairwise_corr_{corr_window}d"].max()),
        f"peak_avg_pairwise_corr_{corr_window}d_date": metrics[f"avg_pairwise_corr_{corr_window}d"].idxmax().date().isoformat(),
        f"peak_lambda1_fraction_{eigen_window}d": float(metrics[f"lambda1_fraction_{eigen_window}d"].max()),
        f"peak_lambda1_fraction_{eigen_window}d_date": metrics[f"lambda1_fraction_{eigen_window}d"].idxmax().date().isoformat(),
        "median_stock_annualized_return": float(stock_summary["annualized_geometric_return"].median()),
        "median_stock_annualized_volatility": float(stock_summary["annualized_volatility"].median()),
        "median_stock_max_drawdown": float(stock_summary["max_drawdown"].median()),
        "tickers_with_zero_run_gt_5": int((audit["max_consecutive_zero_returns"] > 5).sum()),
        "tickers_with_zero_run_gt_20": int((audit["max_consecutive_zero_returns"] > 20).sum()),
    }
    with (output / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    tables, figures = ensure_dirs(args.output)
    returns, audit, extras = audit_and_load(args.input)
    metrics = compute_metrics(returns, args.corr_window, args.eigen_window, args.eigen_step)
    stock_summary = compute_stock_summary(returns, audit)
    events = build_event_candidates(metrics)
    annual_summary = compute_annual_summary(metrics, args.corr_window, args.eigen_window)
    anomalies = find_individual_return_anomalies(returns)

    audit.to_csv(tables / "01_data_audit_by_ticker.csv", index=False)
    extras["excluded"].to_csv(tables / "02_excluded_stocks.csv", index=False)
    extras["coverage"].to_csv(tables / "03_universe_coverage.csv")
    metrics.to_csv(tables / "04_market_daily_metrics.csv")
    stock_summary.to_csv(tables / "05_stock_summary.csv", index=False)
    events.to_csv(tables / "06_event_candidates.csv")
    returns.to_csv(tables / "07_log_return_matrix.csv")
    annual_summary.to_csv(tables / "08_annual_summary.csv", index=False)
    anomalies.to_csv(tables / "09_individual_return_anomalies.csv", index=False)

    save_figures(metrics, returns, extras["coverage"], figures, args.corr_window, args.eigen_window)
    write_text_summary(
        args.output,
        metrics,
        stock_summary,
        audit,
        extras["summary"],
        args.corr_window,
        args.eigen_window,
    )
    print(json.dumps(extras["summary"], ensure_ascii=False, indent=2))
    print(f"Outputs written to: {args.output}")


if __name__ == "__main__":
    main()
