#!/usr/bin/env python3
"""
沪深300固定股票队列的相关矩阵、随机矩阵理论与投资组合分析。

分析口径仿照课件《金融商品价格的相关》：
1. 标准化对数收益 -> 相关矩阵。
2. Marchenko-Pastur 上下界与经验特征值谱比较。
3. 累积强度、特征向量成分和参与率。
4. 市场、组结构和噪声矩阵的分解。
5. 原始协方差、课件式硬阈值矩阵与实用 RMT 特征值截尾矩阵的组合对比。

所有回测使用过去窗口估计、后续持有，不使用未来数据。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


SEED = 20261006
TRADING_DAYS = 242


@dataclass
class RMTResult:
    correlation: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    q: float
    lambda_minus: float
    lambda_plus: float
    high_mask: np.ndarray
    low_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--returns",
        type=Path,
        default=here / "tables" / "07_log_return_matrix.csv",
        help="Date-indexed log-return matrix produced by analyze_hs300.py",
    )
    parser.add_argument("--output", type=Path, default=here)
    parser.add_argument("--portfolio-assets", type=int, default=100)
    parser.add_argument("--training-days", type=int, default=504)
    parser.add_argument("--rebalance-days", type=int, default=21)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    return parser.parse_args()


def set_style() -> None:
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


def load_returns(path: Path) -> pd.DataFrame:
    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    returns = returns.sort_index().astype(float)
    if returns.isna().any().any():
        raise ValueError("Return matrix contains missing values")
    if returns.index.duplicated().any() or not returns.index.is_monotonic_increasing:
        raise ValueError("Return dates must be unique and sorted")
    return returns


def mp_bounds(t: int, n: int) -> tuple[float, float, float]:
    q = t / n
    root = math.sqrt(1.0 / q)
    return q, (1.0 - root) ** 2, (1.0 + root) ** 2


def standardize(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=1, keepdims=True)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("At least one return series has zero or invalid variance")
    return centered / scale


def rmt_decomposition(values: np.ndarray) -> RMTResult:
    t, n = values.shape
    z = standardize(values)
    correlation = z.T @ z / (t - 1)
    correlation = (correlation + correlation.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    q, lambda_minus, lambda_plus = mp_bounds(t, n)
    tolerance = 1e-10
    high_mask = eigenvalues > lambda_plus + tolerance
    low_mask = eigenvalues < lambda_minus - tolerance
    return RMTResult(
        correlation=correlation,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        q=q,
        lambda_minus=lambda_minus,
        lambda_plus=lambda_plus,
        high_mask=high_mask,
        low_mask=low_mask,
    )


def orient_vector(vector: np.ndarray) -> np.ndarray:
    return vector if vector.sum() >= 0 else -vector


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], 1)]


def distribution_shape(values: np.ndarray) -> tuple[float, float]:
    """返回偏度和超额峰度；常数数列没有可解释的形状。"""
    if np.ptp(values) < 1e-12:
        return float("nan"), float("nan")
    return (
        float(stats.skew(values, bias=False)),
        float(stats.kurtosis(values, fisher=True, bias=False)),
    )


def normalize_to_correlation(matrix: np.ndarray) -> np.ndarray:
    diagonal = np.clip(np.diag(matrix), 1e-12, None)
    inverse_scale = 1.0 / np.sqrt(diagonal)
    normalized = matrix * inverse_scale[:, None] * inverse_scale[None, :]
    normalized = (normalized + normalized.T) / 2
    np.fill_diagonal(normalized, 1.0)
    return normalized


def reconstruct(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    matrix = (eigenvectors * eigenvalues) @ eigenvectors.T
    return (matrix + matrix.T) / 2


def correlation_components(result: RMTResult) -> dict[str, np.ndarray]:
    values = result.eigenvalues
    vectors = result.eigenvectors
    market_values = np.zeros_like(values)
    market_values[-1] = values[-1]
    group_values = np.zeros_like(values)
    group_values[result.high_mask] = values[result.high_mask]
    group_values[-1] = 0.0
    noise_values = values.copy()
    noise_values[result.high_mask] = 0.0
    market = reconstruct(market_values, vectors)
    group = reconstruct(group_values, vectors)
    noise = reconstruct(noise_values, vectors)
    signal = market + group
    return {"market": market, "group": group, "noise": noise, "signal": signal}


def clipped_correlation(result: RMTResult) -> np.ndarray:
    clipped = result.eigenvalues.copy()
    noise_mask = ~result.high_mask
    clipped[noise_mask] = clipped[noise_mask].mean()
    return normalize_to_correlation(reconstruct(clipped, result.eigenvectors))


def lecture_signal_correlation(result: RMTResult) -> np.ndarray:
    kept = np.where(result.high_mask, result.eigenvalues, 0.0)
    return normalize_to_correlation(reconstruct(kept, result.eigenvectors))


def lecture_noise_correlation(result: RMTResult) -> np.ndarray:
    kept = np.where(result.high_mask, 0.0, result.eigenvalues)
    return normalize_to_correlation(reconstruct(kept, result.eigenvectors))


def matrix_summary(values: np.ndarray, label: str) -> dict:
    result = rmt_decomposition(values)
    off = off_diagonal(result.correlation)
    ipr = np.sum(result.eigenvectors**4, axis=0)
    skewness, excess_kurtosis = distribution_shape(off)
    return {
        "label": label,
        "T": int(values.shape[0]),
        "N": int(values.shape[1]),
        "Q_T_over_N": result.q,
        "lambda_minus": result.lambda_minus,
        "lambda_plus": result.lambda_plus,
        "lambda_max": float(result.eigenvalues[-1]),
        "lambda_max_fraction": float(result.eigenvalues[-1] / values.shape[1]),
        "high_outlier_count": int(result.high_mask.sum()),
        "low_outlier_count": int(result.low_mask.sum()),
        "high_outlier_strength_fraction": float(result.eigenvalues[result.high_mask].sum() / values.shape[1]),
        "avg_off_diagonal_correlation": float(off.mean()),
        "off_diagonal_variance": float(off.var(ddof=0)),
        "off_diagonal_skewness": skewness,
        "off_diagonal_excess_kurtosis": excess_kurtosis,
        "largest_mode_standard_PR": float(1.0 / ipr[-1]),
        "largest_mode_course_Gamma": float(3.0 / ipr[-1]),
        "matrix_rank": int(np.linalg.matrix_rank(result.correlation, tol=1e-10)),
        "condition_number_positive_spectrum": float(
            result.eigenvalues[-1] / result.eigenvalues[result.eigenvalues > 1e-10].min()
        ),
    }


def full_sample_tables(
    returns: pd.DataFrame, result: RMTResult, tables: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eigenvalues_desc = result.eigenvalues[::-1]
    vectors_desc = result.eigenvectors[:, ::-1].copy()
    for column in range(vectors_desc.shape[1]):
        vectors_desc[:, column] = orient_vector(vectors_desc[:, column])
    ipr = np.sum(vectors_desc**4, axis=0)
    eigen_table = pd.DataFrame(
        {
            "rank_descending": np.arange(1, len(eigenvalues_desc) + 1),
            "eigenvalue": eigenvalues_desc,
            "strength_fraction": eigenvalues_desc / len(eigenvalues_desc),
            "cumulative_strength_fraction": np.cumsum(eigenvalues_desc) / len(eigenvalues_desc),
            "above_mp_upper": eigenvalues_desc > result.lambda_plus,
            "below_mp_lower": eigenvalues_desc < result.lambda_minus,
            "inverse_participation_ratio": ipr,
            "standard_participation_ratio": 1.0 / ipr,
            "course_participation_Gamma": 3.0 / ipr,
        }
    )
    eigen_table.to_csv(tables / "02_full_sample_eigenvalues.csv", index=False)

    selected_ranks = sorted(
        set([1, 2, 3, int(result.high_mask.sum()), int(result.high_mask.sum()) + 1, len(eigenvalues_desc)])
    )
    selected_ranks = [rank for rank in selected_ranks if 1 <= rank <= len(eigenvalues_desc)]
    vector_rows = []
    for rank in selected_ranks:
        vector = vectors_desc[:, rank - 1]
        order = np.argsort(np.abs(vector))[::-1][:20]
        for loading_rank, index in enumerate(order, start=1):
            vector_rows.append(
                {
                    "eigen_rank_descending": rank,
                    "eigenvalue": eigenvalues_desc[rank - 1],
                    "loading_rank": loading_rank,
                    "ticker": returns.columns[index],
                    "loading": vector[index],
                    "absolute_loading": abs(vector[index]),
                }
            )
    loadings = pd.DataFrame(vector_rows)
    loadings.to_csv(tables / "03_top_eigenvector_loadings.csv", index=False)

    components = correlation_components(result)
    component_rows = []
    for name, matrix in {"empirical": result.correlation, **components}.items():
        normalized = result.correlation if name == "empirical" else normalize_to_correlation(matrix)
        off = off_diagonal(normalized)
        skewness, excess_kurtosis = distribution_shape(off)
        component_rows.append(
            {
                "component": name,
                "mean_off_diagonal": float(off.mean()),
                "std_off_diagonal": float(off.std(ddof=0)),
                "skewness": skewness,
                "excess_kurtosis": excess_kurtosis,
                "min_off_diagonal": float(off.min()),
                "max_off_diagonal": float(off.max()),
                "raw_trace": float(np.trace(matrix)),
            }
        )
    component_table = pd.DataFrame(component_rows)
    component_table.to_csv(tables / "04_matrix_component_statistics.csv", index=False)
    return eigen_table, loadings, component_table


def stock_count_sensitivity(
    returns: pd.DataFrame, asset_counts: Iterable[int], repeats: int = 30
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    all_indices = np.arange(returns.shape[1])
    for n in asset_counts:
        run_count = 1 if n == returns.shape[1] else repeats
        for repetition in range(run_count):
            chosen = all_indices if n == returns.shape[1] else np.sort(rng.choice(all_indices, n, replace=False))
            summary = matrix_summary(returns.iloc[:, chosen].to_numpy(), f"N={n}")
            summary["repetition"] = repetition
            rows.append(summary)
    return pd.DataFrame(rows)


def window_length_sensitivity(returns: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    lengths = [120, 242, 484, 968, len(returns)]
    rows = []
    for length in lengths:
        frame = returns.loc[:, columns].iloc[-length:]
        summary = matrix_summary(frame.to_numpy(), f"last_{length}_days")
        summary["start_date"] = frame.index.min().date().isoformat()
        summary["end_date"] = frame.index.max().date().isoformat()
        rows.append(summary)
    return pd.DataFrame(rows)


def regime_sensitivity(returns: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    regimes = [
        ("2019_recovery_rally", "2019-01-01", "2019-12-31", "rising"),
        ("2020_covid_shock_recovery", "2020-01-01", "2020-12-31", "shock_and_recovery"),
        ("2021_low_correlation_rotation", "2021-01-01", "2021-12-31", "rotation"),
        ("2022_bear_market", "2022-01-01", "2022-12-31", "falling"),
        ("2023_weak_low_volatility", "2023-01-01", "2023-12-31", "weak_and_calm"),
        ("2024_stimulus_reversal", "2024-01-01", "2024-12-31", "reversal_high_volatility"),
        ("2025_tariff_shock_recovery", "2025-01-01", "2025-12-31", "shock_and_rising"),
        ("2026_partial_decline", "2026-01-01", "2026-09-28", "partial_falling"),
    ]
    rows = []
    for label, start, end, trend in regimes:
        frame = returns.loc[start:end, columns]
        summary = matrix_summary(frame.to_numpy(), label)
        simple_market = np.expm1(frame).mean(axis=1)
        summary["start_date"] = frame.index.min().date().isoformat()
        summary["end_date"] = frame.index.max().date().isoformat()
        summary["trend_label"] = trend
        summary["equal_weight_period_return"] = float((1.0 + simple_market).prod() - 1.0)
        summary["equal_weight_annualized_volatility"] = float(simple_market.std(ddof=1) * np.sqrt(TRADING_DAYS))
        rows.append(summary)
    return pd.DataFrame(rows)


def covariance_models(log_returns: np.ndarray) -> tuple[dict[str, np.ndarray], RMTResult]:
    result = rmt_decomposition(log_returns)
    vol = log_returns.std(axis=0, ddof=1)
    raw = result.correlation * vol[:, None] * vol[None, :]
    lecture = lecture_signal_correlation(result) * vol[:, None] * vol[None, :]
    clipped = clipped_correlation(result) * vol[:, None] * vol[None, :]
    noise = lecture_noise_correlation(result) * vol[:, None] * vol[None, :]
    return {"raw_gmv": raw, "lecture_signal_gmv": lecture, "rmt_clipped_gmv": clipped, "noise_only_diagnostic": noise}, result


def regularize_covariance(covariance: np.ndarray, ridge_fraction: float = 1e-6) -> np.ndarray:
    scale = float(np.trace(covariance) / covariance.shape[0])
    return (covariance + covariance.T) / 2 + np.eye(covariance.shape[0]) * scale * ridge_fraction


def long_only_minimum_variance(
    covariance: np.ndarray,
    max_weight: float,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    n = covariance.shape[0]
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError("max_weight is infeasible for the selected asset count")
    cov = regularize_covariance(covariance)
    x0 = np.full(n, 1.0 / n) if initial is None else np.clip(initial, 0.0, max_weight)
    x0 = x0 / x0.sum()

    def objective(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * cov @ weights

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, max_weight)
    return weights / weights.sum()


def portfolio_backtest(
    returns: pd.DataFrame,
    columns: list[str],
    training_days: int,
    rebalance_days: int,
    max_weight: float,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log_returns = returns.loc[:, columns]
    simple_returns = np.expm1(log_returns)
    strategies = ["equal_weight", "raw_gmv", "lecture_signal_gmv", "rmt_clipped_gmv"]
    daily_results = {strategy: [] for strategy in strategies}
    daily_dates: list[pd.Timestamp] = []
    current_weights: dict[str, np.ndarray | None] = {strategy: None for strategy in strategies}
    previous_targets: dict[str, np.ndarray | None] = {strategy: None for strategy in strategies}
    rebalance_rows = []
    latest_targets: dict[str, np.ndarray] = {}
    cost_rate = transaction_cost_bps / 10000.0

    positions = list(range(training_days, len(log_returns), rebalance_days))
    for period_number, start in enumerate(positions):
        end = min(start + rebalance_days, len(log_returns))
        training = log_returns.iloc[start - training_days : start]
        models, rmt = covariance_models(training.to_numpy())
        targets = {"equal_weight": np.full(len(columns), 1.0 / len(columns))}
        for strategy in strategies[1:]:
            targets[strategy] = long_only_minimum_variance(
                models[strategy],
                max_weight=max_weight,
                initial=previous_targets[strategy],
            )
        latest_targets = targets

        turnover_map: dict[str, float] = {}
        for strategy in strategies:
            before = current_weights[strategy]
            turnover = 0.0 if before is None else float(0.5 * np.abs(targets[strategy] - before).sum())
            turnover_map[strategy] = turnover
            current_weights[strategy] = targets[strategy].copy()
            previous_targets[strategy] = targets[strategy].copy()

        for strategy in strategies:
            weights = targets[strategy]
            predicted_vol = float(np.sqrt(weights @ models.get(strategy, models["raw_gmv"]) @ weights * TRADING_DAYS))
            rebalance_rows.append(
                {
                    "rebalance_date": log_returns.index[start],
                    "period_number": period_number,
                    "strategy": strategy,
                    "turnover": turnover_map[strategy],
                    "transaction_cost": turnover_map[strategy] * cost_rate,
                    "predicted_annualized_volatility": predicted_vol,
                    "effective_number_of_stocks": float(1.0 / np.sum(weights**2)),
                    "maximum_weight": float(weights.max()),
                    "mp_lambda_plus": rmt.lambda_plus,
                    "high_outlier_count": int(rmt.high_mask.sum()),
                }
            )

        holding = simple_returns.iloc[start:end]
        for local_day, (date, row) in enumerate(holding.iterrows()):
            if not daily_dates or daily_dates[-1] != date:
                daily_dates.append(date)
            asset_returns = row.to_numpy(float)
            for strategy in strategies:
                weights = current_weights[strategy]
                gross_return = float(weights @ asset_returns)
                if local_day == 0:
                    net_return = (1.0 - turnover_map[strategy] * cost_rate) * (1.0 + gross_return) - 1.0
                else:
                    net_return = gross_return
                daily_results[strategy].append(net_return)
                denominator = 1.0 + gross_return
                if denominator <= 0:
                    raise RuntimeError("Portfolio wealth became non-positive")
                current_weights[strategy] = weights * (1.0 + asset_returns) / denominator

    daily = pd.DataFrame(daily_results, index=pd.DatetimeIndex(daily_dates, name="date"))
    performance_rows = []
    for strategy in strategies:
        series = daily[strategy]
        wealth = (1.0 + series).cumprod()
        # Initial capital is 1; include it as a possible peak before day one.
        drawdown = wealth / wealth.cummax().clip(lower=1.0) - 1.0
        years = len(series) / TRADING_DAYS
        rebalance_subset = pd.DataFrame(rebalance_rows)
        rebalance_subset = rebalance_subset[rebalance_subset["strategy"] == strategy]
        performance_rows.append(
            {
                "strategy": strategy,
                "start_date": series.index.min().date().isoformat(),
                "end_date": series.index.max().date().isoformat(),
                "observations": int(len(series)),
                "annualized_geometric_return_net": float(wealth.iloc[-1] ** (1.0 / years) - 1.0),
                "annualized_volatility": float(series.std(ddof=1) * np.sqrt(TRADING_DAYS)),
                "sharpe_rf0": float(series.mean() / series.std(ddof=1) * np.sqrt(TRADING_DAYS)),
                "max_drawdown": float(drawdown.min()),
                "terminal_wealth_net": float(wealth.iloc[-1]),
                "average_rebalance_turnover": float(rebalance_subset["turnover"].mean()),
                "average_effective_number_of_stocks": float(
                    rebalance_subset["effective_number_of_stocks"].mean()
                ),
                "average_predicted_annualized_volatility": float(
                    rebalance_subset["predicted_annualized_volatility"].mean()
                ),
            }
        )
    performance = pd.DataFrame(performance_rows)
    rebalances = pd.DataFrame(rebalance_rows)
    weight_rows = []
    for strategy, weights in latest_targets.items():
        for ticker, weight in zip(columns, weights):
            weight_rows.append({"strategy": strategy, "ticker": ticker, "weight": float(weight)})
    latest_weights = pd.DataFrame(weight_rows)
    return daily, performance, rebalances, latest_weights


def mp_density(grid: np.ndarray, q: float, lower: float, upper: float) -> np.ndarray:
    density = np.zeros_like(grid)
    mask = (grid >= lower) & (grid <= upper) & (grid > 0)
    density[mask] = q / (2.0 * np.pi * grid[mask]) * np.sqrt(
        (upper - grid[mask]) * (grid[mask] - lower)
    )
    return density


def save_core_figures(
    returns: pd.DataFrame,
    result: RMTResult,
    eigen_table: pd.DataFrame,
    figures: Path,
) -> None:
    set_style()
    blue, orange, green, red, purple = "#2F5D8C", "#D69E2E", "#2A7F62", "#B23A48", "#7656A7"
    empirical_off = off_diagonal(result.correlation)
    rng = np.random.default_rng(SEED)
    gaussian = rng.normal(size=returns.shape)
    gaussian_corr = np.corrcoef(gaussian, rowvar=False)
    random_off = off_diagonal(gaussian_corr)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    bins = np.linspace(min(empirical_off.min(), random_off.min()), empirical_off.max(), 75)
    ax.hist(random_off, bins=bins, density=True, color=orange, alpha=0.55, label="Gaussian random matrix")
    ax.hist(empirical_off, bins=bins, density=True, color=blue, alpha=0.65, label="Empirical stocks")
    ax.axvline(empirical_off.mean(), color=blue, lw=1.8, ls="--", label=f"Empirical mean = {empirical_off.mean():.3f}")
    ax.set_title("Off-diagonal correlation distribution")
    ax.set_xlabel("Pairwise correlation")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "01_correlation_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    bulk_limit = max(result.lambda_plus * 1.5, 3.0)
    bulk_values = result.eigenvalues[result.eigenvalues <= bulk_limit]
    bins = np.linspace(0, bulk_limit, 56)
    axes[0].hist(bulk_values, bins=bins, weights=np.full(len(bulk_values), 1 / len(result.eigenvalues) / (bins[1]-bins[0])),
                 color=blue, alpha=0.68, label="Displayed eigenvalues (no renormalization)")
    grid = np.linspace(max(1e-6, result.lambda_minus * 0.5), result.lambda_plus * 1.15, 600)
    axes[0].plot(
        grid,
        mp_density(grid, result.q, result.lambda_minus, result.lambda_plus),
        color=red,
        lw=2.0,
        label="Marchenko-Pastur density",
    )
    axes[0].axvline(result.lambda_minus, color=purple, ls="--", lw=1.2, label=f"MP lower = {result.lambda_minus:.2f}")
    axes[0].axvline(result.lambda_plus, color=green, ls="--", lw=1.2, label=f"MP upper = {result.lambda_plus:.2f}")
    axes[0].set_xlim(0, bulk_limit)
    axes[0].set_title(f"Bulk view: {len(result.eigenvalues)-len(bulk_values)} large eigenvalues outside view")
    axes[0].set_xlabel("Eigenvalue")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9)

    ranks = eigen_table["rank_descending"]
    axes[1].plot(ranks, eigen_table["eigenvalue"], color=blue, lw=1.5)
    axes[1].axhline(result.lambda_plus, color=green, ls="--", lw=1.2, label="MP upper")
    axes[1].axhline(result.lambda_minus, color=purple, ls="--", lw=1.2, label="MP lower")
    axes[1].scatter([1], [result.eigenvalues[-1]], color=red, s=45, zorder=3)
    axes[1].annotate(
        f"lambda1 = {result.eigenvalues[-1]:.2f}",
        xy=(1, result.eigenvalues[-1]),
        xytext=(24, result.eigenvalues[-1] * 0.75),
        arrowprops={"arrowstyle": "->", "color": red},
        color=red,
    )
    axes[1].set_yscale("log")
    axes[1].set_title(f"Full spectrum: {int(result.high_mask.sum())} above MP upper")
    axes[1].set_xlabel("Rank, largest to smallest")
    axes[1].set_ylabel("Eigenvalue, log scale")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "02_eigenvalue_spectrum.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    axes[0].plot(ranks, eigen_table["cumulative_strength_fraction"], color=blue, lw=1.8)
    high_count = int(result.high_mask.sum())
    high_strength = float(result.eigenvalues[result.high_mask].sum() / len(result.eigenvalues))
    axes[0].scatter([high_count], [high_strength], color=red, s=45)
    axes[0].annotate(
        f"{high_count} high modes explain {high_strength:.1%}",
        xy=(high_count, high_strength),
        xytext=(high_count + 30, max(0.15, high_strength - 0.12)),
        arrowprops={"arrowstyle": "->", "color": red},
        color=red,
    )
    axes[0].set_title("Cumulative eigenvalue strength")
    axes[0].set_xlabel("Number of largest modes retained")
    axes[0].set_ylabel("Fraction of total variance")
    axes[0].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].scatter(
        eigen_table["eigenvalue"],
        eigen_table["standard_participation_ratio"],
        s=15,
        alpha=0.65,
        color=blue,
    )
    axes[1].axvline(result.lambda_plus, color=green, ls="--", lw=1.2, label="MP upper")
    axes[1].axvline(result.lambda_minus, color=purple, ls="--", lw=1.2, label="MP lower")
    axes[1].set_xscale("log")
    axes[1].set_title("Eigenvector participation")
    axes[1].set_xlabel("Eigenvalue, log scale")
    axes[1].set_ylabel("Standard participation ratio")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "03_strength_and_participation.png", bbox_inches="tight")
    plt.close(fig)

    vectors_desc = result.eigenvectors[:, ::-1].copy()
    selected = [1, 2, high_count, min(high_count + 1, returns.shape[1])]
    selected = list(dict.fromkeys(selected))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    grid_x = np.linspace(-4, 4, 400)
    for axis, rank in zip(axes.flat, selected):
        vector = orient_vector(vectors_desc[:, rank - 1]) * np.sqrt(returns.shape[1])
        sns.histplot(vector, bins=32, stat="density", color=blue, alpha=0.62, ax=axis)
        axis.plot(grid_x, stats.norm.pdf(grid_x), color=red, lw=1.5, label="N(0,1)")
        axis.set_title(f"Eigenvector rank {rank}, lambda={result.eigenvalues[::-1][rank-1]:.2f}")
        axis.set_xlabel("Scaled component sqrt(N) * u_i")
        axis.set_ylabel("Density")
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "04_eigenvector_component_distributions.png", bbox_inches="tight")
    plt.close(fig)

    components = correlation_components(result)
    distance = np.sqrt(np.clip(0.5 * (1.0 - result.correlation), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    matrices = [
        ("Empirical", result.correlation),
        ("Market mode", normalize_to_correlation(components["market"])),
        ("Group modes", normalize_to_correlation(components["group"])),
        ("Noise remainder", normalize_to_correlation(components["noise"])),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, (title, matrix) in zip(axes.flat, matrices):
        image = axis.imshow(matrix[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
    cbar.set_label("Correlation")
    fig.suptitle("Correlation matrix decomposition, clustered stock order", y=0.995)
    fig.subplots_adjust(top=0.94, right=0.90, wspace=0.08, hspace=0.14)
    fig.savefig(figures / "05_matrix_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def save_sensitivity_figures(
    stock_results: pd.DataFrame,
    window_results: pd.DataFrame,
    regime_results: pd.DataFrame,
    figures: Path,
) -> None:
    set_style()
    blue, orange, green, red = "#2F5D8C", "#D69E2E", "#2A7F62", "#B23A48"
    grouped = stock_results.groupby("N")
    means = grouped.mean(numeric_only=True)
    stds = grouped.std(numeric_only=True).fillna(0.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, column, title, ylabel in [
        (axes[0], "lambda_max_fraction", "Largest market mode", "lambda1 / N"),
        (axes[1], "high_outlier_count", "Modes above MP upper edge", "Count"),
        (axes[2], "avg_off_diagonal_correlation", "Average pairwise correlation", "Correlation"),
    ]:
        axis.errorbar(
            means.index,
            means[column],
            yerr=stds[column],
            color=blue,
            marker="o",
            capsize=3,
            lw=1.5,
        )
        axis.set_title(title)
        axis.set_xlabel("Number of stocks N")
        axis.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(figures / "06_stock_count_sensitivity.png", bbox_inches="tight")
    plt.close(fig)

    ordered = window_results.sort_values("T")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(ordered["T"], ordered["lambda_plus"], color=red, marker="o", label="MP upper")
    axes[0].plot(ordered["T"], ordered["lambda_minus"], color=green, marker="o", label="MP lower")
    axes[0].set_title("Noise band versus window length")
    axes[0].set_xlabel("Trading days T")
    axes[0].set_ylabel("Eigenvalue boundary")
    axes[0].legend()
    axes[1].plot(ordered["T"], ordered["lambda_max_fraction"], color=blue, marker="o")
    axes[1].set_title("Largest market mode")
    axes[1].set_xlabel("Trading days T")
    axes[1].set_ylabel("lambda1 / N")
    axes[2].plot(ordered["T"], ordered["high_outlier_count"], color=orange, marker="o")
    axes[2].set_title("Detected high outliers")
    axes[2].set_xlabel("Trading days T")
    axes[2].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(figures / "07_window_length_sensitivity.png", bbox_inches="tight")
    plt.close(fig)

    regime = regime_results.copy()
    labels = [label.replace("_", "\n") for label in regime["label"]]
    x = np.arange(len(regime))
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].bar(x, regime["equal_weight_period_return"], color=np.where(regime["equal_weight_period_return"] >= 0, green, red))
    axes[0].set_ylabel("Period return")
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[0].set_title("Same 100 stocks across different market regimes")
    axes[1].bar(x, regime["lambda_max_fraction"], color=blue)
    axes[1].set_ylabel("lambda1 / N")
    axes[2].bar(x - 0.18, regime["avg_off_diagonal_correlation"], width=0.36, color=orange, label="Average correlation")
    axes[2].bar(x + 0.18, regime["high_outlier_strength_fraction"], width=0.36, color=blue, label="High-mode strength")
    axes[2].set_ylabel("Fraction")
    axes[2].set_xticks(x, labels, rotation=0)
    axes[2].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "08_market_regime_comparison.png", bbox_inches="tight")
    plt.close(fig)


def save_portfolio_figures(
    daily: pd.DataFrame,
    performance: pd.DataFrame,
    weights: pd.DataFrame,
    figures: Path,
) -> None:
    set_style()
    colors = {
        "equal_weight": "#666666",
        "raw_gmv": "#2F5D8C",
        "lecture_signal_gmv": "#D69E2E",
        "rmt_clipped_gmv": "#2A7F62",
    }
    labels = {
        "equal_weight": "Equal weight",
        "raw_gmv": "Raw covariance GMV",
        "lecture_signal_gmv": "Lecture hard-filter GMV",
        "rmt_clipped_gmv": "RMT clipped GMV",
    }
    wealth = (1.0 + daily).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1.0) - 1.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [1.8, 1.0]})
    for strategy in daily.columns:
        axes[0].plot(wealth.index, wealth[strategy], color=colors[strategy], lw=1.6, label=labels[strategy])
        axes[1].plot(drawdown.index, drawdown[strategy], color=colors[strategy], lw=1.2, label=labels[strategy])
    axes[0].set_title("Out-of-sample rolling portfolio backtest, net of 10 bps turnover cost")
    axes[0].set_ylabel("Growth of 1")
    axes[0].legend(ncol=2)
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(figures / "09_portfolio_backtest.png", bbox_inches="tight")
    plt.close(fig)

    order = performance.set_index("strategy").loc[list(colors)]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(x, order["annualized_geometric_return_net"], color=[colors[s] for s in order.index])
    axes[0].set_title("Annualized net return")
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].bar(x, order["annualized_volatility"], color=[colors[s] for s in order.index])
    axes[1].set_title("Annualized volatility")
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[2].bar(x, order["max_drawdown"], color=[colors[s] for s in order.index])
    axes[2].set_title("Maximum drawdown")
    axes[2].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for axis in axes:
        axis.set_xticks(x, [labels[s] for s in order.index], rotation=24, ha="right")
    fig.tight_layout()
    fig.savefig(figures / "10_portfolio_performance_comparison.png", bbox_inches="tight")
    plt.close(fig)

    selected = weights[weights["strategy"].isin(["raw_gmv", "lecture_signal_gmv", "rmt_clipped_gmv"])].copy()
    pivot = selected.pivot(index="ticker", columns="strategy", values="weight").fillna(0.0)
    top = pivot.max(axis=1).nlargest(25).index
    pivot = pivot.loc[top]
    fig, ax = plt.subplots(figsize=(13, 7))
    positions = np.arange(len(pivot))
    width = 0.26
    for offset, strategy in zip([-width, 0, width], ["raw_gmv", "lecture_signal_gmv", "rmt_clipped_gmv"]):
        ax.bar(positions + offset, pivot[strategy], width=width, color=colors[strategy], label=labels[strategy])
    ax.set_xticks(positions, pivot.index, rotation=60, ha="right")
    ax.set_ylabel("Latest target weight")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("Largest latest portfolio weights")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(figures / "11_latest_portfolio_weights.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output
    tables = output / "rmt_tables"
    figures = output / "rmt_figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    returns = load_returns(args.returns)
    full_result = rmt_decomposition(returns.to_numpy())
    summary = matrix_summary(returns.to_numpy(), "full_2019_2026")
    pd.DataFrame([summary]).to_csv(tables / "01_full_sample_summary.csv", index=False)
    eigen_table, loadings, component_table = full_sample_tables(returns, full_result, tables)

    rng = np.random.default_rng(SEED)
    random_order = list(np.array(returns.columns)[rng.permutation(returns.shape[1])])
    fixed_100 = random_order[:100]
    portfolio_columns = random_order[: args.portfolio_assets]
    pd.DataFrame(
        {
            "order": np.arange(1, len(random_order) + 1),
            "ticker": random_order,
            "in_fixed_100": [ticker in fixed_100 for ticker in random_order],
            "in_portfolio_universe": [ticker in portfolio_columns for ticker in random_order],
        }
    ).to_csv(tables / "05_seeded_stock_selection.csv", index=False)

    stock_results = stock_count_sensitivity(returns, [30, 50, 100, 150, 200, 250, 288], repeats=30)
    stock_results.to_csv(tables / "06_stock_count_sensitivity.csv", index=False)
    stock_summary = (
        stock_results.groupby("N")
        .agg(
            repetitions=("repetition", "count"),
            lambda_max_fraction_mean=("lambda_max_fraction", "mean"),
            lambda_max_fraction_std=("lambda_max_fraction", "std"),
            avg_correlation_mean=("avg_off_diagonal_correlation", "mean"),
            avg_correlation_std=("avg_off_diagonal_correlation", "std"),
            high_outlier_count_mean=("high_outlier_count", "mean"),
            high_outlier_count_std=("high_outlier_count", "std"),
            high_outlier_strength_mean=("high_outlier_strength_fraction", "mean"),
            high_outlier_strength_std=("high_outlier_strength_fraction", "std"),
        )
        .reset_index()
        .fillna(0.0)
    )
    stock_summary.to_csv(tables / "06b_stock_count_summary.csv", index=False)
    window_results = window_length_sensitivity(returns, fixed_100)
    window_results.to_csv(tables / "07_window_length_sensitivity.csv", index=False)
    regime_results = regime_sensitivity(returns, fixed_100)
    regime_results.to_csv(tables / "08_market_regime_sensitivity.csv", index=False)

    daily, performance, rebalances, latest_weights = portfolio_backtest(
        returns,
        portfolio_columns,
        training_days=args.training_days,
        rebalance_days=args.rebalance_days,
        max_weight=args.max_weight,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    daily.to_csv(tables / "09_portfolio_daily_returns.csv")
    performance.to_csv(tables / "10_portfolio_performance.csv", index=False)
    rebalances.to_csv(tables / "11_portfolio_rebalances.csv", index=False)
    latest_weights.to_csv(tables / "12_latest_portfolio_weights.csv", index=False)

    save_core_figures(returns, full_result, eigen_table, figures)
    save_sensitivity_figures(stock_results, window_results, regime_results, figures)
    save_portfolio_figures(daily, performance, latest_weights, figures)

    output_summary = {
        "source_return_matrix": str(args.returns),
        "seed": SEED,
        "full_sample": summary,
        "portfolio_configuration": {
            "asset_count": args.portfolio_assets,
            "training_days": args.training_days,
            "rebalance_days": args.rebalance_days,
            "max_weight": args.max_weight,
            "transaction_cost_bps_per_one_way_turnover": args.transaction_cost_bps,
            "backtest_start": daily.index.min().date().isoformat(),
            "backtest_end": daily.index.max().date().isoformat(),
        },
        "portfolio_performance": performance.to_dict(orient="records"),
    }
    with (output / "rmt_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output_summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
