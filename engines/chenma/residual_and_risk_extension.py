"""Holdout validation of residual stock groups and out-of-sample risk estimates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BLUE = "#245894"
ORANGE = "#db713c"
GREEN = "#287d66"
INK = "#243849"
MUTED = "#536473"


def font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in ["/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def residual_correlation(returns: np.ndarray) -> np.ndarray:
    z = (returns - returns.mean(axis=0)) / returns.std(axis=0, ddof=0)
    corr = z.T @ z / len(z)
    values, vectors = np.linalg.eigh((corr + corr.T) / 2)
    return corr - values[-1] * np.outer(vectors[:, -1], vectors[:, -1])


def adjusted_rand(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand index, invariant to the arbitrary cluster numbering."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Labels must cover the same stocks")
    _, a = np.unique(labels_a, return_inverse=True)
    _, b = np.unique(labels_b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=int)
    np.add.at(table, (a, b), 1)
    choose2 = lambda x: x * (x - 1) / 2
    same = choose2(table).sum()
    rows = choose2(table.sum(axis=1)).sum()
    columns = choose2(table.sum(axis=0)).sum()
    expected = rows * columns / choose2(len(labels_a))
    denominator = (rows + columns) / 2 - expected
    return float((same - expected) / denominator)


def group_contrast(residual: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    n = len(labels)
    same = (labels[:, None] == labels[None, :]) & ~np.eye(n, dtype=bool)
    different = labels[:, None] != labels[None, :]
    within = float(residual[same].mean())
    between = float(residual[different].mean())
    return within, between, within - between


def cluster(returns: np.ndarray, cluster_function: Callable,
            count: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    covariance = np.cov(returns, rowvar=False, ddof=1)
    order, boundaries, labels, _ = cluster_function(returns, covariance,
                                                      n_blocks=count, seed=seed)
    return order, labels, boundaries


def render_residual_holdout(path: Path, early: np.ndarray, late: np.ndarray,
                            order: np.ndarray, boundaries: list[int]) -> None:
    n = len(order)
    clip = float(np.quantile(np.abs(np.r_[early[~np.eye(n, dtype=bool)],
                                          late[~np.eye(n, dtype=bool)]]), .99))
    width, height = 2200, 1300
    size = 864
    lefts = [130, 1190]
    top = 230
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 32), "Residual correlations: groups learned in the earlier sample",
              font=font(43), fill=INK, anchor="mt")
    draw.text((width // 2, 96),
              "Same stock order and color scale in both panels  |  largest market mode removed independently",
              font=font(24), fill=MUTED, anchor="mt")
    for x, matrix, name in zip(lefts, (early, late),
                               ("Discovery: earlier sample", "Holdout: later sample")):
        draw.text((x + size // 2, top - 47), name, font=font(29),
                  fill=INK, anchor="mm")
        values = matrix[np.ix_(order, order)]
        f = np.clip(np.abs(values) / clip, 0, 1)[..., None]
        white = np.full((n, n, 3), 255.0)
        positive = np.array([206., 75., 57.])
        negative = np.array([49., 104., 161.])
        targets = np.where((values >= 0)[..., None], positive, negative)
        rgb = (white + f * (targets - white)).astype(np.uint8)
        rgb[np.diag_indices(n)] = [207, 213, 218]
        image = Image.fromarray(rgb, mode="RGB").resize((size, size),
                                                         Image.Resampling.NEAREST)
        canvas.paste(image, (x, top))
        for boundary in boundaries:
            pixel = x + boundary * size / n
            draw.line((pixel, top, pixel, top + size), fill="#344554", width=2)
            py = top + boundary * size / n
            draw.line((x, py, x + size, py), fill="#344554", width=2)
        draw.rectangle((x, top, x + size, top + size), outline="#344554", width=2)
    draw.text((width // 2, height - 119),
              f"Residual correlation color clipped at ±{clip:.3f}; gray diagonal masked",
              font=font(23), fill=MUTED, anchor="mm")
    draw.text((width // 2, height - 70),
              "Eight groups and within-group order use only discovery returns",
              font=font(23), fill=MUTED, anchor="mm")
    canvas.save(path, format="PNG", optimize=True)


def render_group_stability(path: Path, contingency: np.ndarray,
                           early_late_ari: float, holdout: float, null: np.ndarray,
                           rolling: pd.DataFrame) -> None:
    width, height = 2000, 1320
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 31), "Residual-group stability after removing market mode",
              font=font(42), fill=INK, anchor="mt")
    draw.text((width // 2, 92),
              "Independent eight-group fits in chronological halves; rolling comparisons use disjoint windows",
              font=font(23), fill=MUTED, anchor="mt")
    left, top, cell = 185, 210, 77
    max_count = int(contingency.max())
    for i in range(8):
        draw.text((left - 20, top + i * cell + cell // 2), f"{i + 1}",
                  font=font(23), fill=INK, anchor="rm")
        draw.text((left + i * cell + cell // 2, top - 20), f"{i + 1}",
                  font=font(23), fill=INK, anchor="mm")
        for j in range(8):
            count = int(contingency[i, j])
            frac = count / max_count
            rgb = tuple(int(c) for c in (np.array([248., 251., 253.]) * (1 - frac)
                                         + np.array([37., 95., 145.]) * frac))
            x, y = left + j * cell, top + i * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=rgb,
                           outline="white", width=2)
            draw.text((x + cell // 2, y + cell // 2), str(count),
                      font=font(21), fill="white" if frac > .55 else INK,
                      anchor="mm")
    draw.text((left + 4 * cell, top - 58), "Late group", font=font(26),
              fill=INK, anchor="mm")
    draw.text((left + 4 * cell, top + 8 * cell + 28), "Early group  →  late group stock counts",
              font=font(22), fill=MUTED, anchor="mt")
    info_x = 950
    info = [("Early/late adjusted Rand index", f"{early_late_ari:.3f}"),
            ("Early groups on late residuals", f"{holdout:.3f}"),
            ("Randomized labels: 95th percentile", f"{np.quantile(null, .95):.3f}"),
            ("Randomized labels: maximum", f"{null.max():.3f}"),
            ("Permutation p (one-sided)", f"{(1 + (null >= holdout).sum()) / (len(null) + 1):.4f}")]
    for index, (label, value) in enumerate(info):
        y = top + index * 100
        draw.text((info_x, y), label, font=font(24), fill=MUTED, anchor="lt")
        draw.text((info_x, y + 40), value, font=font(31), fill=BLUE, anchor="lt")
    chart_left, chart_right = 185, width - 80
    chart_top, chart_bottom = 970, 1190
    draw.text((chart_left, chart_top - 42), "Rolling adjusted Rand index vs a past nonoverlapping window",
              font=font(26), fill=INK, anchor="lt")
    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom),
                   outline="#607180", width=2)
    for value in (0, .25, .5, .75, 1):
        py = chart_bottom - value * (chart_bottom - chart_top)
        draw.line((chart_left, py, chart_right, py), fill="#e4eaee", width=2)
        draw.text((chart_left - 13, py), f"{value:.2f}", font=font(19),
                  fill=MUTED, anchor="rm")
    days_all = pd.to_datetime(rolling.end_date).to_numpy(dtype="datetime64[D]").astype(int)
    dmin, dmax = int(days_all.min()), int(days_all.max())
    xpos = lambda day: chart_left + (day - dmin) / (dmax - dmin) * (chart_right - chart_left)
    for year in range(2022, 2027):
        day = int(np.datetime64(f"{year}-01-01", "D").astype(int))
        if dmin <= day <= dmax:
            px = xpos(day)
            draw.line((px, chart_top, px, chart_bottom), fill="#e4eaee", width=2)
            draw.text((px, chart_bottom + 20), str(year), font=font(20),
                      fill=MUTED, anchor="mt")
    for window, color in ((500, BLUE), (750, ORANGE)):
        sample = rolling[rolling.window_days == window]
        days = pd.to_datetime(sample.end_date).to_numpy(dtype="datetime64[D]").astype(int)
        valid = sample.ari_vs_disjoint_window.notna().to_numpy()
        points = [(int(xpos(day)), int(chart_bottom - value * (chart_bottom - chart_top)))
                  for day, value in zip(days[valid], sample.ari_vs_disjoint_window.to_numpy()[valid])]
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
    draw.text((width // 2, height - 39),
              "Blue: 500-day windows   |   Orange: 750-day windows   |   ARI = 1 means identical partitions",
              font=font(21), fill=MUTED, anchor="mm")
    canvas.save(path, format="PNG", optimize=True)


def fixed_portfolios(n_stocks: int, tickers: pd.Index, settings: dict
                     ) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(int(settings["seed"]))
    sizes = [n_stocks] + [10] * int(settings["random_portfolios_10_stock"])
    sizes += [30] * int(settings["random_portfolios_30_stock"])
    weights = np.zeros((n_stocks, len(sizes)))
    manifest = []
    for portfolio_id, size in enumerate(sizes):
        members = np.arange(n_stocks) if portfolio_id == 0 else rng.choice(n_stocks, size, replace=False)
        weights[members, portfolio_id] = 1 / size
        for index in members:
            manifest.append({"portfolio_id": portfolio_id,
                             "portfolio_type": "all_stocks" if portfolio_id == 0 else f"random_{size}_stock",
                             "ticker": str(tickers[index]), "weight": float(1 / size)})
    if not np.allclose(weights.sum(axis=0), 1):
        raise AssertionError("Portfolio weights do not sum to one")
    return weights, pd.DataFrame(manifest)


def covariance_estimators(train: np.ndarray, settings: dict) -> dict[str, np.ndarray]:
    sd = train.std(axis=0, ddof=1)
    z = (train - train.mean(axis=0)) / sd
    corr = z.T @ z / (len(train) - 1)
    values, vectors = np.linalg.eigh((corr + corr.T) / 2)
    sample = corr * sd[:, None] * sd[None, :]
    diagonal = np.diag(sd ** 2)
    estimators = {"sample": sample, "diagonal": diagonal}
    for alpha in settings["shrinkage_levels"]:
        alpha = float(alpha)
        name = f"shrink_{int(round(100 * alpha))}pct"
        estimators[name] = (1 - alpha) * sample + alpha * diagonal
    for k in settings["clip_k"]:
        k = int(k)
        cleaned = values.copy()
        cleaned[:-k] = cleaned[:-k].mean()
        reconstructed = (vectors * cleaned) @ vectors.T
        diag = np.sqrt(np.diag(reconstructed))
        reconstructed /= diag[:, None] * diag[None, :]
        estimators[f"clip_top{k}"] = reconstructed * sd[:, None] * sd[None, :]
    return estimators


def fold_score(covariance: np.ndarray, centered_test: np.ndarray,
               weights: np.ndarray) -> dict:
    forecast = np.sum(weights * (covariance @ weights), axis=0)
    realized = np.mean((centered_test @ weights) ** 2, axis=0)
    if (forecast <= 0).any() or (realized <= 0).any():
        raise ValueError("A portfolio variance is nonpositive")
    forecast_vol = np.sqrt(forecast) * 100
    realized_vol = np.sqrt(realized) * 100
    ratio = realized / forecast
    cholesky = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(cholesky, centered_test.T)
    pseudo_log_score = (2 * np.log(np.diag(cholesky)).sum()
                        + np.mean(np.sum(whitened ** 2, axis=0))) / covariance.shape[0]
    return {"portfolio_vol_mae_pp": float(np.mean(np.abs(forecast_vol - realized_vol))),
            "portfolio_vol_bias_pp": float(np.mean(forecast_vol - realized_vol)),
            "portfolio_variance_qlike": float(np.mean(ratio - np.log(ratio) - 1)),
            "median_realized_to_forecast_variance": float(np.median(ratio)),
            "gaussian_pseudo_log_score_per_stock": float(pseudo_log_score),
            "equal_weight_forecast_vol_percent": float(forecast_vol[0]),
            "equal_weight_realized_vol_percent": float(realized_vol[0])}


def paired_block_bootstrap(folds: pd.DataFrame, method: str, column: str,
                           seed: int, repetitions: int) -> tuple[float, float]:
    sample = folds[folds.method == "sample"].sort_values("test_start_index")
    target = folds[folds.method == method].sort_values("test_start_index")
    if not np.array_equal(sample.test_start_index, target.test_start_index):
        raise AssertionError("Forecast folds do not align")
    difference = sample[column].to_numpy() - target[column].to_numpy()
    count = len(difference)
    rng = np.random.default_rng(seed)
    block = min(3, count)
    means = []
    for _ in range(repetitions):
        starts = rng.integers(0, count - block + 1,
                              size=int(np.ceil(count / block)))
        indices = np.concatenate([np.arange(start, start + block)
                                  for start in starts])[:count]
        means.append(float(difference[indices].mean()))
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def render_risk_comparison(path: Path, summary: pd.DataFrame,
                           methods: list[str]) -> None:
    width, height = 2000, 1190
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 32), "Out-of-sample covariance and portfolio-risk forecasts",
              font=font(42), fill=INK, anchor="mt")
    draw.text((width // 2, 97),
              "Past 500/750 days predict the next 20 trading days  |  fixed long-only portfolios",
              font=font(23), fill=MUTED, anchor="mt")
    left, right = 190, width - 65
    bar_slot = (right - left) / len(methods)
    for title, column, ylow, yhigh, top, bottom in (
        ("Gaussian covariance score improvement vs raw sample (higher is better)",
         "gaussian_score_improvement_vs_sample", -.5, 1.6, 205, 550),
        ("Mean absolute error of daily portfolio volatility, percentage points (lower is better)",
         "portfolio_vol_mae_pp", 0, .7, 690, 1035)):
        draw.text((left, top - 38), title, font=font(25), fill=INK, anchor="lt")
        draw.rectangle((left, top, right, bottom), outline="#607180", width=2)
        ypos = lambda value: bottom - (value - ylow) / (yhigh - ylow) * (bottom - top)
        for value in np.linspace(ylow, yhigh, 5):
            py = ypos(value)
            draw.line((left, py, right, py), fill="#e4eaee", width=2)
            draw.text((left - 13, py), f"{value:.2f}", font=font(19),
                      fill=MUTED, anchor="rm")
        zero = ypos(0)
        for group, method in enumerate(methods):
            center = left + (group + .5) * bar_slot
            for offset, window, color in ((-1, 500, BLUE), (1, 750, ORANGE)):
                row = summary[(summary.window_days == window) & (summary.method == method)].iloc[0]
                value = float(row[column])
                x = center + offset * 28
                baseline = zero if ylow < 0 else bottom
                py = ypos(value)
                draw.rectangle((x - 24, min(py, baseline), x + 24, max(py, baseline)), fill=color)
            if top == 690:
                for line_number, label in enumerate(method.split("_")):
                    draw.text((center, bottom + 19 + 22 * line_number), label,
                              font=font(20), fill=MUTED, anchor="mt")
    draw.text((width // 2, height - 44),
              "Blue: 500-day train   |   Orange: 750-day train   |   Score is Gaussian pseudo likelihood, not a return forecast",
              font=font(20), fill=MUTED, anchor="mm")
    canvas.save(path, format="PNG", optimize=True)


def render_equal_weight(path: Path, folds: pd.DataFrame) -> None:
    data = folds[folds.window_days == 500]
    methods = [("sample", BLUE), ("shrink_20pct", ORANGE), ("clip_top10", GREEN)]
    realized = data[data.method == "sample"].sort_values("test_start_index")
    dates = pd.to_datetime(realized.test_end_date).to_numpy(dtype="datetime64[D]").astype(int)
    dmin, dmax = int(dates.min()), int(dates.max())
    width, height = 2000, 1050
    left, right, top, bottom = 180, 70, 190, 840
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 35), "Equal-weight portfolio: 20-day realized vs predicted risk",
              font=font(42), fill=INK, anchor="mt")
    draw.text((width // 2, 100),
              "500-day training window  |  daily volatility in percent  |  each forecast uses earlier returns only",
              font=font(23), fill=MUTED, anchor="mt")
    ymax = max(3., float(np.ceil(2 * realized.equal_weight_realized_vol_percent.max()) / 2))
    xpos = lambda day: left + (day - dmin) / (dmax - dmin) * (width - left - right)
    ypos = lambda value: bottom - value / ymax * (bottom - top)
    draw.rectangle((left, top, width - right, bottom), outline="#607180", width=2)
    for value in np.linspace(0, ymax, 5):
        py = ypos(value)
        draw.line((left, py, width - right, py), fill="#e4eaee", width=2)
        draw.text((left - 15, py), f"{value:.1f}", font=font(22), fill=MUTED, anchor="rm")
    for year in range(2022, 2027):
        day = int(np.datetime64(f"{year}-01-01", "D").astype(int))
        if dmin <= day <= dmax:
            px = xpos(day)
            draw.line((px, top, px, bottom), fill="#e4eaee", width=2)
            draw.text((px, bottom + 21), str(year), font=font(21), fill=MUTED, anchor="mt")
    realized_points = [(int(xpos(day)), int(ypos(value)))
                       for day, value in zip(dates, realized.equal_weight_realized_vol_percent)]
    draw.line(realized_points, fill="#2d3540", width=5, joint="curve")
    for method, color in methods:
        sample = data[data.method == method].sort_values("test_start_index")
        points = [(int(xpos(day)), int(ypos(value)))
                  for day, value in zip(dates, sample.equal_weight_forecast_vol_percent)]
        draw.line(points, fill=color, width=3, joint="curve")
    labels = [("Realized", "#2d3540"), ("Sample", BLUE),
              ("20% shrinkage", ORANGE), ("Top-10 clipping", GREEN)]
    for index, (name, color) in enumerate(labels):
        x = left + index * 365
        draw.line((x, 922, x + 62, 922), fill=color, width=5)
        draw.text((x + 72, 922), name, font=font(22), fill=MUTED, anchor="lm")
    canvas.save(path, format="PNG", optimize=True)


def run_extension(panel: pd.DataFrame, output: Path, cfg: dict,
                  cluster_function: Callable) -> dict:
    daily = np.diff(np.log(panel.to_numpy(dtype=float)), axis=0)
    dates = panel.index[1:]
    n_times, n_stocks = daily.shape
    group_cfg = cfg["residual_group"]
    risk_cfg = cfg["risk_estimation"]
    group_seed = int(group_cfg["seed"])
    n_groups = int(group_cfg["n_groups"])
    split = pd.Timestamp(cfg["split_date"])
    early_mask = np.asarray(dates < split)
    early, late = daily[early_mask], daily[~early_mask]
    early_residual, late_residual = residual_correlation(early), residual_correlation(late)
    early_order, early_labels, early_bounds = cluster(early, cluster_function, n_groups, group_seed)
    _, late_labels, _ = cluster(late, cluster_function, n_groups, group_seed)
    overlap_ari = adjusted_rand(early_labels, late_labels)
    early_on_early = group_contrast(early_residual, early_labels)
    early_on_late = group_contrast(late_residual, early_labels)
    late_on_late = group_contrast(late_residual, late_labels)
    late_on_early = group_contrast(early_residual, late_labels)
    rng = np.random.default_rng(group_seed)
    repeats = int(group_cfg["permutation_repetitions"])
    null_late = np.array([group_contrast(late_residual, rng.permutation(early_labels))[2]
                          for _ in range(repeats)])
    null_early = np.array([group_contrast(early_residual, rng.permutation(late_labels))[2]
                           for _ in range(repeats)])
    group_tests = pd.DataFrame([
        {"training_period": "early", "evaluation_period": "early",
         "within_residual_mean": early_on_early[0], "between_residual_mean": early_on_early[1],
         "contrast": early_on_early[2], "permutation_p_one_sided": np.nan},
        {"training_period": "early", "evaluation_period": "late",
         "within_residual_mean": early_on_late[0], "between_residual_mean": early_on_late[1],
         "contrast": early_on_late[2],
         "permutation_p_one_sided": (1 + (null_late >= early_on_late[2]).sum()) / (repeats + 1)},
        {"training_period": "late", "evaluation_period": "late",
         "within_residual_mean": late_on_late[0], "between_residual_mean": late_on_late[1],
         "contrast": late_on_late[2], "permutation_p_one_sided": np.nan},
        {"training_period": "late", "evaluation_period": "early",
         "within_residual_mean": late_on_early[0], "between_residual_mean": late_on_early[1],
         "contrast": late_on_early[2],
         "permutation_p_one_sided": (1 + (null_early >= late_on_early[2]).sum()) / (repeats + 1)}])
    group_tests.to_csv(output / "tables/residual_group_holdout_validation.csv", index=False)
    pd.DataFrame({"replicate": np.arange(1, repeats + 1),
                  "early_groups_on_late_null_contrast": null_late,
                  "late_groups_on_early_null_contrast": null_early}).to_csv(
                      output / "tables/residual_group_permutation_null.csv", index=False)
    full_group_data = json.loads((output / "tables/covariance_tau1_order.json").read_text())
    full_labels = np.empty(n_stocks, dtype=int)
    for item in full_group_data["items"]:
        full_labels[item["old_index"]] = item["block"]
    pd.DataFrame({"ticker": panel.columns, "full_sample_group": full_labels,
                  "early_group": early_labels, "late_group": late_labels}).to_csv(
                      output / "tables/residual_group_assignments.csv", index=False)
    contingency = np.zeros((n_groups, n_groups), dtype=int)
    np.add.at(contingency, (early_labels - 1, late_labels - 1), 1)
    pd.DataFrame(contingency, index=[f"early_{j}" for j in range(1, n_groups + 1)],
                 columns=[f"late_{j}" for j in range(1, n_groups + 1)]).to_csv(
                     output / "tables/residual_group_early_late_counts.csv")
    render_residual_holdout(output / "figures/15_residual_correlation_holdout.png",
                            early_residual, late_residual, early_order, early_bounds)

    rolling_rows = []
    assignment_rows = []
    for window in cfg["mode_structure"]["window_days"]:
        window = int(window)
        prior = []
        for end in range(window, n_times + 1, int(cfg["mode_structure"]["step_days"])):
            start = end - window
            piece = daily[start:end]
            _, labels, _ = cluster(piece, cluster_function, n_groups, group_seed)
            residual = residual_correlation(piece)
            earlier = next((item for item in reversed(prior) if item["end"] <= start), None)
            within, between, contrast = group_contrast(residual, labels)
            rolling_rows.append({"window_days": window, "start_date": str(dates[start].date()),
                                 "end_date": str(dates[end - 1].date()),
                                 "end_index_exclusive": end,
                                 "within_residual_mean": within,
                                 "between_residual_mean": between,
                                 "residual_group_contrast": contrast,
                                 "ari_vs_full_sample": adjusted_rand(labels, full_labels),
                                 "ari_vs_disjoint_window": adjusted_rand(labels, earlier["labels"]) if earlier else np.nan,
                                 "disjoint_reference_end_date": str(dates[earlier["end"] - 1].date()) if earlier else ""})
            assignment_rows.extend({"window_days": window, "end_date": str(dates[end - 1].date()),
                                    "ticker": str(ticker), "group": int(group)}
                                   for ticker, group in zip(panel.columns, labels))
            prior.append({"end": end, "labels": labels})
    rolling_groups = pd.DataFrame(rolling_rows)
    rolling_groups.to_csv(output / "tables/residual_group_rolling_metrics.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(output / "tables/residual_group_rolling_assignments.csv", index=False)
    render_group_stability(output / "figures/16_residual_group_stability.png",
                           contingency, overlap_ari, early_on_late[2], null_late, rolling_groups)

    weights, portfolio_manifest = fixed_portfolios(n_stocks, panel.columns, risk_cfg)
    portfolio_manifest.to_csv(output / "tables/risk_portfolio_manifest.csv", index=False)
    methods = ["sample", "diagonal"]
    methods += [f"shrink_{int(round(float(alpha) * 100))}pct"
                for alpha in risk_cfg["shrinkage_levels"]]
    methods += [f"clip_top{int(k)}" for k in risk_cfg["clip_k"]]
    risk_rows = []
    horizon = int(risk_cfg["test_days"])
    step = int(risk_cfg["step_days"])
    for window in risk_cfg["window_days"]:
        window = int(window)
        for end in range(window, n_times - horizon + 1, step):
            train = daily[end - window:end]
            test = daily[end:end + horizon] - train.mean(axis=0)
            estimators = covariance_estimators(train, risk_cfg)
            for name in methods:
                score = fold_score(estimators[name], test, weights)
                risk_rows.append({"window_days": window,
                                  "train_start_date": str(dates[end - window].date()),
                                  "train_end_date": str(dates[end - 1].date()),
                                  "test_start_date": str(dates[end].date()),
                                  "test_end_date": str(dates[end + horizon - 1].date()),
                                  "test_start_index": end, "test_days": horizon,
                                  "method": name, **score})
    risk_folds = pd.DataFrame(risk_rows)
    risk_folds.to_csv(output / "tables/risk_forecast_folds.csv", index=False)
    risk_summary_rows = []
    for window in risk_cfg["window_days"]:
        sub = risk_folds[risk_folds.window_days == int(window)]
        sample = sub[sub.method == "sample"]
        sample_score = sample.gaussian_pseudo_log_score_per_stock.mean()
        for name in methods:
            item = sub[sub.method == name]
            mae_ci = paired_block_bootstrap(sub, name, "portfolio_vol_mae_pp",
                                            int(risk_cfg["seed"]) + int(window),
                                            int(risk_cfg["fold_bootstrap_repetitions"]))
            score_ci = paired_block_bootstrap(sub, name,
                                              "gaussian_pseudo_log_score_per_stock",
                                              int(risk_cfg["seed"]) + int(window) + 1000,
                                              int(risk_cfg["fold_bootstrap_repetitions"]))
            risk_summary_rows.append({"window_days": int(window), "method": name,
                                      "n_folds": len(item),
                                      "gaussian_pseudo_log_score_per_stock": float(item.gaussian_pseudo_log_score_per_stock.mean()),
                                      "gaussian_score_improvement_vs_sample": float(sample_score - item.gaussian_pseudo_log_score_per_stock.mean()),
                                      "score_improvement_ci025": score_ci[0],
                                      "score_improvement_ci975": score_ci[1],
                                      "portfolio_vol_mae_pp": float(item.portfolio_vol_mae_pp.mean()),
                                      "mae_improvement_vs_sample_pp": float(sample.portfolio_vol_mae_pp.mean() - item.portfolio_vol_mae_pp.mean()),
                                      "mae_improvement_ci025_pp": mae_ci[0],
                                      "mae_improvement_ci975_pp": mae_ci[1],
                                      "portfolio_vol_bias_pp": float(item.portfolio_vol_bias_pp.mean()),
                                      "portfolio_variance_qlike": float(item.portfolio_variance_qlike.mean()),
                                      "median_realized_to_forecast_variance": float(item.median_realized_to_forecast_variance.median())})
    risk_summary = pd.DataFrame(risk_summary_rows)
    risk_summary.to_csv(output / "tables/risk_method_summary.csv", index=False)
    render_risk_comparison(output / "figures/17_risk_method_comparison.png",
                           risk_summary, methods)
    render_equal_weight(output / "figures/18_equal_weight_risk_forecast.png", risk_folds)

    lines = ["# 市场模态残差群组与样本外风险估计", "",
             "## 输入与方法", "",
             f"使用 {n_stocks} 只股票的相邻交易日原始收盘价对数收益。群组分析在每个样本中独立估计最大相关矩阵模态并扣除，再以正残差相关的谱嵌入和固定种子聚类为股票分成 {n_groups} 组。相关结构描述不等于因果网络。",
             f"将 {split.date()} 前的 {len(early)} 个收益日作为发现集，之后的 {len(late)} 日作为留出集。早晚分别聚类，用调整兰德指数（ARI）衡量组员是否一致；再把早期分组用于晚期残差相关矩阵，检验组内减组间的平均残差相关。用 {repeats} 次保持组大小不变的股票标签置换形成对照。", "",
             "## 群组结果", "",
             f"早晚独立分组的 ARI 为 {overlap_ari:.3f}。早期分组在早期残差相关的组内−组间差为 {early_on_early[2]:.3f}，在晚期留出数据上为 {early_on_late[2]:.3f}；晚期随机标签对照的第 95 百分位为 {np.quantile(null_late,.95):.3f}，单侧置换 p={(1+(null_late>=early_on_late[2]).sum())/(repeats+1):.4f}。晚期独立分组在早期数据上的反向差值为 {late_on_early[2]:.3f}。", "",
             "| 滚动窗口（日） | 窗口数 | 对不重叠过去窗口的 ARI 中位数 | 范围 |", "|---:|---:|---:|---:|"]
    for window in cfg["mode_structure"]["window_days"]:
        sample = rolling_groups[rolling_groups.window_days == int(window)]
        aris = sample.ari_vs_disjoint_window.dropna()
        lines.append(f"| {window} | {len(sample)} | {aris.median():.3f} | {aris.min():.3f}–{aris.max():.3f} |")
    lines += ["", "上述滚动窗口每 20 日更新；相邻窗口共享大量数据，表中 ARI 特意比较与当前窗口没有共享收益日的过去窗口。留出检验支持这批股票中存在持续的群组结构，但组数八和分组算法仍是研究设定，尚未用全量行业标签确认。", "",
              "## 样本外协方差与风险估计", "",
              "每次仅用过去 500 或 750 个日收益训练，预测接下来的 20 个日收益；测试区间随 20 日步长推进、彼此不重叠。比较未经处理的样本协方差、仅保留方差的对角矩阵、固定 20%/50% 对角收缩，以及保留最大 1/10 个相关特征值并把其余特征值替换为其均值的谱截断。谱截断后重新归一化对角元，再乘回各股票样本标准差。所有方法均使用训练段信息；这些超参数作为固定的探索性对照，不在每个测试块上重新拟合。", "",
              f"风险组合为 1 个 {n_stocks} 股票等权组合、{int(risk_cfg['random_portfolios_10_stock'])} 个随机 10 股票等权组合和 {int(risk_cfg['random_portfolios_30_stock'])} 个随机 30 股票等权组合；抽样种子固定，权重在所有测试期与方法间相同。评估指标：（1）测试收益在预测协方差下的每股 Gaussian 伪对数评分（越低越好）；（2）预测日波动率与随后 20 日实现均方根波动率的平均绝对误差，单位百分点（越低越好）；（3）方差 QLIKE 与校准比。这些是协方差预测评估，不是组合收益预测或交易策略。", "",
              "| 训练窗口 | 方法 | 测试块 | Gaussian 评分 | 比原样本改善 | 波动率 MAE（百分点） | 校准比中位数 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for item in risk_summary.itertuples():
        lines.append(f"| {item.window_days} | {item.method} | {item.n_folds} | {item.gaussian_pseudo_log_score_per_stock:.3f} | {item.gaussian_score_improvement_vs_sample:+.3f} | {item.portfolio_vol_mae_pp:.3f} | {item.median_realized_to_forecast_variance:.2f} |")
    lines += ["", "评分和波动误差可能给出不同排名：收缩降低了某些组合波动的绝对误差，也可能使方差系统性偏低；谱截断在 Gaussian 伪评分上可能优于原样本，但未必改善组合波动误差。表中的改进是这批固定股票和固定组合上的探索性结果。CSV 还给出相对原样本方法的三测试块移动 bootstrap 描述性区间。", "",
              "## 限制与文件", "",
              "20 日实现波动本身估计噪声较大；高斯伪评分也不意味着日收益真的服从正态分布。训练窗口重叠、股票池固定且可能存在存活者偏差；输入价格是否复权、停牌与成分历史规则均未获确认。不能从这些评分推出可交易收益或将其用于实际风控。", "",
              "图：`figures/15_residual_correlation_holdout.png`、`figures/16_residual_group_stability.png`、`figures/17_risk_method_comparison.png`、`figures/18_equal_weight_risk_forecast.png`。表：`tables/residual_group_*` 和 `tables/risk_*`。", ""]
    (output / "residual_and_risk_report.md").write_text("\n".join(lines))
    metadata = {"n_stocks": n_stocks, "n_daily_returns": n_times,
                "residual_group_count": n_groups,
                "early_late_group_ari": overlap_ari,
                "early_groups_on_late_contrast": early_on_late[2],
                "early_groups_on_late_permutation_p": float(group_tests.iloc[1].permutation_p_one_sided),
                "risk_methods": methods,
                "risk_fold_counts": {str(window): int(len(risk_folds[(risk_folds.window_days == int(window))
                                                                   & (risk_folds.method == "sample")]))
                                     for window in risk_cfg["window_days"]},
                "group_config": group_cfg, "risk_config": risk_cfg}
    (output / "residual_and_risk_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata
