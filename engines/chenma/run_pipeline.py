"""Reproducible daily raw-price KL analysis following Chen and Ma (2018)."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PAPER = "https://doi.org/10.1016/j.cjph.2018.03.020"
MODES = {"largest": -1, "second_largest": -2, "44th_smallest": 43, "144th_smallest": 143}
BLUE, ORANGE, GREEN, GREY = "#245894", "#db713c", "#287d66", "#aab5bf"


def load_panel(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(folder.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(folder)
    series, manifest = [], []
    for path in files:
        frame = pd.read_pickle(path)
        if not {"Date", "ClosePrice"}.issubset(frame.columns):
            raise ValueError(f"Required columns missing from {path.name}")
        dates = pd.to_datetime(frame.Date, errors="raise")
        prices = pd.to_numeric(frame.ClosePrice, errors="raise").to_numpy(dtype=float)
        if dates.isna().any() or dates.duplicated().any() or not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(f"Invalid dates or prices in {path.name}")
        series.append(pd.Series(prices, index=dates, name=path.stem))
        manifest.append({"file": path.name, "rows": len(frame),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    panel = pd.concat(series, axis=1, join="inner").sort_index()
    if panel.isna().any().any() or panel.index.has_duplicates:
        raise ValueError("Aligned panel is incomplete")
    return panel, pd.DataFrame(manifest)


def analyze(log_prices: np.ndarray, tau: int, source: str, sample: str,
            keep_detail: bool = False) -> tuple[list[dict], np.ndarray, dict]:
    returns = log_prices[tau:] - log_prices[:-tau]
    standard_deviations = returns.std(axis=0, ddof=0)
    if (standard_deviations <= 1e-12).any():
        raise ValueError(f"Zero-variance returns at tau={tau}")
    z = ((returns - returns.mean(axis=0)) / standard_deviations).T
    n, t = z.shape
    c = z @ z.T / t
    eigenvalues, eigenvectors = np.linalg.eigh((c + c.T) / 2)
    if not np.isclose(eigenvalues.sum(), n, atol=1e-7):
        raise AssertionError("The eigenvalue sum is not the stock count")
    detail = {}
    rows = []
    for name, index in MODES.items():
        eigenvalue = eigenvalues[index]
        b = eigenvectors[:, index] @ z / np.sqrt(t * eigenvalue)
        if not np.isclose(b @ b, 1, atol=1e-7):
            raise AssertionError("K–L time component is not unit normalized")
        frequency = np.fft.rfftfreq(t, d=1)[1:]
        power = np.abs(np.fft.rfft(b))[1:] ** 2 / t
        ranks = np.arange(1, len(frequency) + 1)
        # Preserve the paper's approximate *relative* fit band (202–2101 / 11700).
        lo = max(1, int(np.ceil(202 / 11700 * t)))
        hi = min(len(frequency), int(np.floor(2101 / 11700 * t)))
        fit = (ranks >= lo) & (ranks <= hi) & (power > 0)
        x, y = np.log(frequency[fit]), np.log(power[fit])
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        r2 = 1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append({"source": source, "sample": sample, "tau_days": tau, "mode": name,
                     "n_stocks": n, "n_times": t, "eigenvalue": float(eigenvalue),
                     "variance_share": float(eigenvalue / n), "alpha": float(slope),
                     "fit_r2": float(r2), "fit_rank_low": lo, "fit_rank_high": hi,
                     "low_frequency_share_period_ge_20d": float(power[frequency <= .05].sum() / power.sum())})
        if keep_detail:
            detail[name] = {"time": b, "frequency": frequency, "power": power,
                            "slope": slope, "intercept": intercept, "lo": lo, "hi": hi}
    return rows, eigenvalues, detail


def permute_daily_returns(log_prices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    daily = np.diff(log_prices, axis=0)
    return np.vstack([log_prices[0], log_prices[0] + np.cumsum(daily[rng.permutation(len(daily))], axis=0)])


def font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in ["/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def chart(path: Path, title: str, xlabel: str, ylabel: str, series: list[dict],
          xlog: bool = False, ylog: bool = False, ylim: tuple[float, float] | None = None) -> None:
    """Render a compact, legible PNG line/scatter chart with Pillow."""
    width, height = 1800, 1050
    left, right, top, bottom = 175, 70, 190, 160
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    normalized = []
    for item in series:
        x = np.asarray(item["x"], dtype=float)
        y = np.asarray(item["y"], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if xlog: valid &= x > 0
        if ylog: valid &= y > 0
        x, y = x[valid], y[valid]
        if xlog: x = np.log10(x)
        if ylog: y = np.log10(y)
        normalized.append((item, x, y))
    all_x = np.concatenate([x for _, x, _ in normalized if len(x)])
    all_y = np.concatenate([y for _, _, y in normalized if len(y)])
    xmin, xmax = float(all_x.min()), float(all_x.max())
    ymin, ymax = float(all_y.min()), float(all_y.max())
    if ylim is not None: ymin, ymax = ylim
    if xmin == xmax: xmax += 1
    if ymin == ymax: ymax += 1
    dx, dy = .035 * (xmax - xmin), .07 * (ymax - ymin)
    xmin = xmin - dx
    if not xlog and float(all_x.min()) >= 0:
        xmin = max(0, xmin)
    xmax, ymin, ymax = xmax + dx, ymin - dy, ymax + dy
    sx = lambda value: left + (value - xmin) / (xmax - xmin) * (width - left - right)
    sy = lambda value: height - bottom - (value - ymin) / (ymax - ymin) * (height - top - bottom)
    draw.text((width / 2, 35), title, font=font(40), fill="#202a35", anchor="mt")
    draw.rectangle((left, top, width - right, height - bottom), outline="#58636e", width=2)
    for i in range(6):
        xx = xmin + i * (xmax - xmin) / 5
        yy = ymin + i * (ymax - ymin) / 5
        px, py = sx(xx), sy(yy)
        draw.line((px, top, px, height - bottom), fill="#e9edf0", width=2)
        draw.line((left, py, width - right, py), fill="#e9edf0", width=2)
        xlab = 10 ** xx if xlog else xx
        ylab = 10 ** yy if ylog else yy
        draw.text((px, height - bottom + 22), f"{xlab:.3g}", font=font(24), fill="#384552", anchor="mt")
        draw.text((left - 18, py), f"{ylab:.3g}", font=font(24), fill="#384552", anchor="rm")
    draw.text((width / 2, height - 55), xlabel, font=font(30), fill="#202a35", anchor="mm")
    draw.text((left, 105), ylabel, font=font(27), fill="#394856", anchor="lm")
    legend_count = 0
    for j, (item, x, y) in enumerate(normalized):
        if not len(x): continue
        coords = [(int(sx(a)), int(sy(b))) for a, b in zip(x, y)]
        color = item.get("color", BLUE)
        if item.get("scatter"):
            # Dense eigenvalue clouds use a small pixel mark.
            radius = item.get("radius", 1)
            for px, py in coords:
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
        else:
            if len(coords) > 1:
                draw.line(coords, fill=color, width=item.get("width", 4), joint="curve")
            for px, py in coords[::max(1, len(coords) // 60)]:
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
        if item.get("label"):
            lx = left + (legend_count % 4) * 345
            ly = top - 43 + (legend_count // 4) * 35
            draw.line((lx, ly, lx + 32, ly), fill=color, width=5)
            draw.text((lx + 40, ly), item["label"], font=font(22), fill="#273541", anchor="lm")
            legend_count += 1
    image.save(path, format="PNG", optimize=True)


def make_figures(metrics: pd.DataFrame, eigen: pd.DataFrame, details: dict, out: Path) -> None:
    full = metrics[(metrics.source == "observed") & (metrics["sample"] == "full")]
    cloud = eigen[eigen.rank_ascending <= eigen.n_stocks.iloc[0] - 2]
    lead = full[full["mode"] == "largest"].sort_values("tau_days")
    second = full[full["mode"] == "second_largest"].sort_values("tau_days")
    chart(out / "01_eigenvalue_map.png", "Cross-stock correlation eigenvalues, tau = 1–250",
          "Return horizon tau (trading days)", "Eigenvalue (log scale)",
          [{"x": cloud.tau_days, "y": cloud.eigenvalue.clip(lower=1e-6), "color": GREY, "scatter": True},
           {"x": second.tau_days, "y": second.eigenvalue, "color": ORANGE, "label": "Second mode"},
           {"x": lead.tau_days, "y": lead.eigenvalue, "color": BLUE, "label": "Market mode"}], ylog=True)
    for tau in (1, 250):
        market, bulk = details[tau]["largest"], details[tau]["44th_smallest"]
        chart(out / f"02_time_modes_tau{tau}.png", f"KL time components at tau = {tau} days",
              "Return observation index", "Unit-variance time component",
              [{"x": np.arange(len(market["time"])), "y": market["time"] * np.sqrt(len(market["time"])),
                "color": BLUE, "label": "Market mode", "width": 3},
               {"x": np.arange(len(bulk["time"])), "y": bulk["time"] * np.sqrt(len(bulk["time"])),
                "color": ORANGE, "label": "44th-smallest mode", "width": 2}])
        chart(out / f"03_power_spectra_tau{tau}.png", f"KL mode power spectra at tau = {tau} days",
              "Frequency (cycles / trading day)", "Power (log scale)",
              [{"x": market["frequency"], "y": market["power"], "color": BLUE, "label": "Market mode", "width": 3},
               {"x": bulk["frequency"], "y": bulk["power"], "color": ORANGE,
                "label": "44th-smallest mode", "width": 2}], xlog=True, ylog=True)
    bulk = full[full["mode"] == "44th_smallest"].sort_values("tau_days")
    chart(out / "04_spectral_slope.png", "Power-spectrum slope across all 250 horizons",
          "Return horizon tau (trading days)", "Slope alpha",
          [{"x": lead.tau_days, "y": lead.alpha, "color": BLUE, "label": "Market mode"},
           {"x": second.tau_days, "y": second.alpha, "color": GREEN, "label": "Second mode"},
           {"x": bulk.tau_days, "y": bulk.alpha, "color": ORANGE, "label": "44th-smallest mode"}])
    chart(out / "05_variance_and_low_frequency.png", "Market-mode variance and low-frequency power",
          "Return horizon tau (trading days)", "Share of total",
          [{"x": lead.tau_days, "y": lead.variance_share, "color": BLUE,
            "label": "Cross-stock variance"},
           {"x": lead.tau_days, "y": lead.low_frequency_share_period_ge_20d,
            "color": ORANGE, "label": "Power at periods >=20d"}], ylim=(0, 1))
    control = metrics[metrics.source.str.startswith("shuffle") & (metrics["mode"] == "largest")]
    chart(out / "06_shuffle_control.png", "Market-mode slope versus shuffled-date controls",
          "Return horizon tau (trading days)", "Slope alpha",
          [{"x": lead.tau_days, "y": lead.alpha, "color": BLUE, "label": "Observed"},
           {"x": control.tau_days, "y": control.alpha, "color": "#788895",
            "label": "Date-shuffled controls", "scatter": True, "radius": 4}])


def spectral_block_order(daily: np.ndarray, covariance: np.ndarray,
                         n_blocks: int = 8, seed: int = 2018) -> tuple[np.ndarray, list[int], np.ndarray, dict]:
    """Find a row/column permutation from correlations after removing the market mode."""
    n = daily.shape[1]
    corr = np.corrcoef(daily, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    residual = corr - eigenvalues[-1] * np.outer(eigenvectors[:, -1], eigenvectors[:, -1])
    np.fill_diagonal(residual, 0)
    affinity = np.maximum(residual, 0)
    degree = affinity.sum(axis=1)
    if (degree <= 0).any():
        raise ValueError("At least one stock has no positive residual similarity")
    normalized = affinity / np.sqrt(degree[:, None] * degree[None, :])
    _, vectors = np.linalg.eigh(normalized)
    embedding = vectors[:, -n_blocks:]
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    best_cost = np.inf
    best_labels = None
    for _ in range(20):
        centers = [embedding[rng.integers(n)]]
        for _ in range(1, n_blocks):
            distances = np.min(((embedding[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
            if distances.sum() <= 0:
                raise ValueError("Degenerate spectral embedding")
            centers.append(embedding[rng.choice(n, p=distances / distances.sum())])
        centers = np.asarray(centers)
        for _ in range(100):
            distances = ((embedding[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            if len(np.unique(labels)) != n_blocks:
                break
            next_centers = np.asarray([embedding[labels == j].mean(axis=0) for j in range(n_blocks)])
            if np.allclose(next_centers, centers, atol=1e-10):
                centers = next_centers
                break
            centers = next_centers
        if len(np.unique(labels)) != n_blocks:
            continue
        cost = float(np.sum((embedding - centers[labels]) ** 2))
        if cost < best_cost:
            best_cost, best_labels = cost, labels.copy()
    if best_labels is None:
        raise ValueError("Spectral grouping failed to find nonempty groups")

    groups = sorted(range(n_blocks), key=lambda j: np.median(np.flatnonzero(best_labels == j)))
    order_parts = []
    block_by_original = np.empty(n, dtype=int)
    sizes = []
    for block, label in enumerate(groups, start=1):
        indices = np.flatnonzero(best_labels == label)
        block_by_original[indices] = block
        sub = affinity[np.ix_(indices, indices)]
        local_degree = sub.sum(axis=1)
        if len(indices) > 2 and (local_degree > 0).all():
            local = sub / np.sqrt(local_degree[:, None] * local_degree[None, :])
            _, local_vectors = np.linalg.eigh(local)
            position = local_vectors[:, -2]
            if np.corrcoef(position, indices)[0, 1] < 0:
                position = -position
            indices = indices[np.argsort(position, kind="stable")]
        order_parts.append(indices)
        sizes.append(len(indices))
    order = np.concatenate(order_parts)
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise AssertionError("The row/column order is not a permutation")
    within = (block_by_original[:, None] == block_by_original[None, :]) & ~np.eye(n, dtype=bool)
    between = block_by_original[:, None] != block_by_original[None, :]
    stats = {"block_count": n_blocks, "block_sizes": sizes,
             "within_block_mean_covariance": float(covariance[within].mean()),
             "between_block_mean_covariance": float(covariance[between].mean()),
             "within_between_mean_ratio": float(covariance[within].mean() / covariance[between].mean()),
             "ordering_method": "top-8 spectral embedding of positive residual correlation after removing largest correlation eigenmode; seeded k-means; within-block spectral ordering",
             "ordering_seed": seed}
    return order, np.cumsum(sizes).tolist()[:-1], block_by_original, stats


def make_tau1_covariance_figure(panel: pd.DataFrame, path: Path,
                                order_path: Path) -> dict:
    """Plot original and block-reordered covariance of adjacent-day raw log returns."""
    daily = np.diff(np.log(panel.to_numpy(dtype=float)), axis=0)
    covariance = np.cov(daily, rowvar=False, ddof=1)
    n = covariance.shape[0]
    if covariance.shape != (len(panel.columns), len(panel.columns)):
        raise AssertionError("Unexpected covariance shape")
    if not np.allclose(covariance, covariance.T, atol=1e-14):
        raise AssertionError("Covariance matrix is not symmetric")
    order, block_boundaries, block_by_original, block_stats = spectral_block_order(daily, covariance)
    reordered = covariance[np.ix_(order, order)]
    if not np.allclose(reordered, reordered.T, atol=1e-14):
        raise AssertionError("The reordered covariance is not symmetric")
    if not np.array_equal(np.sort(reordered.ravel()), np.sort(covariance.ravel())):
        raise AssertionError("The reordering changed covariance values")
    order_path.write_text(json.dumps({"method": block_stats["ordering_method"],
                                      "seed": block_stats["ordering_seed"],
                                      "items": [{"new_index": new_index, "old_index": int(old_index),
                                                 "ticker": str(panel.columns[old_index]),
                                                 "block": int(block_by_original[old_index])}
                                                for new_index, old_index in enumerate(order)]},
                                     ensure_ascii=False, indent=2))

    off_diagonal = covariance[~np.eye(n, dtype=bool)]
    negative_limit = min(0.0, float(off_diagonal.min()))
    full_positive_limit = float(covariance.max())
    off_positive_limit = float(np.quantile(off_diagonal, .99))
    if off_positive_limit <= 0:
        raise ValueError("The off-diagonal upper color scale is not positive")

    def colored(values: np.ndarray, positive_limit: float, mask_diagonal: bool = False) -> Image.Image:
        negative_scale = abs(negative_limit) if negative_limit < 0 else positive_limit
        positive_fraction = np.clip(values / positive_limit, 0, 1)
        negative_fraction = np.clip(-values / negative_scale, 0, 1)
        rgb = np.full(values.shape + (3,), 255.0)
        rgb -= positive_fraction[..., None] * np.array([80, 218, 207])
        rgb -= negative_fraction[..., None] * np.array([221, 164, 105])
        rgb = rgb.astype(np.uint8)
        if mask_diagonal:
            rgb[np.diag_indices(n)] = [205, 211, 216]
        return Image.fromarray(rgb, mode="RGB")

    matrix_size = 864
    cell = matrix_size / n
    width, height = 2150, 1335
    left_x, right_x, top_y = 125, 1160, 230
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 42), "Tau = 1: daily log-return covariance", font=font(46),
              fill="#1e2d3b", anchor="mt")
    draw.text((width // 2, 103),
              f"{n} stocks  |  {len(daily)} return observations  |  "
              f"{panel.index.min().date()} to {panel.index.max().date()}",
              font=font(25), fill="#536473", anchor="mt")

    exchange_boundary = sum(name.endswith(".SZ") for name in panel.columns)
    views = [
        (left_x, "Original order (diagonal included)", covariance,
         full_positive_limit, False, [exchange_boundary]),
        (right_x, "Reordered off-diagonal (8 groups)", reordered,
         off_positive_limit, True, block_boundaries),
    ]
    for x, title, values, positive_limit, mask_diagonal, boundaries in views:
        draw.text((x + matrix_size // 2, 175), title, font=font(29),
                  fill="#263846", anchor="mt")
        image = colored(values, positive_limit, mask_diagonal).resize(
            (matrix_size, matrix_size), Image.Resampling.NEAREST)
        canvas.paste(image, (x, top_y))
        draw.rectangle((x - 1, top_y - 1, x + matrix_size, top_y + matrix_size),
                       outline="#526170", width=2)
        for boundary_index in boundaries:
            if 0 < boundary_index < n:
                boundary = boundary_index * cell
                draw.line((x + boundary, top_y, x + boundary, top_y + matrix_size),
                          fill="#132a3a", width=3)
                draw.line((x, top_y + boundary, x + matrix_size, top_y + boundary),
                          fill="#132a3a", width=3)
        for tick in np.unique(np.linspace(0, n, 5).astype(int)):
            position = x + tick * cell
            draw.line((position, top_y + matrix_size, position, top_y + matrix_size + 9),
                      fill="#526170", width=2)
            draw.text((position, top_y + matrix_size + 16), str(tick), font=font(22),
                      fill="#455564", anchor="mt")
            draw.text((x - 15, top_y + tick * cell), str(tick), font=font(22),
                      fill="#455564", anchor="rm")
        draw.text((x + matrix_size // 2, top_y + matrix_size + 55),
                  "Stock index (ticker order)" if not mask_diagonal else "Stock index (spectral group order)",
                  font=font(24), fill="#364655", anchor="mt")
        bar_y = top_y + matrix_size + 103
        half = matrix_size // 2
        gradient = np.concatenate((np.linspace(negative_limit, 0, half, endpoint=False),
                                   np.linspace(0, positive_limit, matrix_size - half)))
        bar = colored(gradient[None, :], positive_limit).resize(
            (matrix_size, 24), Image.Resampling.NEAREST)
        canvas.paste(bar, (x, bar_y))
        draw.rectangle((x, bar_y, x + matrix_size - 1, bar_y + 23),
                       outline="#687887", width=1)
        for value, position, anchor in ((negative_limit, x, "lt"),
                                        (0.0, x + half, "mt"),
                                        (positive_limit, x + matrix_size, "rt")):
            draw.text((position, bar_y + 32), f"{value * 1e4:.2f}",
                      font=font(22), fill="#40515f", anchor=anchor)
    draw.text((width // 2, height - 51),
              "Color scale: covariance x 10^4  |  Blue: negative; red: positive; grey: masked diagonal",
              font=font(24), fill="#40515f", anchor="mm")
    canvas.save(path, format="PNG", optimize=True)
    return {"tau1_covariance_min": float(covariance.min()),
            "tau1_covariance_max": full_positive_limit,
            "tau1_off_diagonal_min": float(off_diagonal.min()),
            "tau1_off_diagonal_p99": off_positive_limit,
            "tau1_off_diagonal_negative_fraction": float(np.mean(off_diagonal < 0)),
            "tau1_return_observations": len(daily), **block_stats}


def make_market_removed_covariance_figure(panel: pd.DataFrame, order_path: Path,
                                          path: Path) -> dict[str, float]:
    """Remove one standardized common PC and compare covariances in the same stock order."""
    daily = np.diff(np.log(panel.to_numpy(dtype=float)), axis=0)
    centered = daily - daily.mean(axis=0)
    standard_deviations = centered.std(axis=0, ddof=1)
    standardized = centered / standard_deviations
    correlation = standardized.T @ standardized / (len(daily) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    market_vector = eigenvectors[:, -1]
    market_score = standardized @ market_vector
    residual_standardized = standardized - np.outer(market_score, market_vector)
    residual_returns = residual_standardized * standard_deviations
    original = np.cov(daily, rowvar=False, ddof=1)
    residual = np.cov(residual_returns, rowvar=False, ddof=1)
    reconstructed = np.diag(standard_deviations) @ (
        correlation - eigenvalues[-1] * np.outer(market_vector, market_vector)
    ) @ np.diag(standard_deviations)
    if not np.allclose(residual, reconstructed, atol=1e-12):
        raise AssertionError("Residual covariance does not match one-mode subtraction")

    mapping = json.loads(order_path.read_text())["items"]
    n = len(panel.columns)
    order = np.asarray([item["old_index"] for item in mapping], dtype=int)
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise AssertionError("Saved group order is not a permutation")
    if any(mapping[i]["ticker"] != panel.columns[order[i]] for i in range(n)):
        raise AssertionError("Saved group order does not match current input tickers")
    groups = np.asarray([item["block"] for item in mapping], dtype=int)
    boundaries = np.flatnonzero(np.diff(groups)) + 1
    diagonal_mask = ~np.eye(n, dtype=bool)
    original_off = original[diagonal_mask]
    residual_off = residual[diagonal_mask]
    clip = float(np.quantile(np.abs(original_off), .99))
    if clip <= 0:
        raise ValueError("Invalid covariance color scale")

    def colored(values: np.ndarray) -> Image.Image:
        scaled = np.clip(values / clip, -1, 1)
        positive = np.maximum(scaled, 0)
        negative = np.maximum(-scaled, 0)
        rgb = np.full(values.shape + (3,), 255.0)
        rgb -= positive[..., None] * np.asarray([80, 218, 204])
        rgb -= negative[..., None] * np.asarray([223, 165, 100])
        rgb = rgb.astype(np.uint8)
        if values.ndim == 2 and values.shape == (n, n):
            rgb[np.diag_indices(n)] = [205, 211, 216]
        return Image.fromarray(rgb, mode="RGB")

    matrix_size = 864
    cell = matrix_size / n
    width, height = 2150, 1335
    left_x, right_x, top_y = 125, 1160, 230
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    share = float(eigenvalues[-1] / n)
    draw.text((width // 2, 42), "Tau = 1: removing the leading common mode",
              font=font(46), fill="#1e2d3b", anchor="mt")
    draw.text((width // 2, 103),
              f"{n} stocks  |  {len(daily)} daily returns  |  leading correlation mode: {share:.1%} of standardized variance",
              font=font(25), fill="#536473", anchor="mt")
    views = [(left_x, "Original covariance", original),
             (right_x, "Covariance after mode removal", residual)]
    for x, title, values in views:
        draw.text((x + matrix_size // 2, 175), title, font=font(29),
                  fill="#263846", anchor="mt")
        reordered = values[np.ix_(order, order)]
        image = colored(reordered).resize((matrix_size, matrix_size), Image.Resampling.NEAREST)
        canvas.paste(image, (x, top_y))
        draw.rectangle((x - 1, top_y - 1, x + matrix_size, top_y + matrix_size),
                       outline="#526170", width=2)
        for boundary_index in boundaries:
            position = int(boundary_index) * cell
            draw.line((x + position, top_y, x + position, top_y + matrix_size),
                      fill="#132a3a", width=3)
            draw.line((x, top_y + position, x + matrix_size, top_y + position),
                      fill="#132a3a", width=3)
        for tick in np.unique(np.linspace(0, n, 5).astype(int)):
            position = x + tick * cell
            draw.line((position, top_y + matrix_size, position, top_y + matrix_size + 9),
                      fill="#526170", width=2)
            draw.text((position, top_y + matrix_size + 16), str(tick), font=font(22),
                      fill="#455564", anchor="mt")
            draw.text((x - 15, top_y + tick * cell), str(tick), font=font(22),
                      fill="#455564", anchor="rm")
        draw.text((x + matrix_size // 2, top_y + matrix_size + 55),
                  "Stock index (same spectral group order)",
                  font=font(24), fill="#364655", anchor="mt")
        bar_y = top_y + matrix_size + 103
        gradient = np.linspace(-clip, clip, matrix_size)
        bar = colored(gradient[None, :]).resize((matrix_size, 24), Image.Resampling.NEAREST)
        canvas.paste(bar, (x, bar_y))
        draw.rectangle((x, bar_y, x + matrix_size - 1, bar_y + 23),
                       outline="#687887", width=1)
        for value, position, anchor in ((-clip, x, "lt"),
                                        (0.0, x + matrix_size // 2, "mt"),
                                        (clip, x + matrix_size, "rt")):
            draw.text((position, bar_y + 32), f"{value * 1e4:.2f}",
                      font=font(22), fill="#40515f", anchor=anchor)
    draw.text((width // 2, height - 51),
              "Same stock order and color scale  |  covariance x 10^4  |  diagonal masked  |  blue negative, red positive",
              font=font(23), fill="#40515f", anchor="mm")
    canvas.save(path, format="PNG", optimize=True)
    return {"market_mode_standardized_variance_share": share,
            "residual_total_variance_share": float(np.trace(residual) / np.trace(original)),
            "original_off_diagonal_mean_abs_covariance": float(np.mean(np.abs(original_off))),
            "residual_off_diagonal_mean_abs_covariance": float(np.mean(np.abs(residual_off))),
            "original_negative_off_diagonal_fraction": float(np.mean(original_off < 0)),
            "residual_negative_off_diagonal_fraction": float(np.mean(residual_off < 0)),
            "shared_color_scale_abs_covariance_p99": clip}


def make_market_mode_tau1_time_figure(panel: pd.DataFrame, path: Path) -> dict:
    """Show the leading tau=1 KL time coordinate against real trading dates."""
    daily = np.diff(np.log(panel.to_numpy(dtype=float)), axis=0)
    standardized = (daily - daily.mean(axis=0)) / daily.std(axis=0, ddof=0)
    correlation = standardized.T @ standardized / len(daily)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    loading = eigenvectors[:, -1]
    score = standardized @ loading / np.sqrt(eigenvalues[-1])
    equal_weight_standardized = standardized.mean(axis=1)
    if np.corrcoef(score, equal_weight_standardized)[0, 1] < 0:
        score, loading = -score, -loading
    n_times = len(score)
    # In the KL convention used elsewhere in this pipeline, b(t) has unit L2 norm.
    # sqrt(T)*b(t) is shown here so its time variance is one and swings are legible.
    b = score / np.sqrt(n_times)
    if not np.isclose(b @ b, 1.0, atol=1e-7):
        raise AssertionError("Market-mode time component is not normalized")
    rms20 = pd.Series(score).pow(2).rolling(20).mean().pow(.5).to_numpy()
    dates = panel.index[1:]
    x_days = dates.to_numpy(dtype="datetime64[D]").astype(int)
    start, finish = int(x_days.min()), int(x_days.max())

    width, height = 2000, 1190
    left, right = 165, 65
    top_start, top_end = 190, 690
    lower_start, lower_end = 780, 1070
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 35), "Tau = 1: leading KL time component",
              font=font(46), fill="#1e2d3b", anchor="mt")
    draw.text((width // 2, 98),
              "Positive = broad same-day gains  |  Negative = broad same-day declines",
              font=font(26), fill="#536473", anchor="mt")
    draw.text((left, 160), "Daily mode score (unit time variance)",
              font=font(27), fill="#273c50", anchor="lm")
    draw.text((left, 745), "20-day rolling RMS (size of recent swings)",
              font=font(27), fill="#273c50", anchor="lm")

    x_pos = lambda value: left + (value - start) / (finish - start) * (width - left - right)
    limit = float(np.ceil(np.max(np.abs(score))))
    y_top = lambda value: top_end - (value + limit) / (2 * limit) * (top_end - top_start)
    lower_limit = float(np.ceil(np.nanmax(rms20) * 2) / 2)
    y_lower = lambda value: lower_end - value / lower_limit * (lower_end - lower_start)
    for start_y, end_y in ((top_start, top_end), (lower_start, lower_end)):
        draw.rectangle((left, start_y, width - right, end_y), outline="#526170", width=2)
    for year in range(dates.min().year + 1, dates.max().year + 1):
        day = int(np.datetime64(f"{year}-01-01", "D").astype(int))
        if start <= day <= finish:
            px = x_pos(day)
            for start_y, end_y in ((top_start, top_end), (lower_start, lower_end)):
                draw.line((px, start_y, px, end_y), fill="#e3e9ed", width=2)
            draw.text((px, lower_end + 25), str(year), font=font(23),
                      fill="#435466", anchor="mt")
    for value in np.linspace(-limit, limit, 5):
        py = y_top(value)
        draw.line((left, py, width - right, py),
                  fill="#aab7c0" if abs(value) < 1e-10 else "#e3e9ed", width=2)
        draw.text((left - 16, py), f"{value:.0f}", font=font(23),
                  fill="#435466", anchor="rm")
    for value in np.linspace(0, lower_limit, 4):
        py = y_lower(value)
        draw.line((left, py, width - right, py), fill="#e3e9ed", width=2)
        draw.text((left - 16, py), f"{value:.1f}", font=font(23),
                  fill="#435466", anchor="rm")

    upper_points = [(int(x_pos(x)), int(y_top(y))) for x, y in zip(x_days, score)]
    draw.line(upper_points, fill="#245894", width=3, joint="curve")
    for index in np.flatnonzero(np.abs(score) >= 4):
        px, py = upper_points[index]
        color = "#287d66" if score[index] > 0 else "#ca594c"
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
    finite = np.isfinite(rms20)
    lower_points = [(int(x_pos(x)), int(y_lower(y))) for x, y in zip(x_days[finite], rms20[finite])]
    fill_points = lower_points + [(lower_points[-1][0], lower_end), (lower_points[0][0], lower_end)]
    draw.polygon(fill_points, fill="#f9e8db")
    draw.line(lower_points, fill="#db713c", width=4, joint="curve")
    draw.text((width // 2, height - 42),
              "Mode sign aligned with the equal-weight standardized return; PCA sign is otherwise arbitrary",
              font=font(22), fill="#536473", anchor="mm")
    canvas.save(path, format="PNG", optimize=True)

    minimum, maximum = int(np.argmin(score)), int(np.argmax(score))
    rms_peak = int(np.nanargmax(rms20))
    return {"variance_share": float(eigenvalues[-1] / len(panel.columns)),
            "loading_positive_fraction": float(np.mean(loading > 0)),
            "correlation_with_equal_weight_standardized_return": float(np.corrcoef(score, equal_weight_standardized)[0, 1]),
            "minimum_score": float(score[minimum]), "minimum_date": str(dates[minimum].date()),
            "maximum_score": float(score[maximum]), "maximum_date": str(dates[maximum].date()),
            "peak_rolling_20d_rms": float(rms20[rms_peak]),
            "peak_rolling_20d_rms_date": str(dates[rms_peak].date()),
            "display_normalization": "sqrt(T) times the unit-L2 KL time component"}


def make_tau1_top10_time_modes(panel: pd.DataFrame, figure: Path,
                               scores_csv: Path, summary_csv: Path) -> list[dict]:
    """Plot and export the ten leading tau=1 KL time coordinates on one scale."""
    daily = np.diff(np.log(panel.to_numpy(dtype=float)), axis=0)
    z = (daily - daily.mean(axis=0)) / daily.std(axis=0, ddof=0)
    eigenvalues, eigenvectors = np.linalg.eigh(z.T @ z / len(z))
    dates = panel.index[1:]
    days = dates.to_numpy(dtype="datetime64[D]").astype(int)
    first_day, last_day = int(days[0]), int(days[-1])
    n_stocks = z.shape[1]
    scores = []
    summary = []
    equal_weight = z.mean(axis=1)
    for rank in range(1, 11):
        index = -rank
        loading = eigenvectors[:, index].copy()
        score = z @ loading / np.sqrt(eigenvalues[index])
        if rank == 1:
            flip = np.corrcoef(score, equal_weight)[0, 1] < 0
            orientation = "positive correlation with equal-weight standardized return"
        else:
            flip = loading[np.argmax(np.abs(loading))] < 0
            orientation = "largest-absolute stock loading positive"
        if flip:
            score = -score
            loading = -loading
        if not np.isclose(score.mean(), 0, atol=1e-10) or not np.isclose(score.var(), 1, atol=1e-10):
            raise AssertionError(f"Unexpected normalization for mode {rank}")
        scores.append(score)
        summary.append({"mode_rank_descending": rank,
                        "eigenvalue": float(eigenvalues[index]),
                        "variance_share": float(eigenvalues[index] / n_stocks),
                        "score_mean": float(score.mean()),
                        "score_time_variance": float(score.var()),
                        "sign_convention": orientation})
    score_table = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"),
                                **{f"mode_{rank:02d}": scores[rank - 1]
                                   for rank in range(1, 11)}})
    score_table.to_csv(scores_csv, index=False)
    pd.DataFrame(summary).to_csv(summary_csv, index=False)

    width, height = 2200, 2770
    left, right = 265, 65
    row_start, row_pitch, plot_offset, plot_height = 165, 245, 34, 182
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 27), "Tau = 1: time components of the 10 largest eigenmodes",
              font=font(42), fill="#1e2d3b", anchor="mt")
    draw.text((width // 2, 89),
              "Daily standardized log returns  |  common vertical scale  |  each mode has unit time variance",
              font=font(24), fill="#536473", anchor="mt")
    x_pos = lambda day: left + (day - first_day) / (last_day - first_day) * (width - left - right)
    limit = float(np.ceil(max(np.max(np.abs(score)) for score in scores)))
    years = [(year, int(np.datetime64(f"{year}-01-01", "D").astype(int)))
             for year in range(dates.min().year + 1, dates.max().year + 1)]
    for rank, score in enumerate(scores, start=1):
        top = row_start + (rank - 1) * row_pitch + plot_offset
        bottom = top + plot_height
        y_pos = lambda value: bottom - (value + limit) / (2 * limit) * plot_height
        item = summary[rank - 1]
        draw.text((left, top - 31),
                  f"Mode {rank:02d}   eigenvalue {item['eigenvalue']:.2f}   variance share {item['variance_share']:.1%}",
                  font=font(24), fill="#273c50", anchor="lt")
        draw.rectangle((left, top, width - right, bottom), outline="#607180", width=2)
        for value in (-limit, 0, limit):
            py = y_pos(value)
            draw.line((left, py, width - right, py),
                      fill="#a7b5c1" if value == 0 else "#e1e7eb", width=2)
            draw.text((left - 16, py), f"{value:+.0f}" if value else "0",
                      font=font(20), fill="#435466", anchor="rm")
        for year, day in years:
            if first_day <= day <= last_day:
                px = int(x_pos(day))
                draw.line((px, top, px, bottom), fill="#e1e7eb", width=2)
                if rank == 10:
                    draw.text((px, bottom + 23), str(year), font=font(21),
                              fill="#435466", anchor="mt")
        points = [(int(x_pos(day)), int(y_pos(value)))
                  for day, value in zip(days, score)]
        draw.line(points, fill="#245894" if rank == 1 else "#3e718e",
                  width=2, joint="curve")
    draw.text((width // 2, height - 42),
              "Vertical units: each mode's own full-sample standard deviation; signs of modes 2-10 are arbitrary",
              font=font(22), fill="#536473", anchor="mm")
    canvas.save(figure, format="PNG", optimize=True)
    return summary


def run(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text())
    root = config_path.parent
    output = root / cfg["output_dir"]
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    panel, manifest = load_panel(Path(cfg["input_dir"]))
    logp = np.log(panel.to_numpy(dtype=float))
    metrics, eigen_rows, details = [], [], {}
    for tau in range(cfg["tau_start"], cfg["tau_end"] + 1):
        rows, vals, detail = analyze(logp, tau, "observed", "full", tau in (1, 250))
        metrics.extend(rows)
        eigen_rows.extend({"tau_days": tau, "rank_ascending": k + 1,
                           "n_stocks": len(vals), "eigenvalue": float(v)} for k, v in enumerate(vals))
        if detail: details[tau] = detail
    rng = np.random.default_rng(cfg["control_seed"])
    for k in range(cfg["control_repetitions"]):
        shuffled = permute_daily_returns(logp, rng)
        for tau in cfg["control_taus"]:
            rows, _, _ = analyze(shuffled, tau, f"shuffle_{k + 1:02d}", "full")
            metrics.extend(rows)
    split = pd.Timestamp(cfg["split_date"])
    for name, mask in (("early", panel.index < split), ("late", panel.index >= split)):
        for tau in cfg["split_taus"]:
            rows, _, _ = analyze(logp[mask], tau, "observed", name)
            metrics.extend(rows)
    table = pd.DataFrame(metrics)
    eig = pd.DataFrame(eigen_rows)
    table.to_csv(output / "tables/mode_metrics.csv", index=False)
    eig.to_csv(output / "tables/eigenvalues.csv", index=False)
    manifest.to_csv(output / "tables/input_manifest.csv", index=False)
    make_figures(table, eig, details, output / "figures")
    covariance_stats = make_tau1_covariance_figure(
        panel, output / "figures/07_covariance_tau1.png",
        output / "tables/covariance_tau1_order.json")
    market_removed_stats = make_market_removed_covariance_figure(
        panel, output / "tables/covariance_tau1_order.json",
        output / "figures/08_market_removed_covariance_tau1.png")
    market_mode_stats = make_market_mode_tau1_time_figure(
        panel, output / "figures/09_market_mode_tau1_time.png")
    top10_time_modes = make_tau1_top10_time_modes(
        panel, output / "figures/10_tau1_top10_time_modes.png",
        output / "tables/tau1_top10_time_modes.csv",
        output / "tables/tau1_top10_eigenmodes.csv")
    full = table[(table.source == "observed") & (table["sample"] == "full")]

    def row(tau: int, mode: str = "largest", sample: str = "full") -> pd.Series:
        x = table[(table.source == "observed") & (table["sample"] == sample) &
                  (table.tau_days == tau) & (table["mode"] == mode)]
        return x.iloc[0]

    lines = ["# Chen–Ma (2018) 日频原始价格研究", "", "## 论文与本研究", "",
             f"原论文：{PAPER}。作者分析 345 只美股的 36 秒盘中价格，使用不同 τ 的重叠对数收益、股票间相关矩阵、K–L 时间模态及功率谱。本研究只使用原始日收盘价，覆盖 τ=1–250 个交易日的全部整数跨度，不做价格移动平均。日频结果不等于论文的盘中结果。",
             f"输入为 {panel.shape[1]} 只股票，{panel.shape[0]} 个完整共同交易日，日期从 {panel.index.min().date()} 至 {panel.index.max().date()}。各跨度按每只股票在时间维标准化收益；相关矩阵的特征值之和等于股票数。谱斜率 α 在论文拟合频率序号的近似相同比例区间内估计，并同时报告 R²。", "",
             "## τ=1 的协方差矩阵", "",
             f"`figures/07_covariance_tau1.png` 由 {covariance_stats['tau1_return_observations']} 个相邻交易日的原始对数收益计算样本协方差（ddof=1），没有对各股票收益标准化。左图保留输入股票顺序、对角线和全部取值；右图对行与列应用同一个股票排列，遮蔽对角线，并把正协方差色标截在非对角元素的第 99 百分位。右图超过阈值的格子颜色饱和，但矩阵数值没有被改写。色标单位为协方差乘 10⁴。样本最大方差为 {covariance_stats['tau1_covariance_max']:.6g}，非对角协方差第 99 百分位为 {covariance_stats['tau1_off_diagonal_p99']:.6g}。", "",
             f"排序方法：先把日收益相关矩阵的最大共同模态扣除，取残差相关的正部作为相似度；在归一化相似矩阵前 8 个特征向量上做固定种子聚类，再在组内做谱排序。这里仅用相关性**寻找顺序**，热力图仍展示原始协方差。8 组大小为 {covariance_stats['block_sizes']}；块内平均协方差 {covariance_stats['within_block_mean_covariance']:.6g}，块间平均协方差 {covariance_stats['between_block_mean_covariance']:.6g}。块间仍普遍为正，因此它是近似块状结构，不是真正的块对角矩阵。这些组没有行业标签，不能据此称为行业板块。具体股票顺序见 `tables/covariance_tau1_order.json`。", "",
             "## 去掉最大共同模态后的协方差", "",
             f"`figures/08_market_removed_covariance_tau1.png` 在相同股票顺序、相同色标下比较原始协方差和残差协方差，两侧都遮蔽对角线。对每只股票的 τ=1 收益先减均值、除以样本标准差；从标准化收益中减去相关矩阵最大特征向量的投影，再乘回各股票标准差并重新计算样本协方差。等价于从原协方差中扣去这一个共同模态对应的秩一矩阵。该模态解释标准化总方差的 {market_removed_stats['market_mode_standardized_variance_share']:.1%}。残差总方差是原来的 {market_removed_stats['residual_total_variance_share']:.1%}；非对角元素的平均绝对值从 {market_removed_stats['original_off_diagonal_mean_abs_covariance']:.6g} 降至 {market_removed_stats['residual_off_diagonal_mean_abs_covariance']:.6g}。图中色标按原协方差非对角元素绝对值第 99 百分位设定，因此两侧颜色可直接比较。这里的“市场”是数据估计的最大共同主成分，而非外部市场指数；去掉一个模态不等于去掉所有系统性因素。", "",
             "## 第一个（最大）K–L 模态的时间变化", "",
             f"`figures/09_market_mode_tau1_time.png` 用真实日期画出 τ=1 日最大特征值模态的时间分量。为便于阅读，图中显示的是单位 L2 范数的 K–L 时间分量乘以 √T，因此整段时间方差为 1；符号选为与股票等权平均标准化收益同向。该模态解释标准化总方差的 {market_mode_stats['variance_share']:.1%}，其股票载荷为正的比例为 {market_mode_stats['loading_positive_fraction']:.1%}，与等权平均标准化收益的相关系数为 {market_mode_stats['correlation_with_equal_weight_standardized_return']:.3f}。最大负向冲击出现在 {market_mode_stats['minimum_date']}（{market_mode_stats['minimum_score']:.2f}），最大正向冲击出现在 {market_mode_stats['maximum_date']}（{market_mode_stats['maximum_score']:.2f}）；20 日滚动 RMS 在 {market_mode_stats['peak_rolling_20d_rms_date']} 达到 {market_mode_stats['peak_rolling_20d_rms']:.2f}。正负表示本样本股票同向涨跌，绝对值表示共同波动强弱；这些日期本身不证明具体事件成因，也不是未来收益预测。", "",
             "## τ=1 前 10 个特征模态的时间分量", "",
             "`figures/10_tau1_top10_time_modes.png` 按特征值从大到小展示前 10 个模态的逐日时间得分，日期与纵轴范围统一。每条曲线都等于单位 L2 范数的 K–L 时间分量乘以 √T，故全样本时间方差为 1，纵轴无量纲。第 1 模态的符号与股票等权平均标准化收益同向；第 2–10 模态将绝对值最大的股票载荷定为正，符号本身没有涨跌含义。逐日得分和特征值摘要分别见 `tables/tau1_top10_time_modes.csv` 与 `tables/tau1_top10_eigenmodes.csv`。", "",
             "## 代表性结果", "", "| τ(日) | 市场模态方差份额 | 市场模态 α | R² | 周期≥20日功率份额 | 第44小模态 α | 第44小模态 R² |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for tau in (1, 5, 20, 60, 120, 250):
        a, b = row(tau), row(tau, "44th_smallest")
        lines.append(f"| {tau} | {a.variance_share:.3f} | {a.alpha:.3f} | {a.fit_r2:.3f} | {a.low_frequency_share_period_ge_20d:.3f} | {b.alpha:.3f} | {b.fit_r2:.3f} |")
    lines += ["", "## 全跨度概览", ""]
    for mode in MODES:
        sub = full[full["mode"] == mode]
        lines.append(f"- {mode}：α 中位数 {sub.alpha.median():.3f}；拟合 R² 中位数 {sub.fit_r2.median():.3f}；方差份额中位数 {sub.variance_share.median():.3f}。")
    lines += ["", "## 日期打乱对照", "",
              "每次对所有股票的单日收益使用同一个随机日期排列，保留单日横截面共同波动和单日收益分布，再重新积累价格与计算各跨度的收益。这会破坏时间顺序，却保留重叠收益窗口的机械效应。以下为 10 次对照的描述性范围，不作显著性检验。", "",
              "| τ(日) | 观察到的市场 α | 打乱 α 中位数 [最小, 最大] | 观察到的低频份额 | 打乱低频份额中位数 [最小, 最大] |", "|---:|---:|---:|---:|---:|"]
    for tau in cfg["control_taus"]:
        a = row(tau)
        n = table[(table.source.str.startswith("shuffle")) & (table.tau_days == tau) & (table["mode"] == "largest")]
        alpha, low = n.alpha, n.low_frequency_share_period_ge_20d
        lines.append(f"| {tau} | {a.alpha:.3f} | {alpha.median():.3f} [{alpha.min():.3f}, {alpha.max():.3f}] | {a.low_frequency_share_period_ge_20d:.3f} | {low.median():.3f} [{low.min():.3f}, {low.max():.3f}] |")
    lines += ["", "## 分时段检查", "", f"按 {split.date()} 分开早期与晚期重新计算：", "",
              "| 时段 | τ(日) | 市场模态方差份额 | α | R² |", "|---|---:|---:|---:|---:|"]
    for sample in ("early", "late"):
        for tau in cfg["split_taus"]:
            a = row(tau, sample=sample)
            lines.append(f"| {sample} | {tau} | {a.variance_share:.3f} | {a.alpha:.3f} | {a.fit_r2:.3f} |")
    a1, a20, a250, b20 = row(1), row(20), row(250), row(20, "44th_smallest")
    n20 = table[(table.source.str.startswith("shuffle")) & (table.tau_days == 20) & (table["mode"] == "largest")]
    lines += ["", "## 结论与限制", "",
              f"1 日跨度市场模态的斜率为 {a1.alpha:.3f}，R² 为 {a1.fit_r2:.3f}，该频带没有可靠的幂律关系。20 日与 250 日跨度的斜率分别为 {a20.alpha:.3f} 和 {a250.alpha:.3f}，不能把原论文约 −1 的谱斜率直接迁移到这批日频数据。",
              f"20 日跨度观察到的市场模态斜率 {a20.alpha:.3f} 与日期打乱后的中位数 {n20.alpha.median():.3f} 接近；低频份额分别为 {a20.low_frequency_share_period_ge_20d:.3f} 和 {n20.low_frequency_share_period_ge_20d.median():.3f}。因此，大量低频结构可能由重叠窗口产生。第 44 小模态在 20 日跨度的 R² 仅 {b20.fit_r2:.4f}，不宜解释其斜率。",
              "谱斜率与低频份额不能单独证明确定性动力学、长记忆或收益可预测性。跨跨度的同一特征值排名也不保证是同一个经济模态。价格复权、历史成分规则及停牌填补方式未由输入文件说明；当前 288 只股票不代表动态调整的历史沪深300成分。", "",
              "## 产物", "", "`tables/mode_metrics.csv` 含每个跨度及模态的指标，`tables/eigenvalues.csv` 含全部特征值，`tables/input_manifest.csv` 含输入哈希。`figures/` 的所有图均为 PNG。", ""]
    from mode_structure_extension import run_extension
    mode_structure = run_extension(panel, output, cfg)
    lines += ["", "## τ=1 股票载荷与滚动结构延伸", "",
              f"`mode_structure_report.md` 详细报告前十模态的股票载荷、有效参与数、早晚时段稳定性和滚动窗口结构。第 1 模态载荷覆盖的有效股票数约 {mode_structure['market_full_sample_effective_stock_participation']:.0f}；早晚载荷重合度 {mode_structure['market_early_late_overlap']:.3f}，前十维子空间重合度 {mode_structure['top10_early_late_subspace_overlap']:.3f}。图 `11`–`14` 均为 PNG，逐股和逐窗口数值见相应 CSV。", ""]
    from residual_and_risk_extension import run_extension as run_residual_and_risk
    residual_and_risk = run_residual_and_risk(panel, output, cfg, spectral_block_order)
    lines += ["", "## 残差群组与样本外风险估计", "",
              f"`residual_and_risk_report.md` 检查扣除最大共同模态后的群组是否跨时段保留，并比较多种协方差估计在未来 20 日上的风险预测。早晚独立聚类的 ARI 为 {residual_and_risk['early_late_group_ari']:.3f}；早期分组在晚期的组内减组间残差相关为 {residual_and_risk['early_groups_on_late_contrast']:.3f}。图 `15`–`18` 均为 PNG，留出检验、滚动分组和风险评分的完整数值见相应 CSV。", ""]
    (output / "report.md").write_text("\n".join(lines))
    metadata = {"completed_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
                "n_stocks": panel.shape[1], "n_dates": panel.shape[0],
                "start": str(panel.index.min().date()), "end": str(panel.index.max().date()),
                "numpy": np.__version__, "pandas": pd.__version__,
                "covariance_tau1": covariance_stats,
                "market_removed_covariance_tau1": market_removed_stats,
                "market_mode_tau1_time": market_mode_stats,
                "tau1_top10_time_modes": top10_time_modes,
                "mode_structure_extension": mode_structure,
                "residual_and_risk_extension": residual_and_risk}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Wrote {output / 'report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    run(parser.parse_args().config.resolve())
