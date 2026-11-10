"""Stock loadings and rolling-mode structure for the daily Chen–Ma extension."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BLUE = "#245894"
ORANGE = "#db713c"
INK = "#243849"
MUTED = "#536473"


def font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in ["/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def principal_modes(returns: np.ndarray, count: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descending eigenpairs of the cross-stock correlation matrix."""
    deviations = returns.std(axis=0, ddof=0)
    if (deviations <= 1e-12).any():
        raise ValueError("At least one stock has constant returns")
    z = (returns - returns.mean(axis=0)) / deviations
    correlation = z.T @ z / len(z)
    values, vectors = np.linalg.eigh((correlation + correlation.T) / 2)
    values = values[::-1]
    vectors = vectors[:, ::-1]
    for rank in range(min(count, vectors.shape[1])):
        if rank == 0:
            score = z @ vectors[:, rank]
            flip = np.corrcoef(score, z.mean(axis=1))[0, 1] < 0
        else:
            flip = vectors[np.argmax(np.abs(vectors[:, rank])), rank] < 0
        if flip:
            vectors[:, rank] *= -1
    return values, vectors[:, :count], correlation


def effective_participation(vectors: np.ndarray) -> np.ndarray:
    return 1 / np.sum(vectors ** 4, axis=0)


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Mean squared cosine of principal angles; invariant to sign/rotation."""
    if left.shape != right.shape:
        raise ValueError("Subspaces need the same dimensions")
    return float(np.sum((left.T @ right) ** 2) / left.shape[1])


def render_loadings(path: Path, vectors: np.ndarray, eigenvalues: np.ndarray,
                    order: np.ndarray, boundaries: list[int]) -> None:
    n, count = vectors.shape
    width, height = 2200, 2130
    left, top, cell_width, row_pitch, row_height = 340, 205, 6, 171, 132
    cell_width = max(1, min(cell_width, (width-left-80)//n))
    width = max(width, left+n*cell_width+80)
    clip = float(np.quantile(np.abs(vectors), .99))
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 35), "Tau = 1: stock loadings of the 10 leading modes",
              font=font(43), fill=INK, anchor="mt")
    draw.text((width // 2, 95),
              "Stocks follow the residual-correlation block order  |  common loading color scale",
              font=font(24), fill=MUTED, anchor="mt")
    for rank in range(count):
        y = top + rank * row_pitch
        draw.text((left - 22, y + row_height // 2),
                  f"Mode {rank + 1:02d}  N_eff {effective_participation(vectors[:, rank:rank + 1])[0]:.0f}",
                  font=font(24), fill=INK, anchor="rm")
        values = vectors[order, rank]
        fraction = np.clip(np.abs(values) / clip, 0, 1)
        white = np.array([255., 255., 255.])
        positive = np.array([202., 74., 58.])
        negative = np.array([49., 105., 162.])
        target = np.where((values >= 0)[:, None], positive, negative)
        rgb = (white + fraction[:, None] * (target - white)).astype(np.uint8)
        for index, color in enumerate(rgb):
            x = left + index * cell_width
            draw.rectangle((x, y, x + cell_width - 1, y + row_height),
                           fill=tuple(int(c) for c in color))
        for boundary in boundaries:
            x = left + boundary * cell_width
            draw.line((x, y, x, y + row_height), fill="#303b45", width=2)
        draw.rectangle((left, y, left + n * cell_width, y + row_height),
                       outline="#62717f", width=2)
    bottom = top + (count - 1) * row_pitch + row_height
    ends = [0] + boundaries + [n]
    for block in range(len(ends) - 1):
        center = left + (ends[block] + ends[block + 1]) * cell_width / 2
        draw.text((center, bottom + 24), f"Group {block + 1}",
                  font=font(20), fill=MUTED, anchor="mt")
    bar_left = left + 360
    bar_top = height - 105
    bar_width = 1000
    for x in range(bar_width):
        value = 2 * x / (bar_width - 1) - 1
        frac = abs(value)
        target = np.array([202., 74., 58.]) if value >= 0 else np.array([49., 105., 162.])
        color = tuple(int(c) for c in (np.array([255., 255., 255.]) + frac * (target - 255)))
        draw.line((bar_left + x, bar_top, bar_left + x, bar_top + 25), fill=color)
    for x, label in ((bar_left, f"-{clip:.3f}"),
                     (bar_left + bar_width // 2, "0"),
                     (bar_left + bar_width, f"+{clip:.3f}")):
        draw.text((x, bar_top + 31), label, font=font(20), fill=MUTED, anchor="mt")
    draw.text((width // 2, height - 28),
              "Color saturates beyond the 99th percentile; ticker-level values are in the CSV",
              font=font(20), fill=MUTED, anchor="mm")
    canvas.save(path, format="PNG", optimize=True)


def render_overlap(path: Path, overlap: np.ndarray, mode1: float,
                   subspace: float, bootstrap: pd.DataFrame) -> None:
    width, height = 1730, 1160
    left, top, cell = 270, 210, 76
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 36), "Tau = 1: early vs late eigenvector overlap",
              font=font(43), fill=INK, anchor="mt")
    draw.text((width // 2, 100),
              "Absolute loading dot product  |  rows: earlier sample  |  columns: later sample",
              font=font(24), fill=MUTED, anchor="mt")
    for i in range(10):
        draw.text((left - 22, top + i * cell + cell // 2), f"{i + 1:02d}",
                  font=font(23), fill=INK, anchor="rm")
        draw.text((left + i * cell + cell // 2, top - 22), f"{i + 1:02d}",
                  font=font(23), fill=INK, anchor="mm")
        for j in range(10):
            value = float(overlap[i, j])
            rgb = tuple(int(c) for c in np.array([248, 251, 253]) * (1 - value)
                        + np.array([41, 100, 153]) * value)
            x, y = left + j * cell, top + i * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=rgb,
                           outline="white", width=2)
            draw.text((x + cell // 2, y + cell // 2), f"{value:.2f}",
                      font=font(20), fill="white" if value > .55 else INK, anchor="mm")
    info_x = left + 10 * cell + 75
    info = ["Early/late comparison", f"Market mode: {mode1:.3f}",
            f"Top-10 subspace: {subspace:.3f}", "",
            "20-day block bootstrap", "Full-sample loading uncertainty",
            f"Market overlap 5–95%: {bootstrap.market_overlap.quantile(.05):.3f}–{bootstrap.market_overlap.quantile(.95):.3f}",
            f"Top-10 subspace 5–95%: {bootstrap.top10_subspace_overlap.quantile(.05):.3f}–{bootstrap.top10_subspace_overlap.quantile(.95):.3f}"]
    for line, content in enumerate(info):
        draw.text((info_x, top + 18 + line * 63), content,
                  font=font(23 if line not in (0, 4) else 25),
                  fill=INK if line not in (0, 4) else BLUE, anchor="lt")
    draw.text((width // 2, height - 66),
              "Rank may change across periods; off-diagonal bright cells reveal mode mixing",
              font=font(22), fill=MUTED, anchor="mm")
    canvas.save(path, format="PNG", optimize=True)


def render_rolling(path: Path, title: str, panels: list[dict], rolling: pd.DataFrame,
                   note: str) -> None:
    width = 2000
    height = 335 + 300 * len(panels)
    left, right, top = 200, 85, 170
    panel_pitch, plot_height = 300, 215
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 30), title, font=font(43), fill=INK, anchor="mt")
    draw.text((width // 2, 96),
              "Blue: 500-day estimation window   |   Orange: 750-day estimation window",
              font=font(24), fill=MUTED, anchor="mt")
    years = range(int(rolling.end_date.str[:4].min()) + 1,
                  int(rolling.end_date.str[:4].max()) + 1)
    x_all = pd.to_datetime(rolling.end_date).to_numpy(dtype="datetime64[D]").astype(int)
    xmin, xmax = int(x_all.min()), int(x_all.max())
    x_pos = lambda value: left + (value - xmin) / (xmax - xmin) * (width - left - right)
    for panel_index, spec in enumerate(panels):
        y_top = top + panel_index * panel_pitch + 33
        y_bottom = y_top + plot_height
        lo, hi = spec["ylim"]
        y_pos = lambda value: y_bottom - (value - lo) / (hi - lo) * plot_height
        draw.text((left, y_top - 29), spec["title"], font=font(25), fill=INK, anchor="lt")
        draw.rectangle((left, y_top, width - right, y_bottom),
                       outline="#607180", width=2)
        for value in np.linspace(lo, hi, 4):
            py = y_pos(value)
            draw.line((left, py, width - right, py), fill="#e4eaee", width=2)
            draw.text((left - 15, py), f"{value:.2f}" if hi <= 1.5 else f"{value:.0f}",
                      font=font(20), fill=MUTED, anchor="rm")
        for year in years:
            day = int(np.datetime64(f"{year}-01-01", "D").astype(int))
            if xmin <= day <= xmax:
                px = x_pos(day)
                draw.line((px, y_top, px, y_bottom), fill="#e4eaee", width=2)
                if panel_index == len(panels) - 1:
                    draw.text((px, y_bottom + 20), str(year),
                              font=font(21), fill=MUTED, anchor="mt")
        for window, color in ((500, BLUE), (750, ORANGE)):
            sample = rolling[rolling.window_days == window]
            values = sample[spec["column"]].to_numpy(dtype=float)
            days = pd.to_datetime(sample.end_date).to_numpy(dtype="datetime64[D]").astype(int)
            valid = np.isfinite(values)
            points = [(int(x_pos(day)), int(y_pos(value)))
                      for day, value in zip(days[valid], values[valid])]
            if len(points) > 1:
                draw.line(points, fill=color, width=4, joint="curve")
    draw.text((width // 2, height - 55), note,
              font=font(21), fill=MUTED, anchor="mm")
    canvas.save(path, format="PNG", optimize=True)


def run_extension(panel: pd.DataFrame, output: Path, cfg: dict) -> dict:
    """Write reproducible loading and rolling analyses, returning concise metadata."""
    settings = cfg["mode_structure"]
    windows = [int(value) for value in settings["window_days"]]
    if windows != [500, 750]:
        raise ValueError("Current figures are configured for 500 and 750 day windows")
    step = int(settings["step_days"])
    repetitions = int(settings["bootstrap_repetitions"])
    block_length = int(settings["bootstrap_block_days"])
    rng = np.random.default_rng(int(settings["seed"]))
    dates = panel.index[1:]
    daily = np.diff(np.log(panel.to_numpy(dtype=float)), axis=0)
    n_times, n_stocks = daily.shape
    values, vectors, _ = principal_modes(daily)
    if not np.isclose(values.sum(), n_stocks, atol=1e-7):
        raise AssertionError("Correlation eigenvalue trace mismatch")

    order_data = json.loads((output / "tables/covariance_tau1_order.json").read_text())
    order = np.array([entry["old_index"] for entry in order_data["items"]], dtype=int)
    if not np.array_equal(np.sort(order), np.arange(n_stocks)):
        raise AssertionError("Stock order does not match the panel")
    labels = np.empty(n_stocks, dtype=int)
    for entry in order_data["items"]:
        if entry["ticker"] != panel.columns[entry["old_index"]]:
            raise AssertionError("Stored block order has different tickers")
        labels[entry["old_index"]] = entry["block"]
    boundaries = np.flatnonzero(np.diff(labels[order]) != 0).astype(int) + 1
    within = (labels[:, None] == labels[None, :]) & ~np.eye(n_stocks, dtype=bool)
    between = labels[:, None] != labels[None, :]

    loadings = pd.DataFrame({"ticker": panel.columns, "existing_residual_group": labels,
                             **{f"mode_{rank:02d}": vectors[:, rank - 1]
                                for rank in range(1, 11)}})
    loadings.to_csv(output / "tables/tau1_top10_loadings.csv", index=False)
    split = pd.Timestamp(cfg["split_date"])
    early_mask = np.asarray(dates < split)
    early_values, early_vectors, _ = principal_modes(daily[early_mask])
    late_values, late_vectors, _ = principal_modes(daily[~early_mask])
    overlap = np.abs(early_vectors.T @ late_vectors)
    pd.DataFrame(overlap, index=[f"early_{rank:02d}" for rank in range(1, 11)],
                 columns=[f"late_{rank:02d}" for rank in range(1, 11)]).to_csv(
                     output / "tables/tau1_early_late_overlap.csv")
    early_late_subspace = subspace_overlap(early_vectors, late_vectors)

    bootstrap_rows = []
    bootstrap_max = []
    for repeat in range(repetitions):
        starts = rng.integers(0, n_times - block_length + 1,
                              size=int(np.ceil(n_times / block_length)))
        indices = np.concatenate([np.arange(start, start + block_length)
                                  for start in starts])[:n_times]
        _, boot_vectors, _ = principal_modes(daily[indices])
        similarities = np.abs(vectors.T @ boot_vectors)
        bootstrap_rows.append({"replicate": repeat + 1,
                               "market_overlap": float(similarities[0, 0]),
                               "top10_subspace_overlap": subspace_overlap(vectors, boot_vectors)})
        bootstrap_max.append(similarities.max(axis=1))
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output / "tables/tau1_block_bootstrap_stability.csv", index=False)
    bootstrap_max = np.asarray(bootstrap_max)

    summary_rows = []
    for rank in range(10):
        vector = vectors[:, rank]
        positive = np.argsort(vector)[-5:][::-1]
        negative = np.argsort(vector)[:5]
        summary_rows.append({"mode_rank_descending": rank + 1,
                             "eigenvalue": float(values[rank]),
                             "variance_share": float(values[rank] / n_stocks),
                             "effective_stock_participation": float(effective_participation(vectors[:, rank:rank + 1])[0]),
                             "early_eigenvalue": float(early_values[rank]),
                             "late_eigenvalue": float(late_values[rank]),
                             "early_late_same_rank_abs_overlap": float(overlap[rank, rank]),
                             "early_late_best_late_rank": int(np.argmax(overlap[rank]) + 1),
                             "early_late_best_abs_overlap": float(np.max(overlap[rank])),
                             "bootstrap_best_match_overlap_median": float(np.median(bootstrap_max[:, rank])),
                             "bootstrap_best_match_overlap_p05": float(np.quantile(bootstrap_max[:, rank], .05)),
                             "top5_positive_tickers": ";".join(panel.columns[positive]),
                             "top5_negative_tickers": ";".join(panel.columns[negative])})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "tables/tau1_top10_loading_summary.csv", index=False)
    render_loadings(output / "figures/11_tau1_top10_loadings.png", vectors, values,
                    order, boundaries.tolist())
    render_overlap(output / "figures/12_tau1_mode_stability.png", overlap,
                   float(overlap[0, 0]), early_late_subspace, bootstrap)

    rolling_rows = []
    for window in windows:
        prior = []
        for end in range(window, n_times + 1, step):
            start = end - window
            window_returns = daily[start:end]
            window_values, window_vectors, correlation = principal_modes(window_returns)
            residual = correlation - window_values[0] * np.outer(window_vectors[:, 0], window_vectors[:, 0])
            disjoint = next((item for item in reversed(prior) if item["end"] <= start), None)
            p = np.maximum(window_values, 0) / np.sum(np.maximum(window_values, 0))
            positive_p = p[p > 0]
            row = {"window_days": window, "start_date": str(dates[start].date()),
                   "end_date": str(dates[end - 1].date()), "start_index": start,
                   "end_index_exclusive": end,
                   "market_variance_share": float(window_values[0] / n_stocks),
                   "top10_variance_share": float(window_values[:10].sum() / n_stocks),
                   "market_effective_stock_participation": float(effective_participation(window_vectors[:, :1])[0]),
                   "effective_eigenvalue_rank": float(np.exp(-np.sum(positive_p * np.log(positive_p)))),
                   "market_raw_daily_volatility_percent": float(window_returns.mean(axis=1).std(ddof=0) * 100),
                   "residual_group_within_mean": float(residual[within].mean()),
                   "residual_group_between_mean": float(residual[between].mean()),
                   "residual_group_contrast": float(residual[within].mean() - residual[between].mean()),
                   "market_overlap_disjoint": float(abs(window_vectors[:, 0] @ disjoint["vectors"][:, 0])) if disjoint else np.nan,
                   "top10_subspace_overlap_disjoint": subspace_overlap(window_vectors, disjoint["vectors"]) if disjoint else np.nan,
                   "disjoint_reference_end_date": str(dates[disjoint["end"] - 1].date()) if disjoint else ""}
            rolling_rows.append(row)
            prior.append({"end": end, "vectors": window_vectors})
    rolling = pd.DataFrame(rolling_rows)
    rolling.to_csv(output / "tables/rolling_mode_structure.csv", index=False)
    render_rolling(output / "figures/13_rolling_market_structure.png",
                   "Rolling market structure from tau = 1 daily returns",
                   [{"title": "Market-mode share of standardized variance", "column": "market_variance_share", "ylim": (0, .6)},
                    {"title": "Number of effectively participating stocks in market mode", "column": "market_effective_stock_participation", "ylim": (0, n_stocks)},
                    {"title": "Variance share of 10 leading modes", "column": "top10_variance_share", "ylim": (0, .75)},
                    {"title": "Volatility of equal-weight raw daily log return (%)", "column": "market_raw_daily_volatility_percent", "ylim": (0, 3)}],
                   rolling, "Every 20 trading days; rolling estimates are descriptive, not forecasts")
    render_rolling(output / "figures/14_rolling_mode_stability.png",
                   "Rolling mode stability and residual groups",
                   [{"title": "Market-mode loading overlap with a nonoverlapping past window", "column": "market_overlap_disjoint", "ylim": (0, 1)},
                    {"title": "Top-10 subspace overlap with a nonoverlapping past window", "column": "top10_subspace_overlap_disjoint", "ylim": (0, 1)},
                    {"title": "Residual group contrast: within minus between correlation", "column": "residual_group_contrast", "ylim": (-.05, .35)}],
                   rolling, "Groups fixed from full sample; disjoint comparisons share no return dates")

    lines = ["# τ=1 模态载荷与滚动市场结构", "",
             "## 方法", "",
             f"输入为 {n_stocks} 只股票、{n_times} 个相邻交易日对数收益；每个估计窗口按股票重新标准化。前十模态按特征值降序排列。第一个股票载荷方向与等权标准化收益同向，其余模态让绝对值最大的载荷为正；符号本身不代表涨跌。有效参与股票数定义为 1/Σaᵢ⁴，范围 1 到 {n_stocks}。", "",
             f"早晚样本按 {split.date()} 分割，分别有 {int(early_mask.sum())} 与 {int((~early_mask).sum())} 个交易日。早晚载荷用绝对内积比较；十维子空间重合度为平均平方余弦，范围 0–1。另做 {repetitions} 次 {block_length} 日移动块 bootstrap，以估计全样本载荷对抽样的敏感性。", "",
             "滚动窗口为 500 与 750 个交易日，每 20 日更新。稳定性比较寻找最近一个与当前窗口**没有共享日收益**的过去窗口；所以时间曲线最左端没有稳定性数值。残差群组使用先前全样本拟合出的八组股票，比较每个滚动窗口去掉最大相关模态后组内与组间的平均残差相关。全样本分组用于滚动描述，存在样本内选择效应。", "",
             "## 前十模态", "",
             "| 模态 | 特征值 | 方差份额 | 有效参与股票数 | 早晚同排名载荷重合度 | 早晚最佳匹配排名 |", "|---:|---:|---:|---:|---:|---:|"]
    for item in summary.itertuples():
        lines.append(f"| {item.mode_rank_descending} | {item.eigenvalue:.2f} | {item.variance_share:.1%} | {item.effective_stock_participation:.0f} | {item.early_late_same_rank_abs_overlap:.3f} | {item.early_late_best_late_rank} |")
    lines += ["", f"第 1 模态的早晚载荷重合度为 {overlap[0,0]:.3f}；前十维子空间重合度为 {early_late_subspace:.3f}。后面几个单独模态可能改变排名或混合，不能直接用同一编号断定其经济身份。",
              f"移动块 bootstrap 下，市场模态与全样本载荷的重合度 5–95% 分位为 {bootstrap.market_overlap.quantile(.05):.3f}–{bootstrap.market_overlap.quantile(.95):.3f}；前十维子空间为 {bootstrap.top10_subspace_overlap.quantile(.05):.3f}–{bootstrap.top10_subspace_overlap.quantile(.95):.3f}。这衡量抽样敏感性，不构成早晚差异的显著性检验。", "",
              "各模态的头部股票与正负载荷见 tau1_top10_loadings.csv；没有本市场行业标签时不能直接解释成行业因子。", "",
              "## 滚动窗口概览", "",
              "| 窗口（日） | 窗口数 | 市场方差份额范围 | 有效参与股票数范围 | 十模态份额范围 | 不重叠窗口市场载荷重合度中位数 |", "|---:|---:|---:|---:|---:|---:|"]
    for window in windows:
        x = rolling[rolling.window_days == window]
        lines.append(f"| {window} | {len(x)} | {x.market_variance_share.min():.1%}–{x.market_variance_share.max():.1%} | {x.market_effective_stock_participation.min():.0f}–{x.market_effective_stock_participation.max():.0f} | {x.top10_variance_share.min():.1%}–{x.top10_variance_share.max():.1%} | {x.market_overlap_disjoint.median():.3f} |")
    x500 = rolling[rolling.window_days == 500]
    lines += ["", f"500 日窗口的市场模态份额在 {x500.loc[x500.market_variance_share.idxmin(), 'end_date']} 为最低值 {x500.market_variance_share.min():.1%}，在 {x500.loc[x500.market_variance_share.idxmax(), 'end_date']} 为最高值 {x500.market_variance_share.max():.1%}。固定八组的残差相关组内减组间差值在 500 日窗口为 {x500.residual_group_contrast.min():.3f}–{x500.residual_group_contrast.max():.3f}，各窗口均为正；这只说明全样本所选分组在这些窗口中仍有组内优势。", "",
              "## 解释边界", "",
              "相关矩阵和 PCA 描述共同波动，不给出因果关系或未来收益预测。相邻滚动窗口仍共享大量交易日；图中单独的稳定性指标使用不重叠窗口比较。全样本固定分组可能高估群组稳定性，后续若获得行业标签和动态成分信息，应独立验证。原始收盘价的复权和成分历史规则未由输入数据说明。", "",
              "## 文件", "",
              "图：`figures/11_tau1_top10_loadings.png`、`figures/12_tau1_mode_stability.png`、`figures/13_rolling_market_structure.png`、`figures/14_rolling_mode_stability.png`。表：`tables/tau1_top10_loadings.csv`、`tables/tau1_top10_loading_summary.csv`、`tables/tau1_early_late_overlap.csv`、`tables/tau1_block_bootstrap_stability.csv`、`tables/rolling_mode_structure.csv`。", ""]
    (output / "mode_structure_report.md").write_text("\n".join(lines))
    metadata = {"window_days": windows, "step_days": step,
                "bootstrap_repetitions": repetitions, "bootstrap_block_days": block_length,
                "seed": int(settings["seed"]), "n_stocks": n_stocks,
                "n_daily_returns": n_times, "market_early_late_overlap": float(overlap[0, 0]),
                "top10_early_late_subspace_overlap": early_late_subspace,
                "market_full_sample_effective_stock_participation": float(summary.effective_stock_participation.iloc[0]),
                "rolling_window_counts": {str(window): int((rolling.window_days == window).sum()) for window in windows}}
    (output / "mode_structure_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata
