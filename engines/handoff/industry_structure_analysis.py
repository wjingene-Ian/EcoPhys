#!/usr/bin/env python3
"""
将 RMT 组模式相关矩阵与股票的一级/细分行业标签对齐。

主要原则：
1. 聚类仅使用收益率和 RMT 矩阵，不使用行业标签。
2. 行业标签只在聚类后用于注释、纯度和置换检验。
3. 东方财富 f127 作为细分行业；一级行业使用显式交叉表归并。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import hypergeom

import rmt_analysis as ra


SEED = 20261008

# Eastmoney f127 labels are fine-grained current industry labels.  This explicit
# crosswalk groups them into SW2021-like level-1 sectors for readable annotation.
L2_TO_L1 = {
    "IT服务Ⅱ": "计算机",
    "一般零售": "商贸零售",
    "专业工程": "建筑装饰",
    "中药Ⅱ": "医药生物",
    "乘用车": "汽车",
    "互联网电商": "商贸零售",
    "保险Ⅱ": "非银金融",
    "元件": "电子",
    "光伏设备": "电力设备",
    "光学光电子": "电子",
    "其他电源设备Ⅱ": "电力设备",
    "养殖业": "农林牧渔",
    "军工电子Ⅱ": "国防军工",
    "农化制品": "基础化工",
    "冶钢原料": "钢铁",
    "化学制品": "基础化工",
    "化学制药": "医药生物",
    "化学原料": "基础化工",
    "医疗器械": "医药生物",
    "医疗服务": "医药生物",
    "医药商业": "医药生物",
    "半导体": "电子",
    "厨卫电器": "家用电器",
    "商用车": "汽车",
    "基础建设": "建筑装饰",
    "多元金融": "非银金融",
    "家居用品": "轻工制造",
    "家电零部件Ⅱ": "家用电器",
    "小家电": "家用电器",
    "小金属": "有色金属",
    "工业金属": "有色金属",
    "工程机械": "机械设备",
    "广告营销": "传媒",
    "影视院线": "传媒",
    "房地产开发": "房地产",
    "房屋建设Ⅱ": "建筑装饰",
    "旅游及景区": "社会服务",
    "旅游零售Ⅱ": "商贸零售",
    "普钢": "钢铁",
    "服装家纺": "纺织服饰",
    "水泥": "建筑材料",
    "汽车零部件": "汽车",
    "油服工程": "石油石化",
    "消费电子": "电子",
    "游戏Ⅱ": "传媒",
    "炼化及贸易": "石油石化",
    "煤炭开采": "煤炭",
    "物流": "交通运输",
    "环境治理": "环保",
    "玻璃玻纤": "建筑材料",
    "生物制品": "医药生物",
    "电力": "公用事业",
    "电网设备": "电力设备",
    "电视广播Ⅱ": "传媒",
    "白色家电": "家用电器",
    "白酒Ⅱ": "食品饮料",
    "能源金属": "有色金属",
    "自动化设备": "机械设备",
    "航天装备Ⅱ": "国防军工",
    "航空机场": "交通运输",
    "航空装备Ⅱ": "国防军工",
    "航运港口": "交通运输",
    "装修建材": "建筑材料",
    "装修装饰Ⅱ": "建筑装饰",
    "计算机设备": "计算机",
    "证券Ⅱ": "非银金融",
    "调味发酵品Ⅱ": "食品饮料",
    "贵金属": "有色金属",
    "轨交设备Ⅱ": "机械设备",
    "软件开发": "计算机",
    "通信服务": "通信",
    "通信设备": "通信",
    "铁路公路": "交通运输",
    "银行Ⅱ": "银行",
    "风电设备": "电力设备",
    "食品加工": "食品饮料",
    "饮料乳品": "食品饮料",
    "饲料": "农林牧渔",
}

FALLBACKS = {
    "002008.SZ": {
        "code": "002008",
        "stock_name": "大族激光",
        "eastmoney_industry": "自动化设备",
        "classification_provider": "Tonghuashun F10 fallback",
        "source_url": "https://basic.10jqka.com.cn/002008/company.html",
        "fetch_error": "Eastmoney request unavailable; filled from cited fallback",
    }
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returns", type=Path, default=here / "tables" / "07_log_return_matrix.csv")
    parser.add_argument("--metadata", type=Path, default=here / "rmt_tables" / "13_stock_industry_raw.csv")
    parser.add_argument("--output", type=Path, default=here)
    parser.add_argument("--permutations", type=int, default=2000)
    return parser.parse_args()


def clean_name(value: str) -> str:
    return "".join(str(value).split())


def prepare_metadata(path: Path, tickers: list[str]) -> pd.DataFrame:
    metadata = pd.read_csv(path, dtype=str).fillna("").set_index("ticker")
    for ticker, replacement in FALLBACKS.items():
        for key, value in replacement.items():
            metadata.loc[ticker, key] = value
    metadata = metadata.reindex(tickers)
    if metadata.index.isna().any() or metadata["eastmoney_industry"].eq("").any():
        missing = metadata.index[metadata["eastmoney_industry"].eq("")].tolist()
        raise ValueError(f"Missing industry metadata: {missing}")
    unknown = sorted(set(metadata["eastmoney_industry"]) - set(L2_TO_L1))
    if unknown:
        raise ValueError(f"Unmapped Eastmoney industries: {unknown}")
    metadata["stock_name"] = metadata["stock_name"].map(clean_name)
    metadata["industry_l2"] = metadata["eastmoney_industry"]
    metadata["industry_l1"] = metadata["industry_l2"].map(L2_TO_L1)
    metadata["classification_level_1_note"] = "Explicit SW2021-like crosswalk from Eastmoney f127 label"
    metadata.index.name = "ticker"
    return metadata


def correlation_distance(correlation: np.ndarray) -> np.ndarray:
    distance = np.sqrt(np.clip((1.0 - correlation) / 2.0, 0.0, None))
    np.fill_diagonal(distance, 0.0)
    return distance


def silhouette_from_distance(distance: np.ndarray, labels: np.ndarray) -> float:
    result = []
    unique = np.unique(labels)
    for index, own in enumerate(labels):
        own_mask = labels == own
        if own_mask.sum() <= 1:
            result.append(0.0)
            continue
        a = distance[index, own_mask].sum() / (own_mask.sum() - 1)
        b = min(distance[index, labels == other].mean() for other in unique if other != own)
        result.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(result))


def relabel_by_leaf_order(labels: np.ndarray, order: np.ndarray) -> np.ndarray:
    first_position = {label: int(np.flatnonzero(labels[order] == label)[0]) for label in np.unique(labels)}
    mapping = {label: idx + 1 for idx, label in enumerate(sorted(first_position, key=first_position.get))}
    return np.array([mapping[label] for label in labels], dtype=int)


def contingency_scores(cluster_labels: np.ndarray, class_labels: np.ndarray) -> dict[str, float]:
    clusters, cluster_codes = np.unique(cluster_labels, return_inverse=True)
    classes, class_codes = np.unique(class_labels, return_inverse=True)
    table = np.zeros((len(clusters), len(classes)), dtype=int)
    np.add.at(table, (cluster_codes, class_codes), 1)
    n = table.sum()
    purity = table.max(axis=1).sum() / n
    row = table.sum(axis=1)
    col = table.sum(axis=0)

    def comb2(values: np.ndarray) -> float:
        values = values.astype(float)
        return float(np.sum(values * (values - 1.0) / 2.0))

    total_pairs = n * (n - 1.0) / 2.0
    observed = comb2(table.ravel())
    expected = comb2(row) * comb2(col) / total_pairs
    maximum = 0.5 * (comb2(row) + comb2(col))
    ari = (observed - expected) / (maximum - expected) if maximum > expected else 0.0

    probabilities = table / n
    row_p = row / n
    col_p = col / n
    mi = 0.0
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if probabilities[i, j] > 0:
                mi += probabilities[i, j] * math.log(probabilities[i, j] / (row_p[i] * col_p[j]))
    hx = -float(np.sum(row_p[row_p > 0] * np.log(row_p[row_p > 0])))
    hy = -float(np.sum(col_p[col_p > 0] * np.log(col_p[col_p > 0])))
    nmi = mi / math.sqrt(hx * hy) if hx > 0 and hy > 0 else 0.0
    return {"purity": float(purity), "adjusted_rand_index": float(ari), "normalized_mutual_information": float(nmi)}


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def cluster_composition(
    correlation: np.ndarray,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    order: np.ndarray,
) -> pd.DataFrame:
    rows = []
    n_total = len(labels)
    industry_totals = metadata["industry_l1"].value_counts()
    ordered_labels = labels[order]
    for cluster in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == cluster)
        l1 = metadata.iloc[indices]["industry_l1"]
        l2 = metadata.iloc[indices]["industry_l2"]
        l1_counts = l1.value_counts()
        l2_counts = l2.value_counts()
        dominant_l1 = l1_counts.index[0]
        dominant_count = int(l1_counts.iloc[0])
        off = correlation[np.ix_(indices, indices)][np.triu_indices(len(indices), 1)]
        p_value = float(
            hypergeom.sf(
                dominant_count - 1,
                n_total,
                int(industry_totals[dominant_l1]),
                len(indices),
            )
        )
        rows.append(
            {
                "cluster": int(cluster),
                "size": int(len(indices)),
                "leaf_start": int(np.flatnonzero(ordered_labels == cluster).min()),
                "leaf_end": int(np.flatnonzero(ordered_labels == cluster).max()),
                "dominant_industry_l1": dominant_l1,
                "dominant_l1_count": dominant_count,
                "dominant_l1_purity": dominant_count / len(indices),
                "dominant_industry_l2": l2_counts.index[0],
                "dominant_l2_count": int(l2_counts.iloc[0]),
                "dominant_l2_purity": float(l2_counts.iloc[0] / len(indices)),
                "top_l1_composition": "; ".join(f"{name}:{count}" for name, count in l1_counts.head(4).items()),
                "top_l2_composition": "; ".join(f"{name}:{count}" for name, count in l2_counts.head(5).items()),
                "mean_within_group_correlation": float(off.mean()) if len(off) else np.nan,
                "dominant_l1_hypergeom_p": p_value,
            }
        )
    frame = pd.DataFrame(rows)
    frame["dominant_l1_fdr_q"] = benjamini_hochberg(frame["dominant_l1_hypergeom_p"].to_numpy())
    return frame


def pair_relation_table(
    empirical: np.ndarray,
    group: np.ndarray,
    metadata: pd.DataFrame,
    permutations: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    left, right = np.triu_indices(len(metadata), 1)
    l1 = metadata["industry_l1"].to_numpy()
    l2 = metadata["industry_l2"].to_numpy()
    same_l1 = l1[left] == l1[right]
    same_l2 = l2[left] == l2[right]
    categories = {
        "same_l2": same_l2,
        "same_l1_different_l2": same_l1 & ~same_l2,
        "different_l1": ~same_l1,
    }
    rows = []
    for key, mask in categories.items():
        rows.append(
            {
                "pair_relation": key,
                "pair_count": int(mask.sum()),
                "empirical_mean_correlation": float(empirical[left[mask], right[mask]].mean()),
                "group_mode_mean_correlation": float(group[left[mask], right[mask]].mean()),
                "group_mode_median_correlation": float(np.median(group[left[mask], right[mask]])),
            }
        )

    rng = np.random.default_rng(SEED)
    values = group[left, right]
    observed_l1 = float(values[same_l1].mean() - values[~same_l1].mean())
    observed_l2 = float(values[same_l2].mean() - values[same_l1 & ~same_l2].mean())
    null_l1 = np.empty(permutations)
    null_l2 = np.empty(permutations)
    l1_groups = [np.flatnonzero(l1 == label) for label in np.unique(l1)]
    for repetition in range(permutations):
        permuted_l1 = rng.permutation(l1)
        null_same_l1 = permuted_l1[left] == permuted_l1[right]
        null_l1[repetition] = values[null_same_l1].mean() - values[~null_same_l1].mean()

        permuted_l2 = l2.copy()
        for indices in l1_groups:
            permuted_l2[indices] = rng.permutation(permuted_l2[indices])
        null_same_l2 = permuted_l2[left] == permuted_l2[right]
        valid_cross = same_l1 & ~null_same_l2
        null_l2[repetition] = values[null_same_l2].mean() - values[valid_cross].mean()

    tests = {
        "l1_within_minus_between": observed_l1,
        "l1_permutation_p_one_sided": float((1 + np.sum(null_l1 >= observed_l1)) / (permutations + 1)),
        "l2_same_minus_same_l1_different_l2": observed_l2,
        "l2_permutation_p_one_sided": float((1 + np.sum(null_l2 >= observed_l2)) / (permutations + 1)),
        "permutations": int(permutations),
    }
    return pd.DataFrame(rows), tests


def industry_summary(empirical: np.ndarray, group: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for industry, subset in metadata.groupby("industry_l1", sort=False):
        indices = metadata.index.get_indexer(subset.index)
        upper = np.triu_indices(len(indices), 1)
        empirical_values = empirical[np.ix_(indices, indices)][upper]
        group_values = group[np.ix_(indices, indices)][upper]
        l2_counts = subset["industry_l2"].value_counts()
        rows.append(
            {
                "industry_l1": industry,
                "stock_count": len(indices),
                "subindustry_count": int(len(l2_counts)),
                "subindustry_composition": "; ".join(f"{name}:{count}" for name, count in l2_counts.items()),
                "empirical_mean_within_correlation": float(empirical_values.mean()) if len(empirical_values) else np.nan,
                "group_mode_mean_within_correlation": float(group_values.mean()) if len(group_values) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["stock_count", "industry_l1"], ascending=[False, True])


def within_l1_subindustry_contrast(
    empirical: np.ndarray,
    group: np.ndarray,
    metadata: pd.DataFrame,
    permutations: int,
) -> pd.DataFrame:
    """Test whether fine-industry pairs are closer within each broad industry."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for industry, subset in metadata.groupby("industry_l1", sort=False):
        if subset["industry_l2"].nunique() < 2:
            continue
        indices = metadata.index.get_indexer(subset.index)
        left, right = np.triu_indices(len(indices), 1)
        labels = subset["industry_l2"].to_numpy()
        same = labels[left] == labels[right]
        if not same.any() or same.all():
            continue
        empirical_values = empirical[np.ix_(indices, indices)][left, right]
        group_values = group[np.ix_(indices, indices)][left, right]
        observed = float(group_values[same].mean() - group_values[~same].mean())
        null = np.empty(permutations)
        for repetition in range(permutations):
            permuted = rng.permutation(labels)
            permuted_same = permuted[left] == permuted[right]
            null[repetition] = float(
                group_values[permuted_same].mean() - group_values[~permuted_same].mean()
            )
        rows.append(
            {
                "industry_l1": industry,
                "stock_count": int(len(indices)),
                "subindustry_count": int(subset["industry_l2"].nunique()),
                "same_l2_pair_count": int(same.sum()),
                "different_l2_pair_count": int((~same).sum()),
                "empirical_same_l2_mean": float(empirical_values[same].mean()),
                "empirical_different_l2_mean": float(empirical_values[~same].mean()),
                "empirical_contrast": float(empirical_values[same].mean() - empirical_values[~same].mean()),
                "group_same_l2_mean": float(group_values[same].mean()),
                "group_different_l2_mean": float(group_values[~same].mean()),
                "group_contrast": observed,
                "permutation_p_one_sided": float((1 + np.sum(null >= observed)) / (permutations + 1)),
            }
        )
    frame = pd.DataFrame(rows)
    frame["fdr_q"] = benjamini_hochberg(frame["permutation_p_one_sided"].to_numpy())
    return frame.sort_values("group_contrast", ascending=False)


def categorical_palette(labels: pd.Series) -> tuple[dict[str, str], ListedColormap]:
    categories = sorted(labels.unique())
    base = list(mpl.colormaps["tab20"].colors) + list(mpl.colormaps["tab20b"].colors)
    colors = [mpl.colors.to_hex(base[index % len(base)]) for index in range(len(categories))]
    mapping = dict(zip(categories, colors))
    return mapping, ListedColormap(colors)


def contiguous_runs(values: np.ndarray) -> list[tuple[int, int, str]]:
    runs = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            runs.append((start, index - 1, str(values[start])))
            start = index
    return runs


def apply_style() -> None:
    # Nature-figure requirements: editable SVG text and a sans-serif CJK fallback.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Arial", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 8


def save_all(fig: plt.Figure, base: Path, png_dpi: int = 300) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


def plot_annotated_original_order(
    empirical: np.ndarray,
    group: np.ndarray,
    metadata: pd.DataFrame,
    output: Path,
) -> None:
    apply_style()
    distance = correlation_distance(empirical)
    tree = linkage(squareform(distance, checks=False), method="average", optimal_ordering=True)
    order = leaves_list(tree)
    ordered = metadata.iloc[order]
    l1_colors, _ = categorical_palette(metadata["industry_l1"])
    l2_categories = sorted(metadata["industry_l2"].unique())
    l2_code = {name: index for index, name in enumerate(l2_categories)}
    l2_cmap = ListedColormap([mpl.colors.to_hex(mpl.colormaps["turbo"](x)) for x in np.linspace(0, 1, len(l2_categories))])

    fig = plt.figure(figsize=(15.8, 13.8), constrained_layout=False)
    grid = fig.add_gridspec(
        5,
        4,
        width_ratios=[0.23, 0.23, 9.6, 2.8],
        height_ratios=[0.36, 0.22, 0.22, 9.6, 1.7],
        left=0.04,
        right=0.98,
        bottom=0.05,
        top=0.95,
        wspace=0.03,
        hspace=0.03,
    )
    ax_l1_top = fig.add_subplot(grid[1, 2])
    ax_l2_top = fig.add_subplot(grid[2, 2])
    ax_l1_left = fig.add_subplot(grid[3, 0])
    ax_l2_left = fig.add_subplot(grid[3, 1])
    ax = fig.add_subplot(grid[3, 2])
    ax_legend = fig.add_subplot(grid[3:, 3])
    ax_labels = fig.add_subplot(grid[4, 2])

    image = ax.imshow(group[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    l1_names = ordered["industry_l1"].to_numpy()
    l1_palette_order = sorted(metadata["industry_l1"].unique())
    l1_code = {name: index for index, name in enumerate(l1_palette_order)}
    l1_array = np.array([l1_code[name] for name in l1_names])
    l1_cmap = ListedColormap([l1_colors[name] for name in l1_palette_order])
    l2_array = np.array([l2_code[name] for name in ordered["industry_l2"]])
    ax_l1_top.imshow(l1_array[None, :], aspect="auto", cmap=l1_cmap, vmin=-0.5, vmax=len(l1_palette_order)-0.5)
    ax_l2_top.imshow(l2_array[None, :], aspect="auto", cmap=l2_cmap, vmin=-0.5, vmax=len(l2_categories)-0.5)
    ax_l1_left.imshow(l1_array[:, None], aspect="auto", cmap=l1_cmap, vmin=-0.5, vmax=len(l1_palette_order)-0.5)
    ax_l2_left.imshow(l2_array[:, None], aspect="auto", cmap=l2_cmap, vmin=-0.5, vmax=len(l2_categories)-0.5)
    for strip in [ax_l1_top, ax_l2_top, ax_l1_left, ax_l2_left]:
        strip.set_xticks([])
        strip.set_yticks([])
        for spine in strip.spines.values():
            spine.set_visible(False)
    ax_l1_top.set_ylabel("L1", rotation=0, ha="right", va="center", fontsize=7)
    ax_l2_top.set_ylabel("L2", rotation=0, ha="right", va="center", fontsize=7)

    # Label only repeated contiguous L2 runs; one-stock runs would be unreadable.
    label_runs = [run for run in contiguous_runs(ordered["industry_l2"].to_numpy()) if run[1] - run[0] + 1 >= 3]
    ax_labels.set_xlim(-0.5, len(order)-0.5)
    ax_labels.set_ylim(0, 1)
    ax_labels.axis("off")
    used = []
    for start, end, label in sorted(label_runs, key=lambda item: item[1] - item[0], reverse=True):
        center = (start + end) / 2
        if any(abs(center - previous) < 9 for previous in used):
            continue
        used.append(center)
        ax_labels.plot([center, center], [0.82, 1.0], color="#555555", lw=0.6)
        ax_labels.text(center, 0.78, label.replace("Ⅱ", ""), rotation=58, ha="right", va="top", fontsize=6.2)

    handles = [Patch(facecolor=l1_colors[name], edgecolor="none", label=name) for name in l1_palette_order]
    ax_legend.legend(handles=handles, title="一级行业", loc="upper left", ncol=2, fontsize=6.5, title_fontsize=7.5, frameon=False)
    ax_legend.axis("off")
    colorbar = fig.colorbar(image, ax=ax_legend, orientation="horizontal", fraction=0.045, pad=0.03, anchor=(0.0, 0.0))
    colorbar.set_label("Group-mode correlation", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    fig.text(0.04, 0.975, "a", fontsize=13, fontweight="bold", va="top")
    fig.text(
        0.42,
        0.968,
        "RMT group modes with current industry annotations",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.022,
        "Stock order is unchanged from the original figure: average-linkage clustering of the empirical correlation matrix.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    save_all(fig, output / "rmt_figures" / "12_group_modes_industry_annotated")
    plt.close(fig)


def plot_clustered_group_modes(
    group: np.ndarray,
    metadata: pd.DataFrame,
    tree: np.ndarray,
    order: np.ndarray,
    labels: np.ndarray,
    composition: pd.DataFrame,
    output: Path,
) -> None:
    apply_style()
    ordered = metadata.iloc[order]
    l1_colors, _ = categorical_palette(metadata["industry_l1"])
    l1_palette_order = sorted(metadata["industry_l1"].unique())
    l1_code = {name: index for index, name in enumerate(l1_palette_order)}
    l1_array = np.array([l1_code[name] for name in ordered["industry_l1"]])
    l1_cmap = ListedColormap([l1_colors[name] for name in l1_palette_order])

    fig = plt.figure(figsize=(15.5, 11.5))
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=[0.22, 8.0, 3.7],
        height_ratios=[1.25, 0.22, 8.0],
        left=0.05,
        right=0.98,
        bottom=0.07,
        top=0.95,
        hspace=0.02,
        wspace=0.04,
    )
    ax_tree = fig.add_subplot(grid[0, 1])
    ax_strip = fig.add_subplot(grid[1, 1])
    ax_left = fig.add_subplot(grid[2, 0])
    ax = fig.add_subplot(grid[2, 1])
    ax_info = fig.add_subplot(grid[:, 2])
    dendrogram(tree, ax=ax_tree, no_labels=True, color_threshold=0, above_threshold_color="#555555")
    ax_tree.set_xticks([])
    ax_tree.set_yticks([])
    for spine in ax_tree.spines.values():
        spine.set_visible(False)
    ax_strip.imshow(l1_array[None, :], aspect="auto", cmap=l1_cmap, vmin=-0.5, vmax=len(l1_palette_order)-0.5)
    ax_left.imshow(l1_array[:, None], aspect="auto", cmap=l1_cmap, vmin=-0.5, vmax=len(l1_palette_order)-0.5)
    for strip in [ax_strip, ax_left]:
        strip.set_xticks([])
        strip.set_yticks([])
        for spine in strip.spines.values():
            spine.set_visible(False)

    image = ax.imshow(group[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ordered_labels = labels[order]
    for cluster in sorted(np.unique(labels)):
        positions = np.flatnonzero(ordered_labels == cluster)
        start, end = positions.min(), positions.max()
        rect = Rectangle((start - 0.5, start - 0.5), end - start + 1, end - start + 1, fill=False, ec="#1f1f1f", lw=0.8)
        ax.add_patch(rect)
        ax.text((start + end) / 2, start + 1.1, f"C{cluster}", ha="center", va="top", fontsize=6.5, fontweight="bold")

    ax_info.axis("off")
    ax_info.text(0.0, 0.985, "Data-driven cluster composition", fontsize=10, fontweight="bold", va="top")
    y = 0.95
    for row in composition.itertuples(index=False):
        significant = row.dominant_l1_fdr_q < 0.05
        marker = "*" if significant else ""
        text = (
            f"C{row.cluster}  n={row.size:>2}  {row.dominant_industry_l1}{marker} "
            f"({row.dominant_l1_purity:.0%})\n"
            f"    L2: {row.dominant_industry_l2} ({row.dominant_l2_purity:.0%})"
        )
        ax_info.text(0.0, y, text, fontsize=7.1, va="top", linespacing=1.2)
        y -= 0.052
        if y < 0.08:
            break
    ax_info.text(0.0, 0.035, "* dominant L1 enrichment: BH-FDR q < 0.05", fontsize=6.5, color="#444444")
    cax = ax_info.inset_axes([0.0, 0.0, 0.65, 0.018])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_label("Group-mode correlation", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    fig.suptitle("RMT group-mode hierarchy and industry correspondence", fontsize=13, fontweight="bold")
    fig.text(0.035, 0.965, "b", fontsize=13, fontweight="bold")
    save_all(fig, output / "rmt_figures" / "13_group_mode_hierarchy_clusters")
    plt.close(fig)


def plot_alignment_statistics(
    pair_table: pd.DataFrame,
    composition: pd.DataFrame,
    tests: dict[str, float],
    scores: dict[str, float],
    output: Path,
) -> None:
    apply_style()
    labels = ["Same L2", "Same L1,\ndifferent L2", "Different L1"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.0, 1.35]})
    axes[0].bar(x - 0.18, pair_table["empirical_mean_correlation"], 0.36, color="#B8B8B8", label="Empirical")
    axes[0].bar(x + 0.18, pair_table["group_mode_mean_correlation"], 0.36, color="#2F5D8C", label="RMT group modes")
    axes[0].axhline(0, color="#333333", lw=0.7)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean pairwise correlation")
    axes[0].set_title("Industry hierarchy in pairwise correlations", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].text(
        0.02,
        0.98,
        f"L1 permutation p={tests['l1_permutation_p_one_sided']:.4f}\n"
        f"L2-within-L1 permutation p={tests['l2_permutation_p_one_sided']:.4f}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=7,
    )

    table = composition.sort_values("cluster")
    y = np.arange(len(table))
    axes[1].barh(y, table["dominant_l1_purity"], color="#D69E2E", alpha=0.9, label="L1 purity")
    axes[1].barh(y, table["dominant_l2_purity"], color="#2A7F62", alpha=0.85, height=0.45, label="L2 purity")
    axes[1].set_yticks(y, [f"C{value}" for value in table["cluster"]])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].set_xlabel("Dominant-label share")
    axes[1].set_title("Industry purity of unsupervised clusters", fontweight="bold")
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.73, -0.13),
        ncol=2,
    )
    axes[1].text(
        0.00,
        -0.16,
        f"Overall L1 purity={scores['purity']:.1%}; NMI={scores['normalized_mutual_information']:.3f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    for ax, panel in zip(axes, ["c", "d"]):
        ax.text(-0.10, 1.06, panel, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_all(fig, output / "rmt_figures" / "14_industry_alignment_statistics")
    plt.close(fig)


def plot_within_industry_subclusters(
    group: np.ndarray,
    metadata: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    apply_style()
    selected = summary.query("stock_count >= 8 and subindustry_count >= 2").head(8)["industry_l1"].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for ax, industry in zip(axes.flat, selected):
        indices = np.flatnonzero(metadata["industry_l1"].to_numpy() == industry)
        sub = group[np.ix_(indices, indices)]
        if len(indices) >= 3:
            distance = correlation_distance(sub)
            tree = linkage(squareform(distance, checks=False), method="average", optimal_ordering=True)
            local_order = leaves_list(tree)
        else:
            local_order = np.arange(len(indices))
        ordered_indices = indices[local_order]
        image = ax.imshow(group[np.ix_(ordered_indices, ordered_indices)], cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        labels = [
            f"{metadata.iloc[index]['stock_name']}  {metadata.iloc[index]['industry_l2'].replace('Ⅱ', '')}"
            for index in ordered_indices
        ]
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=5.2)
        ax.set_xticks([])
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_title(f"{industry}  n={len(indices)}", fontsize=9, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.suptitle("Within-industry structure: stock names and fine industry labels", fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.13, right=0.88, top=0.94, bottom=0.05, wspace=0.68, hspace=0.22)
    cax = fig.add_axes([0.91, 0.19, 0.012, 0.64])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Group-mode correlation", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    save_all(fig, output / "rmt_figures" / "15_within_industry_subclusters")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output
    tables = output / "rmt_tables"
    figures = output / "rmt_figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    returns = ra.load_returns(args.returns)
    metadata = prepare_metadata(args.metadata, list(returns.columns))
    result = ra.rmt_decomposition(returns.to_numpy())
    components = ra.correlation_components(result)
    group = ra.normalize_to_correlation(components["group"])
    distance = correlation_distance(group)
    tree = linkage(squareform(distance, checks=False), method="average", optimal_ordering=True)
    order = leaves_list(tree)

    selection_rows = []
    for k in range(4, 25):
        candidate = fcluster(tree, k, criterion="maxclust")
        selection_rows.append(
            {
                "requested_clusters": k,
                "actual_clusters": int(len(np.unique(candidate))),
                "silhouette": silhouette_from_distance(distance, candidate),
                "smallest_cluster": int(pd.Series(candidate).value_counts().min()),
                "largest_cluster": int(pd.Series(candidate).value_counts().max()),
            }
        )
    selection = pd.DataFrame(selection_rows)
    eligible = selection.query("actual_clusters >= 6 and actual_clusters <= 20")
    selected_k = int(eligible.loc[eligible["silhouette"].idxmax(), "actual_clusters"])
    labels = fcluster(tree, selected_k, criterion="maxclust")
    labels = relabel_by_leaf_order(labels, order)
    metadata["unsupervised_cluster"] = labels

    composition = cluster_composition(group, metadata, labels, order)
    pair_table, tests = pair_relation_table(result.correlation, group, metadata, args.permutations)
    ind_summary = industry_summary(result.correlation, group, metadata)
    subindustry_contrast = within_l1_subindustry_contrast(
        result.correlation,
        group,
        metadata,
        args.permutations,
    )
    scores = contingency_scores(labels, metadata["industry_l1"].to_numpy())
    scores["selected_cluster_count"] = selected_k
    scores["selected_silhouette"] = float(selection.loc[selection["actual_clusters"] == selected_k, "silhouette"].iloc[0])
    scores.update(tests)

    metadata.reset_index().to_csv(tables / "14_stock_industry_mapping.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(tables / "15_cluster_count_selection.csv", index=False)
    composition.to_csv(tables / "16_group_cluster_industry_composition.csv", index=False, encoding="utf-8-sig")
    pair_table.to_csv(tables / "17_industry_pair_correlations.csv", index=False)
    ind_summary.to_csv(tables / "18_industry_summary.csv", index=False, encoding="utf-8-sig")
    with (tables / "19_industry_alignment_tests.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, ensure_ascii=False, indent=2)
    subindustry_contrast.to_csv(
        tables / "20_within_l1_subindustry_contrast.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_annotated_original_order(result.correlation, group, metadata, output)
    plot_clustered_group_modes(group, metadata, tree, order, labels, composition, output)
    plot_alignment_statistics(pair_table, composition, tests, scores, output)
    plot_within_industry_subclusters(group, metadata, ind_summary, output)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
