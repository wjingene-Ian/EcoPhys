"""Project 3 basic RMT reproduction from the provided CSI 300 price snapshot."""

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


ROOT = Path(__file__).resolve().parent
DATA = Path("/Users/wangxujing/Documents/courses/经济物理学/task/data300_price")
OUT = ROOT / "analysis"
FIG = OUT / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont("/System/Library/Fonts/Hiragino Sans GB.ttc")
    plt.rcParams.update({
        "font.family": "Hiragino Sans GB",
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })

    files = sorted(DATA.glob("*.pkl"))
    assert len(files) == 288
    selected = files

    series = []
    manifest = []
    for p in selected:
        df = pd.read_pickle(p)
        assert list(df.columns) == ["Date", "ClosePrice"]
        x = df.set_index("Date")["ClosePrice"].astype(float).sort_index()
        assert x.index.is_unique and np.isfinite(x).all() and (x > 0).all()
        series.append(x.rename(p.stem))
        manifest.append({
            "ticker": p.stem,
            "exchange": p.suffixes[-2].lstrip(".") if len(p.suffixes) > 1 else p.stem.split(".")[-1],
            "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        })
    prices = pd.concat(series, axis=1, join="inner").sort_index()
    assert prices.shape[1] == len(files) and not prices.isna().any().any()
    returns = np.log(prices).diff().iloc[1:]
    X = returns.to_numpy(dtype=float)
    T, N = X.shape
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    assert np.all(stds > 0)
    Z = (X - means) / stds
    C = Z.T @ Z / T
    raw_cov = np.cov(X, rowvar=False, ddof=0)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    q = N / T
    lower = (1 - np.sqrt(q)) ** 2
    upper = (1 + np.sqrt(q)) ** 2
    above = int((eigvals > upper).sum())
    below = int((eigvals < lower).sum())
    within = N - above - below
    v1 = eigvecs[:, 0].copy()
    if v1.sum() < 0:
        v1 *= -1
    participation = 1 / np.sum(v1 ** 4)
    time_mode = Z @ v1 / np.sqrt(eigvals[0])
    assert abs(np.trace(C) - N) < 1e-8
    assert abs(np.var(time_mode) - 1) < 1e-8
    assert abs(np.mean(np.diag(C)) - 1) < 1e-8

    pd.DataFrame(manifest).to_csv(OUT / "included_all.csv", index=False)
    pd.DataFrame({"rank": np.arange(1, N + 1), "eigenvalue": eigvals,
                  "MP_class": np.where(eigvals > upper, "above", np.where(eigvals < lower, "below", "bulk"))}).to_csv(
        OUT / "eigenvalues_all.csv", index=False)
    summary = {
        "source": str(DATA), "selection_rule": "Every readable positive-price pkl file with a common date; no sampling",
        "input_files": len(files), "selected": N, "shanghai": sum(p.stem.endswith('.SS') for p in files), "shenzhen": sum(p.stem.endswith('.SZ') for p in files),
        "price_rows": len(prices), "return_rows": T,
        "price_start": str(prices.index.min().date()), "return_start": str(returns.index.min().date()),
        "end": str(prices.index.max().date()), "q": q, "mp_lower": lower, "mp_upper": upper,
        "top_eigenvalue": float(eigvals[0]), "top_variance_share": float(eigvals[0] / N),
        "above_upper": above, "within_bulk": within, "below_lower": below,
        "participation_ratio": float(participation), "positive_loading_fraction": float(np.mean(v1 > 0)),
        "all_eigenvalues_positive": bool(np.all(eigvals > 0)),
        "offdiag_mean": float(C[np.triu_indices(N, 1)].mean()),
        "offdiag_median": float(np.median(C[np.triu_indices(N, 1)])),
        "price_provenance_note": "Provided local price snapshot; live API request provenance and adjustment convention not independently verified here.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # Figure 1: matrix and off-diagonal distribution.
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.7), gridspec_kw={"width_ratios": [1.2, 1]})
    im = axes[0].imshow(C, cmap="RdBu_r", vmin=-0.6, vmax=0.6, interpolation="nearest")
    axes[0].set_title(f"{N} 只股票的经验标准化协方差矩阵 C")
    axes[0].set_xlabel("股票序号"); axes[0].set_ylabel("股票序号")
    fig.colorbar(im, ax=axes[0], fraction=.046, pad=.03, label="相关系数")
    off = C[np.triu_indices(N, 1)]
    axes[1].hist(off, bins=50, color="#2D6688", alpha=.88)
    axes[1].axvline(off.mean(), color="#CA6545", linewidth=2, label=f"均值 = {off.mean():.3f}")
    axes[1].set_title("非对角元素分布")
    axes[1].set_xlabel("两只股票的相关系数"); axes[1].set_ylabel("股票对数")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "01_empirical_matrix.png", dpi=180)
    plt.close(fig)

    # Figure 2: MP density for the noise bulk and observed small eigenvalues.
    xs = np.linspace(lower + 1e-5, upper - 1e-5, 1000)
    mp = np.sqrt((upper - xs) * (xs - lower)) / (2 * np.pi * q * xs)
    fig, ax = plt.subplots(figsize=(12.8, 6.5))
    bins = np.linspace(0, 3.2, 42)
    shown = eigvals[eigvals <= 3.2]
    ax.hist(shown, bins=bins, weights=np.full(len(shown), 1 / N), color="#9CC3D6",
            edgecolor="white", alpha=.88, label="经验特征值（只显示 ≤3.2，不重新归一化）")
    ax.plot(xs, mp * (bins[1] - bins[0]), color="#176F60", linewidth=3,
            label="MP 理论：独立、同方差零假设")
    ax.axvspan(lower, upper, color="#3E947A", alpha=.09)
    ax.axvline(lower, color="#176F60", linestyle="--", linewidth=1.5)
    ax.axvline(upper, color="#176F60", linestyle="--", linewidth=1.5)
    ax.text(lower + .03, ax.get_ylim()[1] * .92, f"MP low = {lower:.3f}", color="#176F60")
    ax.text(upper + .03, ax.get_ylim()[1] * .62, f"MP high = {upper:.3f}", color="#176F60")
    ax.set(xlabel="特征值 λ", ylabel="每个区间占全部特征值的比例", title="经验谱与 MP 随机谱主体")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "02_mp_bulk.png", dpi=180)
    plt.close(fig)

    # Figure 3: every ranked eigenvalue, retaining the large market outlier.
    fig, ax = plt.subplots(figsize=(12.8, 6.5))
    ax.scatter(np.arange(1, N + 1), eigvals, s=28,
               c=np.where(eigvals > upper, "#C65E46", "#2D6688"))
    ax.axhspan(lower, upper, color="#3E947A", alpha=.12, label="MP 噪声区间")
    ax.axhline(upper, color="#176F60", linestyle="--", linewidth=1.5, label=f"MP 上界 {upper:.3f}")
    ax.axhline(lower, color="#176F60", linestyle=":", linewidth=1.3)
    ax.annotate(f"最大 λ = {eigvals[0]:.2f}\n占总方差 {eigvals[0]/N:.1%}", (1, eigvals[0]),
                xytext=(13, eigvals[0] * .8), arrowprops={"arrowstyle": "->", "color": "#C65E46"},
                color="#9B4534", fontsize=14)
    ax.set(xlabel="特征值排名（降序）", ylabel="特征值 λ（对数刻度）",
           title=f"完整经验谱：{above} 个高于 MP 上界，{below} 个低于下界")
    ax.set_yscale("log"); ax.set_ylim(max(.01, eigvals[-1] * .8), eigvals[0] * 1.3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "03_ranked_spectrum.png", dpi=180)
    plt.close(fig)

    # Figure 4: loadings and the time realization of the leading component.
    fig, axes = plt.subplots(2, 1, figsize=(13.7, 6.9), gridspec_kw={"height_ratios": [1, 1.2]})
    order = np.argsort(v1)
    axes[0].bar(np.arange(N), v1[order], width=1, color="#2D6688")
    axes[0].set(title=f"第一特征向量：同号占比 {np.mean(v1 > 0):.0%}，有效参与数 {participation:.1f}",
                ylabel="股票载荷", xlabel="股票（按载荷排序）")
    axes[1].plot(returns.index, time_mode, linewidth=.8, color="#245894")
    axes[1].axhline(0, color="#8C9AA4", linewidth=.8)
    axes[1].set(title="第一模态的逐日得分（标准化，方差为 1）", xlabel="交易日期", ylabel="无量纲得分")
    fig.tight_layout()
    fig.savefig(FIG / "04_market_mode.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
