"""Analyze one index price folder from the command line.

    python analyze_index.py /path/to/index_prices --output /path/to/results
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def default_output(input_folder: Path) -> Path:
    return ROOT / 'runs' / input_folder.name


def analyze(input_folder: Path, output_folder: Path | None = None) -> tuple[dict, Path, str]:
    import pandas as pd
    from build_global import configure_plots, run_group

    source = input_folder.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f'输入文件夹不存在：{source}')
    destination = (output_folder or default_output(source)).expanduser().resolve()
    configure_plots()
    summary = run_group(source, destination)
    metrics = pd.read_csv(destination / 'full_period_metrics.csv').set_index('strategy')
    raw = metrics.loc['原始协方差']
    clean = metrics.loc['RMT 去噪']
    report = (
        f"指数文件夹：{summary['group']}\n"
        f"纳入股票：{summary['assets']} 只；共同价格日：{summary['price_days']} 天\n"
        f"最大特征值：{summary['top_eigenvalue']:.2f}；高于 MP 上界：{summary['above_upper']} 个\n"
        f"样本外 Sharpe：原始 {raw['sharpe']:.3f} → 去噪 {clean['sharpe']:.3f}\n"
        f"最大回撤：原始 {raw['max_drawdown']:.1%} → 去噪 {clean['max_drawdown']:.1%}\n\n"
        f"结果文件夹：{destination}\n"
        '图表：figures/01_ranked_spectrum.png、02_backtest.png、03_stress_comparison.png'
    )
    return summary, destination, report


def main() -> None:
    parser = argparse.ArgumentParser(description='分析一个指数价格文件夹中的全部可用股票。')
    parser.add_argument('input_folder', type=Path, help='包含股票 .pkl 文件的文件夹')
    parser.add_argument('--output', type=Path, help='结果文件夹；默认 runs/<输入文件夹名>')
    args = parser.parse_args()
    try:
        _, _, report = analyze(args.input_folder, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'分析失败：{exc}\n')
    print('\n' + report)


if __name__ == '__main__':
    main()
