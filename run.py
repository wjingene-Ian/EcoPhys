#!/usr/bin/env python3
"""统一命令行入口：python run.py hs300 --analysis basic。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', help='数据集简称、价格文件夹路径，或 all')
    parser.add_argument('--analysis', choices=['basic', 'audit', 'rmt', 'structure', 'industry', 'scales', 'modes', 'all'], default='basic')
    parser.add_argument('--output', type=Path, help='输出根目录，默认本项目 outputs/')
    parser.add_argument('--assets', type=int, help='显式抽样数量；默认全部股票。100 可复现旧股票池')
    parser.add_argument('--solver', choices=['auto', 'slsqp', 'projected'], default='auto',
                        help='rmt组合求解器；auto在超过100股时使用快速投影法，否则保留旧SLSQP')
    parser.add_argument('--list', action='store_true', help='列出随项目提供的数据集')
    parser.add_argument('--quick', action='store_true', help='减少随机重复次数的流程预览，不用于最终结论')
    parser.add_argument('--tau-max', type=int, default=None, help='modes 最大 τ；默认250，quick默认20')
    parser.add_argument('--null-reps', type=int, default=None, help='随机对照次数；默认100')
    parser.add_argument('--permutations', type=int, default=None, help='行业/群组标签置换次数；默认999')
    parser.add_argument('--bootstrap-reps', type=int, default=None, help='组合差异重采样次数；默认2000')
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'datasets.json').read_text())
    if args.list:
        for key, spec in catalog.items():
            print(f"{key:12} {spec['name_zh']}  {spec['files']}只股票  {spec['prices']}")
        return
    if not args.dataset:
        parser.error('请提供数据集简称或价格文件夹；使用 --list 查看全部数据集。')
    if args.assets is not None and args.assets < 20:
        parser.error('当前5%权重上限至少需要20只股票。')
    args.tau_max = args.tau_max if args.tau_max is not None else (20 if args.quick else 250)
    args.null_reps = args.null_reps if args.null_reps is not None else (3 if args.quick else 100)
    args.permutations = args.permutations if args.permutations is not None else (19 if args.quick else 999)
    args.bootstrap_reps = args.bootstrap_reps if args.bootstrap_reps is not None else (20 if args.quick else 2000)
    if not 1 <= args.tau_max <= 250 or min(args.null_reps, args.permutations, args.bootstrap_reps) < 1:
        parser.error('τ须在1–250之间；随机重复次数须为正整数。')
    # Set these before importing the numerical libraries.
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    cache = ROOT / '.cache' / 'matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from pipeline import run_dataset

    selected = list(catalog) if args.dataset == 'all' else [args.dataset]
    failed = []
    for dataset in selected:
        if dataset in catalog:
            spec = dict(catalog[dataset])
            prices = ROOT / spec['prices']
        else:
            prices = Path(dataset).expanduser().resolve()
            if not prices.is_dir():
                parser.error(f'未知数据集或文件夹不存在：{dataset}')
            dataset = prices.parent.name if prices.name == 'prices' else prices.name
            spec = {'name_zh': dataset, 'prices': str(prices)}
        try:
            run_dataset(dataset, spec, prices, args, ROOT)
        except Exception as exc:
            failed.append(dataset)
            print(f'失败：{dataset}：{exc}', file=sys.stderr)
            if len(selected) == 1:
                raise
    if failed:
        parser.exit(1, '未完成的数据集：' + ', '.join(failed) + '\n')


if __name__ == '__main__':
    main()
