#!/usr/bin/env python3
"""Download maximal Yahoo daily adjusted-close history and select 100 early histories.

Selection is restricted to the existing fixed hs300 research cohort. All candidate
responses are cached; selected source responses and source hashes are retained.
No forward filling, price repair, synthetic prices or cross-provider splicing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time as day_time, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SHANGHAI = ZoneInfo('Asia/Shanghai')


def completed_date() -> date:
    now = datetime.now(SHANGHAI)
    # A conservative buffer for publication of the completed Chinese daily bar.
    return now.date() if now.time() >= day_time(16) else now.date()-timedelta(days=1)


def longest_run(mask) -> int:
    current = best = 0
    for value in mask:
        current = current+1 if value else 0
        best = max(best, current)
    return best


def listing_information(ticker: str, cache: Path, retries: int):
    market = '1' if ticker.endswith('.SS') else '0'
    query = urllib.parse.urlencode({'secid': market+'.'+ticker.split('.')[0], 'fields': 'f57,f58,f189'})
    url = 'https://push2.eastmoney.com/api/qt/stock/get?'+query
    path = cache / 'listing' / f'{ticker}.json.gz'
    last_error = ''
    for attempt in range(retries):
        try:
            if path.exists():
                payload = gzip.decompress(path.read_bytes())
            else:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'})
                with urllib.request.urlopen(req, timeout=20) as response:
                    payload = response.read()
            data = json.loads(payload).get('data') or {}
            if data.get('f57') != ticker.split('.')[0]:
                raise ValueError('Missing/mismatched listing metadata')
            listing = datetime.strptime(str(data['f189']), '%Y%m%d').date()
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists(): path.write_bytes(gzip.compress(payload, mtime=0))
            return listing, data['f58'], url, hashlib.sha256(payload).hexdigest()
        except Exception as exc:
            last_error = str(exc)
            if attempt+1 < retries: time.sleep(min(2**attempt, 8))
    raise ValueError('Cannot verify listing date: '+last_error)


def request_history(ticker: str, cutoff: date, cache: Path, retries: int):
    end_exclusive = datetime.combine(cutoff+timedelta(days=1), day_time(), SHANGHAI)
    query = urllib.parse.urlencode({'period1': 0, 'period2': int(end_exclusive.timestamp()),
                                   'interval': '1d', 'events': 'div,splits'})
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{query}'
    cached = cache / f'{ticker}.json.gz'
    payload = None
    last_error = 'request failed'
    try:
        listing_date, stock_name, listing_url, listing_sha = listing_information(ticker, cache, retries)
    except Exception as exc:
        return {'ticker': ticker, 'status': 'failed', 'error': str(exc), 'request_url': url}, None, None
    for attempt in range(retries):
        try:
            if cached.exists():
                payload = gzip.decompress(cached.read_bytes())
            else:
                request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json', 'Referer': 'https://finance.yahoo.com/'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read()
            document = json.loads(payload)
            error = document.get('chart', {}).get('error')
            if error:
                raise ValueError(str(error))
            result = document['chart']['result'][0]
            stamps = result.get('timestamp', [])
            indicators = result['indicators']
            adjusted = indicators.get('adjclose', [{}])[0].get('adjclose', [])
            quotes = indicators['quote'][0]
            if not stamps or len(adjusted) != len(stamps):
                raise ValueError('Missing daily timestamps or adjusted-close series')
            dates = pd.to_datetime(stamps, unit='s', utc=True).tz_convert(SHANGHAI).tz_localize(None).normalize()
            raw = pd.DataFrame({'Date': dates, 'AdjClose': adjusted})
            for field in ['open', 'high', 'low', 'close', 'volume']:
                raw[field] = quotes.get(field, [None]*len(stamps))
            raw = raw.loc[raw.Date.dt.date <= cutoff].sort_values('Date').reset_index(drop=True)
            vendor_first_date = str(raw.Date.iloc[0].date())
            pre_listing_rows = int((raw.Date.dt.date < listing_date).sum())
            raw = raw.loc[raw.Date.dt.date >= listing_date].reset_index(drop=True)
            if raw.Date.duplicated().any():
                raise ValueError('Duplicate daily dates')
            prices = pd.to_numeric(raw.AdjClose, errors='coerce')
            valid = prices.notna() & np.isfinite(prices) & (prices > 0)
            selected = raw.loc[valid].copy()
            if len(selected) < 1000:
                raise ValueError(f'Only {len(selected)} positive adjusted prices')
            frame = selected[['Date', 'AdjClose']].rename(columns={'AdjClose': 'ClosePrice'}).reset_index(drop=True)
            frame['ClosePrice'] = frame.ClosePrice.astype(float)
            if frame.ClosePrice.nunique() < 20:
                raise ValueError('Near-constant price history')
            if result['meta'].get('dataGranularity') != '1d':
                raise ValueError('Provider did not return daily bars')
            cached.parent.mkdir(parents=True, exist_ok=True)
            if not cached.exists():
                cached.write_bytes(gzip.compress(payload, mtime=0))
            simple = frame.ClosePrice.pct_change().dropna()
            gaps = frame.Date.diff().dt.days
            row = {'ticker': ticker, 'status': 'ok', 'provider': 'Yahoo Finance chart',
                   'stock_name': stock_name, 'listing_date': str(listing_date),
                   'listing_source_url': listing_url, 'listing_json_sha256': listing_sha,
                   'vendor_first_date': vendor_first_date, 'pre_listing_rows_removed': pre_listing_rows,
                   'adjustment': 'indicators.adjclose; vendor split/dividend adjustment',
                   'currency': result['meta'].get('currency'),
                   'first_date': str(frame.Date.iloc[0].date()), 'last_date': str(frame.Date.iloc[-1].date()),
                   'valid_price_rows': len(frame), 'provider_rows_to_cutoff': len(raw),
                   'invalid_or_missing_price_rows': int((~valid).sum()),
                   'zero_simple_return_days': int(np.isclose(simple, 0, atol=1e-14).sum()),
                   'max_consecutive_zero_returns': longest_run(np.isclose(simple, 0, atol=1e-14)),
                   'absolute_simple_return_gt25pct': int((abs(simple) > .25).sum()),
                   'largest_date_gap_days': int(gaps.max()),
                   'weekend_rows': int((frame.Date.dt.weekday >= 5).sum()),
                   'provider_first_trade_timestamp': result['meta'].get('firstTradeDate'),
                   'dividend_events': len(result.get('events', {}).get('dividends', {})),
                   'split_events': len(result.get('events', {}).get('splits', {})),
                   'requested_end_inclusive': str(cutoff), 'request_url': url,
                   'raw_json_sha256': hashlib.sha256(payload).hexdigest(),
                   'cached_response': str(cached),
                   'fetched_at_utc': datetime.now(timezone.utc).isoformat()}
            return row, frame, selected
        except Exception as exc:
            last_error = str(exc)
            if isinstance(exc, urllib.error.HTTPError) and exc.code in [404, 422]:
                break
            if attempt+1 < retries:
                time.sleep(min(2**attempt, 8))
    return {'ticker': ticker, 'status': 'failed', 'error': last_error,
            'request_url': url}, None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source', type=Path, help='Candidate prices folder; default data/hs300/prices')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--end', type=date.fromisoformat, default=completed_date())
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--dataset-id', default='hs300_long100')
    args = parser.parse_args()
    if args.end > completed_date():
        parser.error('截止日期不能晚于最近一个已完成日线的日期。')
    root = args.root.expanduser().resolve()
    source = (args.source or root / 'data/hs300/prices').expanduser().resolve()
    codes = sorted(p.stem for p in source.glob('*.pkl'))
    if len(codes) < args.count:
        parser.error(f'Only {len(codes)} candidate stocks for requested {args.count}')
    cache = root / '.cache' / 'long_history' / str(args.end)
    out = root / 'data' / args.dataset_id
    if (out / 'summary.json').exists():
        parser.error(f'{out} already contains a completed dataset; use another --dataset-id')
    audit = root / 'outputs' / 'downloads' / args.dataset_id
    audit.mkdir(parents=True, exist_ok=True)
    rows, frames, rawframes = [], {}, {}
    print(f'Candidates={len(codes)}; target={args.count}; last completed date <= {args.end}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(request_history, code, args.end, cache, args.retries): code for code in codes}
        for future in as_completed(jobs):
            row, frame, rawframe = future.result()
            rows.append(row)
            if frame is not None:
                frames[row['ticker']] = frame
                rawframes[row['ticker']] = rawframe
            if len(rows) % 20 == 0 or len(rows) == len(codes):
                pd.DataFrame(rows).sort_values('ticker').to_csv(audit / 'candidate_download_progress.csv', index=False)
                print(f'Checked {len(rows)}/{len(codes)}; successful={len(frames)}', flush=True)
    candidates = pd.DataFrame(rows).sort_values('ticker').reset_index(drop=True)
    candidates['eligibility_note'] = np.select(
        [candidates.status.ne('ok'), candidates.last_date.fillna('').lt(str(args.end-timedelta(days=14))),
         candidates.max_consecutive_zero_returns.fillna(0).ge(500)],
        ['download_or_listing_verification_failed', 'last_date_stale', 'unchanged_price_run_at_least_500_days'],
        default='eligible')
    eligible = candidates.loc[candidates.status == 'ok'].copy()
    # Stale/delisted series cannot silently shorten the latest common interval.
    fresh_after = str(args.end-timedelta(days=14))
    eligible = eligible.loc[eligible.last_date >= fresh_after]
    # Avoid a zero-variance 500-day training window in the existing research pipeline.
    eligible = eligible.loc[eligible.max_consecutive_zero_returns < 500]
    eligible = eligible.sort_values(['first_date', 'valid_price_rows', 'ticker'], ascending=[True, False, True])
    if len(eligible) < args.count:
        candidates.to_csv(audit / 'candidate_histories.csv', index=False)
        raise RuntimeError(f'Only {len(eligible)} recent long histories; need {args.count}')
    selected = eligible.head(args.count).copy().reset_index(drop=True)
    selected.insert(0, 'selection_rank', np.arange(1, len(selected)+1))
    selected_codes = selected.ticker.tolist()
    # Retain each chosen stock's full vendor history; compute a separate aligned matrix.
    prices_dir = out / 'prices'; csv_dir = out / 'daily_details'; evidence = out / 'provenance/raw_yahoo'
    listing_evidence = out / 'provenance/listing_dates'
    for folder in [prices_dir, csv_dir, evidence, listing_evidence]: folder.mkdir(parents=True, exist_ok=True)
    for code in selected_codes:
        frames[code].to_pickle(prices_dir / f'{code}.pkl')
        rawframes[code].to_csv(csv_dir / f'{code}.csv.gz', index=False, compression='gzip')
        shutil.copy2(cache / f'{code}.json.gz', evidence / f'{code}.json.gz')
        shutil.copy2(cache / 'listing' / f'{code}.json.gz', listing_evidence / f'{code}.json.gz')
    candidates['selected'] = candidates.ticker.isin(selected_codes)
    candidates.to_csv(out / 'candidate_histories.csv', index=False)
    selected.to_csv(out / 'selected_stocks.csv', index=False)
    aligned = pd.concat([frames[c].set_index('Date').ClosePrice.rename(c) for c in selected_codes], axis=1, join='inner').sort_index()
    aligned.index.name = 'Date'
    assert aligned.shape[1] == args.count and not aligned.isna().any().any()
    assert np.isfinite(aligned.to_numpy()).all() and (aligned > 0).all().all()
    aligned.to_csv(out / 'common_adjusted_close.csv.gz', compression='gzip')
    joint_start = selected.first_date.max(); joint_end = selected.last_date.min()
    union = pd.concat([frames[c].set_index('Date').ClosePrice.rename(c) for c in selected_codes], axis=1).loc[joint_start:joint_end]
    missing = pd.DataFrame({'ticker': union.columns, 'missing_dates_in_joint_union': union.isna().sum().to_numpy()})
    missing.to_csv(out / 'missing_dates_audit.csv', index=False)
    # Existing industry/benchmark metadata are snapshots, not historical constituent labels.
    metadata_dir = root / 'data/metadata'
    for suffix in ['industry_raw.csv', 'industry_mapping.csv']:
        old = metadata_dir / f'hs300_{suffix}'
        if old.exists():
            metadata = pd.read_csv(old)
            metadata.loc[metadata.ticker.isin(selected_codes)].to_csv(metadata_dir / f'{args.dataset_id}_{suffix}', index=False)
    benchmark = metadata_dir / 'hs300_index_response.json'
    if benchmark.exists(): shutil.copy2(benchmark, metadata_dir / f'{args.dataset_id}_index_response.json')
    summary = {'dataset_id': args.dataset_id, 'source_pool': 'Existing fixed 288-stock hs300 research cohort; not current index membership',
               'candidate_count': len(codes), 'successful_candidates': len(frames), 'selected_count': len(selected),
               'selection_rule': 'Verified listing date floor; recent last date within 14 calendar days; fewer than 500 consecutive unchanged prices; earliest available adjusted-price start first; ties by valid-row count descending, then ticker',
               'requested_end_inclusive': str(args.end), 'individual_earliest_start': selected.first_date.min(),
               'individual_latest_start': selected.first_date.max(),
               'individual_valid_rows_min': int(selected.valid_price_rows.min()),
               'individual_valid_rows_max': int(selected.valid_price_rows.max()),
               'common_price_start': str(aligned.index[0].date()), 'common_price_end': str(aligned.index[-1].date()),
               'common_price_rows': len(aligned), 'common_return_rows': len(aligned)-1,
               'common_price_calendar_years': (aligned.index[-1]-aligned.index[0]).days/365.2425,
               'no_forward_fill': True, 'prices': 'Yahoo split/dividend adjusted close; no repair or cross-provider splice',
               'max_weekend_rows_per_stock': int(selected.weekend_rows.max()),
               'selected_pre_listing_rows_removed': int(selected.pre_listing_rows_removed.sum()),
               'selected_max_consecutive_zero_returns': int(selected.max_consecutive_zero_returns.max()),
               'downloaded_utc': datetime.now(timezone.utc).isoformat()}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / 'README.md').write_text(f'''# 沪深固定股票池长历史{args.count}股

从现有288股固定研究股票池检查最长可用日频复权记录，先按东方财富上市日期截断供应商的上市前记录，排除连续500日及以上价格不变的历史，再按最早起始日期选择{args.count}股。不是当前沪深300成分的历史重建，也不是动态成分股指数。

- 单股最早日期：{summary['individual_earliest_start']}。
- 第{args.count}只股票的起始日期：{summary['individual_latest_start']}。
- 真正共有价格日期：{summary['common_price_start']} 至 {summary['common_price_end']}，{len(aligned)}日，约{summary['common_price_calendar_years']:.1f}年。
- 收益观测：{len(aligned)-1}日。
- 截止日期：{args.end}，排除未完成的当日日线。

`prices/`为Date、ClosePrice两列pickle，可直接交给run.py；每只股票保留自身完整可用历史。
`daily_details/`保存复权收盘、供应商OHLC及成交量；`provenance/raw_yahoo/`保存100份原始接口响应。
`provenance/listing_dates/`保存100份上市日期核对响应。上市前记录共移除{summary['selected_pre_listing_rows_removed']}条。
`candidate_histories.csv`记录全部候选及失败；`selected_stocks.csv`记录选择顺序和来源URL、哈希。
`common_adjusted_close.csv.gz`是100股共同日期的价格矩阵；没有补齐、前向填充或合成价格。

股票按历史可得性和事后数据质量选择，有上市历史/幸存者偏差，不能宣称代表1990年代以来每个时点的沪深300。供应商的最早可用记录不直接等同交易所上市日期；供应商复权历史可能随后续分红而更新。原始OHLC与AdjClose不是同一种调整口径，分析入口只使用AdjClose。停牌期间供应商可能提供连续不变价格；本程序没有自行填充，也未将它们当成新的成交记录。

运行：`python run.py {args.dataset_id}`。
复抓：`python download_long_history.py --end {args.end} --dataset-id 另一个新名字`。
''', encoding='utf-8')
    catalog_path = root / 'datasets.json'
    catalog = json.loads(catalog_path.read_text())
    catalog[args.dataset_id] = {'name_zh': f'沪深固定股票池长历史{args.count}股',
        'prices': f'data/{args.dataset_id}/prices', 'files': args.count, 'source_pool': 'hs300',
        'common_start': summary['common_price_start'], 'common_end': summary['common_price_end']}
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
    manifest = []
    for file in sorted((root / 'data').rglob('*')):
        if file.is_file():
            manifest.append({'path': str(file.relative_to(root)), 'size': file.stat().st_size,
                             'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    (root / 'data_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
