#!/usr/bin/env python3
"""从东方财富公开个股基本信息接口批量获取股票名称和细分行业标签。

输入为 RMT 分析的日对数收益率矩阵，列名是 Yahoo 风格证券代码
（例如 600519.SS、000001.SZ）。输出保存为可重现的 CSV 映射表。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


API_URL = "https://push2.eastmoney.com/api/qt/stock/get"
FIELDS = "f57,f58,f127"
USER_AGENT = "Mozilla/5.0 (compatible; econophysics-course-project/1.0)"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returns", type=Path, default=here / "tables" / "07_log_return_matrix.csv")
    parser.add_argument("--output", type=Path, default=here / "rmt_tables" / "13_stock_industry_raw.csv")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def eastmoney_secid(ticker: str) -> str:
    code, suffix = ticker.split(".")
    market = "1" if suffix == "SS" else "0"
    return f"{market}.{code}"


def fetch_one(ticker: str, retries: int = 4) -> dict[str, str]:
    secid = eastmoney_secid(ticker)
    query = urllib.parse.urlencode(
        {
            "secid": secid,
            "fields": FIELDS,
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        }
    )
    url = f"{API_URL}?{query}"
    last_error = ""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            data = payload.get("data") or {}
            if data.get("f58"):
                return {
                    "ticker": ticker,
                    "code": data.get("f57") or ticker.split(".")[0],
                    "stock_name": data.get("f58") or "",
                    "eastmoney_industry": data.get("f127") or "",
                    "source_url": url,
                    "fetch_error": "",
                }
            last_error = f"empty data: rc={payload.get('rc')}"
        except Exception as exc:  # network retry is deliberate
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.35 * (attempt + 1))
    return {
        "ticker": ticker,
        "code": ticker.split(".")[0],
        "stock_name": "",
        "eastmoney_industry": "",
        "source_url": url,
        "fetch_error": last_error,
    }


def main() -> None:
    args = parse_args()
    tickers = list(pd.read_csv(args.returns, nrows=0).columns[1:])
    cached: dict[str, dict[str, str]] = {}
    if args.output.exists():
        previous = pd.read_csv(args.output, dtype=str).fillna("")
        cached = {
            row["ticker"]: row
            for row in previous.to_dict(orient="records")
            if row.get("ticker") in tickers and row.get("eastmoney_industry")
        }
    pending = [ticker for ticker in tickers if ticker not in cached]
    rows: list[dict[str, str]] = list(cached.values())
    print(f"cached {len(cached)}, pending {len(pending)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, ticker): ticker for ticker in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 20 == 0 or index == len(pending):
                print(f"fetched {index}/{len(pending)} pending", flush=True)

    order = {ticker: index for index, ticker in enumerate(tickers)}
    rows.sort(key=lambda row: order[row["ticker"]])
    fetched_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    for row in rows:
        row["classification_provider"] = "Eastmoney"
        if not row.get("classification_as_of"):
            row["classification_as_of"] = fetched_at
        row["classification_note"] = "Current Eastmoney stock industry label (field f127); not historical point-in-time"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ticker",
        "code",
        "stock_name",
        "eastmoney_industry",
        "classification_provider",
        "classification_as_of",
        "classification_note",
        "source_url",
        "fetch_error",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = [row for row in rows if row["fetch_error"]]
    missing_industry = [row for row in rows if not row["eastmoney_industry"]]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "stocks": len(rows),
                "errors": len(errors),
                "missing_industry": len(missing_industry),
                "unique_industries": len({row["eastmoney_industry"] for row in rows if row["eastmoney_industry"]}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
