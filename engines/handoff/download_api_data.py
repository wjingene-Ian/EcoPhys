"""Optional fresh Yahoo API download. Original archived data are never overwritten.
Requires yfinance. The presentation uses archived data, not this optional refresh.
Official API wrapper documentation: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
"""
from pathlib import Path
import argparse
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=int,default=100)
    p.add_argument('--start',default='2018-12-28')
    p.add_argument('--end',default='2026-09-29',help='Yahoo end date is exclusive')
    a=p.parse_args()
    import yfinance as yf
    root=Path(__file__).resolve().parent
    codes=pd.read_csv(root/'rmt_tables/05_seeded_stock_selection.csv').ticker.tolist()[:a.assets]
    raw=yf.download(codes,start=a.start,end=a.end,auto_adjust=True,progress=False,threads=4,timeout=12)
    prices=raw['Close']
    out=root/'api_refresh'
    out.mkdir(exist_ok=True)
    if not len(prices) or prices.empty:raise RuntimeError('API returned no prices')
    prices.to_csv(out/'adjusted_close.csv')
    pd.DataFrame({'ticker':prices.columns,'valid_rows':prices.count().to_numpy()}).to_csv(out/'download_manifest.csv',index=False)
    print('Saved fresh adjusted prices to',out)

if __name__=='__main__':main()
