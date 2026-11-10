"""Course completion experiments. All inputs and outputs remain in Analysis_codex."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import rmt_analysis as ra

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'presentation_data'
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261109)

def metric(x):
    x = np.asarray(x)
    w = np.r_[1., np.cumprod(1+x)]
    return dict(days=len(x), total_return=float(w[-1]-1),
                volatility=float(x.std(ddof=1)*np.sqrt(242)),
                sharpe=float(x.mean()/x.std(ddof=1)*np.sqrt(242)),
                drawdown=float((w/np.maximum.accumulate(w)-1).min()))

def eta(v, labels):
    mu = v.mean()
    return float(sum(np.sum(labels==g)*(v[labels==g].mean()-mu)**2
                     for g in np.unique(labels))/np.sum((v-mu)**2))

def main():
    returns=ra.load_returns(ROOT/'tables/07_log_return_matrix.csv')
    select=pd.read_csv(ROOT/'rmt_tables/05_seeded_stock_selection.csv')
    cols=select.loc[select.in_fixed_100,'ticker'].tolist()
    x=returns[cols]
    result=ra.rmt_decomposition(x.to_numpy())
    pd.DataFrame([ra.matrix_summary(x.to_numpy(),'core_100')]).to_csv(OUT/'core_100.csv',index=False)
    meta=pd.read_csv(ROOT/'rmt_tables/14_stock_industry_mapping.csv').set_index('ticker').loc[cols]
    labels=meta.industry_l1.to_numpy()
    z=ra.standardize(x.to_numpy())
    v=ra.orient_vector(result.eigenvectors[:,-1])
    pc=z@v
    market=z.mean(axis=1)
    eig=[]
    for rank in range(1,9):
        vec=ra.orient_vector(result.eigenvectors[:,-rank])
        observed=eta(vec,labels)
        null=[eta(vec,RNG.permutation(labels)) for _ in range(999)]
        eig.append(dict(rank=rank,eigenvalue=float(result.eigenvalues[-rank]),
            industry_eta2=observed,p=float((1+np.sum(np.asarray(null)>=observed))/1000)))
    pd.DataFrame(eig).to_csv(OUT/'eigenvector_industry.csv',index=False)
    pd.DataFrame(dict(ticker=cols,stock_name=meta.stock_name.to_numpy(),industry=labels,
                     market_loading=v,second_loading=ra.orient_vector(result.eigenvectors[:,-2]))).to_csv(OUT/'loadings_100.csv',index=False)
    null=[]
    for _ in range(100):
        shuffled=np.column_stack([RNG.permutation(z[:,i]) for i in range(z.shape[1])])
        null.append(ra.rmt_decomposition(shuffled).eigenvalues[-1])
    pd.DataFrame({'lambda_max':null}).to_csv(OUT/'shuffle_control.csv',index=False)
    daily=pd.read_csv(ROOT/'rmt_tables/09_portfolio_daily_returns.csv',index_col=0,parse_dates=True)
    periods=[('2022下跌年','2022-01-01','2022-12-31'),
             ('2024急涨急跌','2024-09-24','2024-10-31'),
             ('2025四月冲击','2025-04-01','2025-04-30')]
    rows=[]
    for label,start,end in periods:
        for strategy in daily:
            rows.append(dict(period=label,strategy=strategy,**metric(daily.loc[start:end,strategy])))
    pd.DataFrame(rows).to_csv(OUT/'stress_metrics.csv',index=False)
    # Matched block bootstrap: identical sampled dates for raw and clipped strategies.
    a=daily.raw_gmv.to_numpy(); b=daily.rmt_clipped_gmv.to_numpy(); n=len(a)
    boot=[]
    for _ in range(2000):
        starts=RNG.integers(0,n,size=int(np.ceil(n/21)))
        ix=np.concatenate([(s+np.arange(21))%n for s in starts])[:n]
        boot.append(metric(b[ix])['sharpe']-metric(a[ix])['sharpe'])
    pd.DataFrame({'delta_sharpe_clipped_minus_raw':boot}).to_csv(OUT/'bootstrap_delta_sharpe.csv',index=False)
    # Retain every date; remove suspect stocks as an explicitly retrospective sensitivity.
    stock=pd.read_csv(ROOT/'tables/05_stock_summary.csv').set_index('ticker')
    anomalies=pd.read_csv(ROOT/'tables/09_individual_return_anomalies.csv')
    bad=set(stock.index[stock.max_consecutive_zero_returns>=6])|set(anomalies.ticker)
    clean=[c for c in cols if c not in bad]
    robust=[]
    for name,c in [('baseline100',cols),('retrospective_clean',clean)]:
        robust.append(ra.matrix_summary(returns[c].to_numpy(),name))
    pd.DataFrame(robust).to_csv(OUT/'data_quality_sensitivity.csv',index=False)
    # Same stocks, equal-length windows ending at distinct regimes.
    win=[]
    for end in ['2020-12-31','2022-12-31','2024-12-31','2026-09-28']:
        for length in [120,242,484]:
            sub=x.loc[:end].tail(length)
            if len(sub)==length:
                win.append(dict(end=end,window=length,**ra.matrix_summary(sub.to_numpy(),end)))
    pd.DataFrame(win).to_csv(OUT/'matched_window_sensitivity.csv',index=False)
    out=dict(market_pc_equalweight_correlation=float(np.corrcoef(pc,market)[0,1]),
             market_positive_loading_fraction=float((v>0).mean()),
             null_lambda_max_mean=float(np.mean(null)),null_lambda_max_max=float(np.max(null)),
             bootstrap_sharpe_delta=float(metric(b)['sharpe']-metric(a)['sharpe']),
             bootstrap_sharpe_delta_ci95=np.quantile(boot,[.025,.975]).tolist(),
             cleaned_asset_count=len(clean),note='clean filtering uses full sample, diagnostic only')
    (OUT/'completion_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    pack={}
    paths={'core':'presentation_data/core_100.csv','eigen':'presentation_data/eigenvector_industry.csv',
           'stress':'presentation_data/stress_metrics.csv','quality':'presentation_data/data_quality_sensitivity.csv',
           'windows':'presentation_data/matched_window_sensitivity.csv',
           'performance':'rmt_tables/10_portfolio_performance.csv',
           'counts':'rmt_tables/06b_stock_count_summary.csv',
           'regime':'rmt_tables/08_market_regime_sensitivity.csv',
           'subindustry':'rmt_tables/20_within_l1_subindustry_contrast.csv'}
    for name,p in paths.items():pack[name]=pd.read_csv(ROOT/p).to_dict('records')
    pack['summary']=out
    pack['spectrum']=result.eigenvalues[::-1].tolist()
    bins=np.linspace(0,2,21)
    centers=(bins[:-1]+bins[1:])/2
    density,_=np.histogram(result.eigenvalues,bins=bins,density=False)
    pack['bulk']={'x':[f'{v:.2f}' for v in centers], 'empirical':(density/100/.1).tolist(),
                  'mp':ra.mp_density(centers,result.q,result.lambda_minus,result.lambda_plus).tolist()}
    wealth=(1+daily).cumprod()
    sampled=wealth.iloc[np.unique(np.r_[np.arange(0,len(wealth),10),len(wealth)-1])]
    pack['wealth']={'dates':sampled.index.strftime('%Y-%m-%d').tolist(),
                    'series':{c:sampled[c].tolist() for c in sampled}}
    group_sector=meta.assign(loading=v).groupby('industry_l1').loading.agg(['mean','size']).sort_values('mean',ascending=False)
    pack['market_sector']=group_sector.reset_index().rename(columns={'industry_l1':'industry'}).to_dict('records')
    (OUT/'deck_data.json').write_text(json.dumps(pack,ensure_ascii=False,allow_nan=True))
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
