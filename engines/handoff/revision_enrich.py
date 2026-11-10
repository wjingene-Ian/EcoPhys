"""Index benchmark and compact source data for editable revision charts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
import rmt_analysis as ra
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'revision_data'
p=json.loads((OUT/'revision_pack.json').read_text())
codes=pd.read_csv(ROOT/'rmt_tables/05_seeded_stock_selection.csv').query('in_fixed_100').ticker.tolist()
r=ra.load_returns(ROOT/'tables/07_log_return_matrix.csv')[codes]
x=r.to_numpy();z=ra.standardize(x);res=ra.rmt_decomposition(x);v=ra.orient_vector(res.eigenvectors[:,-1])
api=json.loads((OUT/'index_api_response.json').read_text())
idx=pd.DataFrame([s.split(',') for s in api['response']['data']['klines']])
idx.index=pd.to_datetime(idx[0]);close=idx[2].astype(float)
index_return=np.log(close/close.shift()).dropna()
common=r.index.intersection(index_return.index)
d=pd.DataFrame({'pc_standardized_returns':z@v,'course_raw_return_factor':x@v,
                'equal_weight_standardized_proxy':z.mean(axis=1)},index=r.index).loc[common]
d['CSI300_log_return']=index_return.loc[common]
d.to_csv(OUT/'official_index_comparison.csv')
p['market']={'pc_proxy_correlation':float(d.pc_standardized_returns.corr(d.equal_weight_standardized_proxy)),
 'official_days':len(d),'official_pc_correlation':float(d.pc_standardized_returns.corr(d.CSI300_log_return)),
 'official_raw_factor_correlation':float(d.course_raw_return_factor.corr(d.CSI300_log_return)),
 'source_url':api['url'],'source':'东方财富沪深300指数行情'}
d['pc_standardized']=(d.pc_standardized_returns-d.pc_standardized_returns.mean())/d.pc_standardized_returns.std(ddof=1)
d['index_standardized']=(d.CSI300_log_return-d.CSI300_log_return.mean())/d.CSI300_log_return.std(ddof=1)
p['market']['scatter_x']=d.index_standardized.round(8).tolist();p['market']['scatter_y']=d.pc_standardized.round(8).tolist()
bins=np.linspace(-8,8,65);counts,edges=np.histogram(z,bins=bins)
cent=(edges[:-1]+edges[1:])/2
p['density']={'x':cent.tolist(),'empirical':(counts/z.size/np.diff(edges)).tolist(),'gaussian':norm.pdf(cent).tolist()}
b=pd.read_csv(ROOT/'presentation_data/bootstrap_delta_sharpe.csv').iloc[:,0]
counts,edges=np.histogram(b,bins=np.linspace(-.15,.35,36))
p['bootstrap_histogram']={'x':((edges[:-1]+edges[1:])/2).tolist(),'counts':counts.tolist()}
p['split_summary']={}
split=pd.read_csv(OUT/'chronological_mode_reliability.csv')
for name,lo,hi in [('market',1,1),('large_modes',2,10),('near_bulk',11,20)]:
 sub=split.query('@lo<=train_rank<=@hi')
 p['split_summary'][name]={'mean_best_overlap':float(sub.best_overlap.mean()),'mean_test_variance':float(sub.test_variance.mean())}
sectors=pd.read_csv(OUT/'all_mode_industry_loadings.csv')
des=[]
for k in range(1,11):
 a=sectors.query('rank==@k and n>=2').sort_values('energy_enrichment',ascending=False).head(3)
 des.append({'rank':k,'top':a.to_dict('records')})
p['mode_interpretations']=des
(OUT/'revision_pack.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))
print(json.dumps({'market':{k:v for k,v in p['market'].items() if 'scatter' not in k},'split':p['split_summary']},ensure_ascii=False,indent=2))
