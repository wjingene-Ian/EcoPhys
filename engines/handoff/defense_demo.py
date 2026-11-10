"""Live parameter exercise. Example: python defense_demo.py --assets 100 --days 120."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import rmt_analysis as ra

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=int,default=100)
    p.add_argument('--days',type=int,default=504)
    p.add_argument('--threshold',type=float,default=1.0,help='multiplier on MP upper edge')
    p.add_argument('--end',default='2026-09-28')
    p.add_argument('--max-weight',type=float,default=.05)
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    returns=ra.load_returns(root/'tables/07_log_return_matrix.csv')
    chosen=pd.read_csv(root/'rmt_tables/05_seeded_stock_selection.csv').ticker.tolist()[:a.assets]
    if a.assets<2 or a.assets>returns.shape[1] or a.days<3 or a.threshold<=0:
        p.error('assets must be 2..288, days >=3, threshold >0')
    x=returns.loc[:a.end,chosen].tail(a.days)
    result=ra.rmt_decomposition(x.to_numpy())
    threshold=a.threshold*result.lambda_plus
    mask=result.eigenvalues>threshold
    eig=result.eigenvalues.copy()
    if (~mask).any(): eig[~mask]=eig[~mask].mean()
    corr=ra.normalize_to_correlation(ra.reconstruct(eig,result.eigenvectors))
    vol=x.std(ddof=1).to_numpy()
    cov=corr*vol[:,None]*vol[None,:]
    w=ra.long_only_minimum_variance(cov,a.max_weight)
    out=root/'presentation_data/live'
    out.mkdir(parents=True,exist_ok=True)
    summary=dict(N=len(chosen),T=len(x),rank=int(np.linalg.matrix_rank(result.correlation)),
        MP_upper=result.lambda_plus,threshold=threshold,signals=int(mask.sum()),
        market_fraction=float(result.eigenvalues[-1]/len(chosen)),
        effective_stocks=float(1/(w@w)),max_weight=float(w.max()))
    print(json.dumps(summary,indent=2))
    pd.DataFrame({'ticker':chosen,'weight':w}).to_csv(out/'weights.csv',index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    ax.scatter(np.arange(1,len(eig)+1),result.eigenvalues[::-1],s=16)
    ax.axhline(result.lambda_plus,color='#008C95',label='MP upper edge')
    ax.axhline(threshold,color='#C76A3B',ls='--',label='Selected threshold')
    ax.set(yscale='log',xlabel='Eigenvalue rank',ylabel='Eigenvalue',
           title=f'N={len(chosen)}, T={len(x)}, signals={mask.sum()}')
    ax.legend();fig.tight_layout();fig.savefig(out/'live_spectrum.png',dpi=180)
    print('Figure:',out/'live_spectrum.png')

if __name__=='__main__':main()
