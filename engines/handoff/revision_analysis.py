"""Revision experiments: stylized facts, within-MP information, scales and DMA.

Fixed 100-stock cohort. No portfolio claims from centered (future-using) filters.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm
from scipy.cluster.hierarchy import linkage,dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib as mpl
import rmt_analysis as ra

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'revision_data';OUT.mkdir(exist_ok=True)
FIG=ROOT/'revision_figures';FIG.mkdir(exist_ok=True)
RNG=np.random.default_rng(20261110)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial Unicode MS','Arial'],
 'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'font.size':12,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})

def save(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(FIG/f'{name}.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)

def eta(vec,labels):
    return sum((labels==g).sum()*(vec[labels==g].mean()-vec.mean())**2
               for g in np.unique(labels))/np.sum((vec-vec.mean())**2)

def bh(p):
    order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum.accumulate(
        (np.asarray(p)[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    return np.minimum(q,1)

def corr(x,gram=False):
    x=np.asarray(x)
    if not gram:x=x-x.mean(axis=0)
    lengths=np.sqrt(np.sum(x*x,axis=0))
    if np.any(lengths<1e-12):raise ValueError('Zero energy series')
    z=x/lengths
    return z.T@z

def acf(x,k):
    return np.array([np.corrcoef(x[:-k,i],x[k:,i])[0,1] for i in range(x.shape[1])])

def transform(x,method,h):
    x=pd.DataFrame(x)
    if method=='daily':return x.to_numpy()
    if method=='overlap':return x.rolling(h).sum().dropna().to_numpy()
    if method=='nonoverlap':
        a=x.to_numpy();return a[:len(a)//h*h].reshape(-1,h,a.shape[1]).sum(axis=1)
    if method=='return_detrend':return (x-x.rolling(h,center=True).mean()).dropna().to_numpy()
    if method=='profile_dma':
        y=(x-x.mean()).cumsum()
        return (y-y.rolling(h,center=True).mean()).dropna().to_numpy()
    raise ValueError(method)

def main():
    all_returns=ra.load_returns(ROOT/'tables/07_log_return_matrix.csv')
    cols=pd.read_csv(ROOT/'rmt_tables/05_seeded_stock_selection.csv').query('in_fixed_100').ticker.tolist()
    returns=all_returns[cols];x=returns.to_numpy();z=ra.standardize(x)
    meta=pd.read_csv(ROOT/'rmt_tables/14_stock_industry_mapping.csv').set_index('ticker').loc[cols]
    labels=meta.industry_l1.to_numpy();r=ra.rmt_decomposition(x)
    vals=r.eigenvalues[::-1];V=r.eigenvectors[:,::-1]
    for j in range(V.shape[1]):V[:,j]=ra.orient_vector(V[:,j])
    pack={}
    # Stylized facts: pooled standardised single-stock returns, not market proxy.
    ac=[]
    for k in range(1,61):
        a=acf(z,k);b=acf(np.abs(z),k)
        ac.append(dict(lag=k,return_median=float(np.median(a)),abs_median=float(np.median(b)),
                       return_q25=float(np.quantile(a,.25)),return_q75=float(np.quantile(a,.75))))
    ac=pd.DataFrame(ac);ac.to_csv(OUT/'stylized_autocorrelation.csv',index=False)
    tail=[dict(sigma=s,empirical=float(np.mean(np.abs(z)>s)),gaussian=float(2*norm.sf(s))) for s in [2,3,4,5]]
    pack['stylized']={'pooled_excess_kurtosis':float(kurtosis(z.ravel(),fisher=True,bias=False)),
        'median_stock_excess_kurtosis':float(np.median(kurtosis(z,axis=0,fisher=True,bias=False))),
        'tail':tail,'acf':ac.to_dict('records')}
    fig,ax=plt.subplots(figsize=(10,4.7))
    bins=np.linspace(-8,8,100);den,edges=np.histogram(z,bins,density=True)
    centers=(edges[1:]+edges[:-1])/2
    ax.plot(centers,den,label='100股标准化日收益',color='#008C95',lw=2)
    ax.plot(centers,norm.pdf(centers),'--',label='标准正态',color='#17354A',lw=2)
    ax.set(yscale='log',ylim=(1e-5,1),xlabel='标准化对数收益 z',ylabel='概率密度（对数轴）')
    ax.legend();fig.tight_layout();save(fig,'01_fat_tails')
    fig,ax=plt.subplots(figsize=(10,4.6))
    ax.plot(ac.lag,ac.return_median,label='收益 ACF 中位数',color='#17354A',lw=2)
    ax.plot(ac.lag,ac.abs_median,label='绝对收益 ACF 中位数',color='#008C95',lw=2)
    ax.axhline(0,color='#AAAAAA',lw=.6)
    ax.set(xlabel='滞后交易日',ylabel='100股逐股自相关的中位数');ax.legend();fig.tight_layout();save(fig,'02_volatility_clustering')
    # Sector association for all modes. Permutation null preserves sector sizes.
    perms=[RNG.permutation(labels) for _ in range(999)]
    mode=[];sector_rows=[]
    for j in range(100):
        obs=eta(V[:,j],labels);null=np.array([eta(V[:,j],g) for g in perms])
        mode.append(dict(rank=j+1,eigenvalue=float(vals[j]),eta2=float(obs),
                         p=float((1+(null>=obs).sum())/1000),within_mp=bool(r.lambda_minus<=vals[j]<=r.lambda_plus)))
        for g in np.unique(labels):
            ix=labels==g
            sector_rows.append(dict(rank=j+1,industry=g,n=int(ix.sum()),
                mean_loading=float(np.sqrt(100)*V[ix,j].mean()),
                energy_fraction=float(np.sum(V[ix,j]**2)),
                energy_enrichment=float(np.mean(V[ix,j]**2)*100)))
    mode=pd.DataFrame(mode);mode['q']=bh(mode.p.to_numpy())
    mode.to_csv(OUT/'all_mode_information.csv',index=False)
    sectors=pd.DataFrame(sector_rows);sectors.to_csv(OUT/'all_mode_industry_loadings.csv',index=False)
    # Independent chronological halves, no labels used in mode matching.
    A=ra.rmt_decomposition(x[:len(x)//2]);B=ra.rmt_decomposition(x[len(x)//2:])
    va=A.eigenvectors[:,::-1];vb=B.eigenvectors[:,::-1]
    overlap=np.abs(va.T@vb)
    split=[]
    for j in range(25):
        split.append(dict(train_rank=j+1,train_eigenvalue=float(A.eigenvalues[-1-j]),
            test_variance=float(va[:,j]@B.correlation@va[:,j]),
            best_test_rank=int(overlap[j].argmax()+1),best_overlap=float(overlap[j].max())))
    pd.DataFrame(split).to_csv(OUT/'chronological_mode_reliability.csv',index=False)
    pack['modes']=mode.to_dict('records');pack['mode_sectors']=sectors.query('rank <= 10').to_dict('records')
    pack['split']=split
    # Course convention u=sqrt(N)*v; expected Gaussian N(0,1) only under random eigenvectors.
    from revision_plot_modes import main as plot_industry_bars
    plot_industry_bars()
    fig,axes=plt.subplots(1,3,figsize=(13,3.7));grid=np.linspace(-4,4,150)
    for ax,j in zip(axes,[0,1,29]):
        ax.hist(np.sqrt(100)*V[:,j],bins=14,density=True,color='#8CBFC2',alpha=.8)
        ax.plot(grid,norm.pdf(grid),'--',color='#17354A',label='N(0,1)随机向量基准')
        ax.set(title=f'模式 {j+1}，λ={vals[j]:.3f}',xlabel='u = √N v',ylabel='成分概率密度')
    axes[-1].legend(fontsize=9);fig.tight_layout();save(fig,'04_component_distributions')
    chosen=sectors.query('rank <= 10').pivot(index='industry',columns='rank',values='mean_loading')
    fig,ax=plt.subplots(figsize=(12,7))
    im=ax.imshow(chosen.T,cmap='RdBu_r',vmin=-1.5,vmax=1.5,aspect='auto')
    ax.set_xticks(np.arange(len(chosen)),chosen.index,rotation=55,ha='right',fontsize=10)
    ax.set_yticks(np.arange(10),[f'模式 {j}' for j in range(1,11)])
    ax.set_xlabel('一级行业（当前标签）');fig.colorbar(im,ax=ax,label='行业平均 u = √N v',shrink=.7)
    fig.tight_layout();save(fig,'05_mode_industry_heatmap')
    # L2 centroids from group-mode covariance, fixed common-date Gram matrix is PSD.
    rr=ra.rmt_decomposition(all_returns.to_numpy())
    gm=ra.correlation_components(rr)['group']
    full_meta=pd.read_csv(ROOT/'rmt_tables/14_stock_industry_mapping.csv').set_index('ticker').loc[all_returns.columns]
    groups=full_meta.groupby(['industry_l1','industry_l2'],sort=True)
    cats=[];weights=[]
    for (a,b),sub in groups:
        w=np.zeros(len(all_returns.columns));w[all_returns.columns.get_indexer(sub.index)]=1/len(sub)
        cats.append(dict(l1=a,l2=b,n=len(sub)));weights.append(w)
    W=np.asarray(weights);cg=ra.normalize_to_correlation(W@gm@W.T)
    tree=linkage(squareform(np.sqrt(np.maximum(0,2*(1-cg))),checks=False),method='average',optimal_ordering=True)
    fig,ax=plt.subplots(figsize=(13,23))
    dendrogram(tree,orientation='left',labels=[f"{a['l1']} / {a['l2'].replace('Ⅱ','')} ({a['n']})" for a in cats],leaf_font_size=10,ax=ax,color_threshold=0,above_threshold_color='#17354A')
    ax.set_xlabel('组模式距离 √[2(1−C)]');ax.set_title('收益数据聚类：二级行业组合的关联树（78叶）')
    fig.tight_layout();save(fig,'06_second_level_correlation_tree')
    pack['industry_hierarchy']=cats
    # Time-scale and detrending experiments. Matched shuffling includes filter effects.
    specifications=[('daily',1)]+[(m,h) for m in ['overlap','nonoverlap','return_detrend','profile_dma'] for h in [5,21,63]]
    matrices={};scale=[]
    shifts=[]
    for method,h in specifications:
        a=transform(x,method,h);gram=method=='profile_dma';C=corr(a,gram)
        eig,vec=np.linalg.eigh(C);u1=ra.orient_vector(vec[:,-1]);n=len(C);t=len(a)
        mp=(1+np.sqrt(n/t))**2
        nullmax=[];nullthird=[]
        for b in range(100):
            shuffled=np.column_stack([RNG.permutation(x[:,i]) for i in range(n)])
            cn=corr(transform(shuffled,method,h),gram)
            e=np.linalg.eigvalsh(cn);nullmax.append(e[-1]);nullthird.append(e[-3])
        scale.append(dict(method=method,scale=h,T=t,rank=int(np.linalg.matrix_rank(C,tol=1e-9)),
            market_fraction=float(eig[-1]/100),avg_correlation=float(C[np.triu_indices(n,1)].mean()),
            naive_mp_upper=float(mp),null_max_mean=float(np.mean(nullmax)),null_max_q95=float(np.quantile(nullmax,.95)),
            lambda1=float(eig[-1]),lambda3=float(eig[-3]),null_third_q95=float(np.quantile(nullthird,.95)),
            market_vector_overlap=float(abs(u1@V[:,0])),
            eigen_min=float(eig[0])))
        matrices[f'{method}_{h}']=C
        print('scale',method,h,flush=True)
    frame=pd.DataFrame(scale);frame.to_csv(OUT/'scale_dma_comparison.csv',index=False)
    np.savez_compressed(OUT/'scale_matrices.npz',**matrices)
    pack['scales']=scale
    # Market comparison with official CSI300 downloaded separately if accessible.
    pc=z@V[:,0];proxy=z.mean(axis=1)
    pack['market']={'pc_proxy_correlation':float(np.corrcoef(pc,proxy)[0,1])}
    try:
        import yfinance as yf
        d=yf.download('000300.SS',start='2018-12-28',end='2026-09-29',auto_adjust=False,progress=False,threads=False)
        price=d['Close'].squeeze();idx=np.log(price/price.shift()).dropna();common=returns.index.intersection(idx.index)
        if len(common)<1000:raise ValueError('Not enough matched dates')
        f=pd.DataFrame({'pc':pc,'proxy':proxy},index=returns.index).loc[common];f['CSI300_log_return']=idx.loc[common]
        f.to_csv(OUT/'official_index_comparison.csv')
        pack['market'].update({'official_days':len(f),'official_pc_correlation':float(f.pc.corr(f.CSI300_log_return)),
                               'official_proxy_correlation':float(f.proxy.corr(f.CSI300_log_return))})
    except Exception as exc:pack['market']['official_unavailable']=str(exc)
    pack['references']=['https://finance.martinsewell.com/stylized-facts/distribution/Cont2001.pdf',
      'https://arxiv.org/abs/1102.4076','https://arxiv.org/abs/1311.0657']
    (OUT/'revision_pack.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2))
    print(json.dumps({'stylized':{k:v for k,v in pack['stylized'].items() if k!='acf'},
                      'market':pack['market'],'within_mp_q05':int(mode.query('within_mp and q<.05').shape[0])},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
