"""Industry-coloured bars for eigenvector loadings, same ordering in every panel."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rmt_analysis as ra
import revision_analysis as rev

def main():
    cols=pd.read_csv(rev.ROOT/'rmt_tables/05_seeded_stock_selection.csv').query('in_fixed_100').ticker.tolist()
    r=ra.load_returns(rev.ROOT/'tables/07_log_return_matrix.csv')[cols]
    meta=pd.read_csv(rev.ROOT/'rmt_tables/14_stock_industry_mapping.csv').set_index('ticker').loc[cols]
    labels=meta.industry_l1.to_numpy();categories=sorted(set(labels))
    palette=list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('tab20b').colors)
    colors={g:palette[i] for i,g in enumerate(categories)}
    result=ra.rmt_decomposition(r.to_numpy());values=result.eigenvalues[::-1];V=result.eigenvectors[:,::-1]
    for j in range(100):V[:,j]=ra.orient_vector(V[:,j])
    order=np.lexsort((np.asarray(cols),labels));u=10*V[order];ordered=labels[order]
    boundaries=np.flatnonzero(ordered[1:]!=ordered[:-1])+.5
    fig,axes=plt.subplots(2,3,figsize=(15.5,9))
    for ax,j in zip(axes.flat,[0,1,2,3,4,10]):
        ax.bar(np.arange(1,101),u[:,j],color=[colors[g] for g in ordered],width=.92,linewidth=0)
        for b in boundaries:ax.axvline(b+1,color='#D8DFE1',lw=.5,zorder=0)
        ax.axhline(0,color='#65757E',lw=.8)
        ax.set(title=f'模式 {j+1}，λ={values[j]:.3f}',ylabel='u = √N v',xlim=(.2,100.8),xlabel='股票序号（按行业连续排列）')
        ax.tick_params(labelsize=10)
    handles=[Patch(facecolor=colors[g],label=f'{g} ({np.sum(labels==g)})') for g in categories]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=7,fontsize=9.5,frameon=False,
               handlelength=1.1,columnspacing=1.3)
    fig.subplots_adjust(left=.06,right=.99,top=.95,bottom=.21,hspace=.42,wspace=.25)
    rev.save(fig,'03_mode_loadings')
    for tag,which in [('top',[0,1,2]),('bottom',[3,4,10])]:
        fig,axes=plt.subplots(1,3,figsize=(15.5,6.2))
        for ax,j in zip(axes,which):
            ax.bar(np.arange(1,101),u[:,j],color=[colors[g] for g in ordered],width=.94,linewidth=0)
            for b in boundaries:ax.axvline(b+1,color='#D8DFE1',lw=.5,zorder=0)
            ax.axhline(0,color='#65757E',lw=.8)
            ax.set_title(f'模式 {j+1}，λ={values[j]:.3f}',fontsize=18)
            ax.set_xlabel('股票序号（按行业排列）',fontsize=15)
            ax.set_ylabel('u = √N v',fontsize=15)
            ax.set_xlim(.2,100.8);ax.tick_params(labelsize=12)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=7,fontsize=12,
                   frameon=False,handlelength=1,columnspacing=1)
        fig.subplots_adjust(left=.055,right=.995,top=.92,bottom=.32,wspace=.30)
        rev.save(fig,f'03_mode_bars_{tag}')
    # A slide-size heatmap excludes singleton industries and labels each sample size.
    sector=pd.read_csv(rev.OUT/'all_mode_industry_loadings.csv')
    selected=sector.query('rank<=10 and n>=2')
    mat=selected.pivot(index='industry',columns='rank',values='mean_loading')
    ns=selected.groupby('industry').n.first()
    fig,ax=plt.subplots(figsize=(16,6.3))
    limit=float(max(2.5,np.abs(mat.to_numpy()).max()))
    im=ax.imshow(mat.T,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(np.arange(len(mat)),[f'{g}({ns[g]})' for g in mat.index],rotation=50,ha='right',fontsize=13)
    ax.set_yticks(np.arange(10),[f'模式 {i}' for i in range(1,11)],fontsize=13)
    fig.colorbar(im,ax=ax,label='行业平均 u=√N v',shrink=.85)
    fig.tight_layout();rev.save(fig,'05_mode_industry_heatmap_slides')
    import matplotlib.colors as mc
    pd.DataFrame({'industry':categories,'hex_color':[mc.to_hex(colors[g]) for g in categories],
                  'stock_count':[int(np.sum(labels==g)) for g in categories]}).to_csv(rev.OUT/'industry_bar_palette.csv',index=False)

if __name__=='__main__':main()
