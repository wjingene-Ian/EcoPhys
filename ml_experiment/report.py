from __future__ import annotations
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image
from .data import load, sha, dump, factor_task, HORIZON, ANNUAL
from .models import ALL, CLASSICAL, NEURAL
from .run import simulate, covariance

LABELS = {'equal_weight':'等权', 'raw_gmv':'原始GMV', 'shrink20_gmv':'20%收缩GMV',
          'rmt_clipped_gmv':'同口径RMT clipped GMV', 'legacy_rmt_clipped_gmv':'原有RMT曲线（共同区间）',
          'ridge':'Ridge', 'elastic_net':'Elastic Net', 'lightgbm':'LightGBM', 'xgboost':'XGBoost',
          'rnn':'RNN', 'lstm':'LSTM', 'gru':'GRU', 'tcn':'TCN', 'transformer':'分块Transformer',
          'gnn_gru':'GNN＋GRU', 'direct_weights':'直接权重学习', 'hrp':'HRP（权重上限投影）',
          'kl_static':'KL静态因子风险', 'kl_ewma':'KL历史均方风险', 'kl_gru':'KL＋GRU风险', 'kl_tcn':'KL＋TCN风险'}
FAMILIES = [('Allocation baselines', ['equal_weight','raw_gmv','shrink20_gmv']),
            ('Tabular return forecasts', ['ridge','elastic_net','lightgbm','xgboost']),
            ('Recurrent return forecasts', ['rnn','lstm','gru']),
            ('Convolution, attention and graph', ['tcn','transformer','gnn_gru']),
            ('Risk forecasts and direct allocation', ['hrp','direct_weights','kl_static','kl_ewma','kl_gru','kl_tcn'])]


def metrics(values):
    values = np.asarray(values); wealth = np.r_[1., np.cumprod(1+values)]
    dd = wealth/np.maximum.accumulate(wealth)-1
    cagr = wealth[-1]**(ANNUAL/len(values))-1
    return {'days':len(values), 'terminal_wealth':wealth[-1], 'annual_return':cagr,
            'annual_volatility':values.std(ddof=1)*np.sqrt(ANNUAL),
            'sharpe_rf0':values.mean()/values.std(ddof=1)*np.sqrt(ANNUAL),
            'max_drawdown':dd.min(), 'calmar':cagr/max(abs(dd.min()),1e-12)}


def mdtable(frame):
    def cell(v):
        if isinstance(v,(float,np.floating)):
            return '—' if pd.isna(v) else f'{v:.4f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def bootstrap_comparison(daily, repetitions=2000):
    base = daily.rmt_clipped_gmv.to_numpy()
    rng = np.random.default_rng(20261110)
    boot = {}
    for block in [21,63]:
        sampled = rng.integers(0,len(base),size=(repetitions,int(np.ceil(len(base)/block))))
        ix = ((sampled[:,:,None]+np.arange(block)) % len(base)).reshape(repetitions,-1)[:,:len(base)]
        a = base[ix]; base_sharpe = a.mean(axis=1)/a.std(axis=1,ddof=1)*np.sqrt(ANNUAL)
        for name in daily:
            if name == 'rmt_clipped_gmv':
                continue
            b = daily[name].to_numpy()[ix]
            delta = b.mean(axis=1)/b.std(axis=1,ddof=1)*np.sqrt(ANNUAL)-base_sharpe
            obs = metrics(daily[name])['sharpe_rf0']-metrics(base)['sharpe_rf0']
            p = (1+np.sum(abs(delta-obs)>=abs(obs)))/(repetitions+1)
            boot[(name,block)] = {'model':name,'block_days':block,'delta_sharpe':obs,
                'ci025':np.quantile(delta,.025),'ci975':np.quantile(delta,.975),
                'approx_centered_bootstrap_p_two_sided':p}
    frame = pd.DataFrame(boot.values())
    for block, sample in frame.groupby('block_days'):
        order = sample.sort_values('approx_centered_bootstrap_p_two_sided').index
        raw = frame.loc[order,'approx_centered_bootstrap_p_two_sided'].to_numpy()
        frame.loc[order,'holm_adjusted_p'] = np.minimum(1,np.maximum.accumulate(raw*(len(raw)-np.arange(len(raw)))))
    return frame


def risk_evaluation(root,out,data):
    manifest = pd.read_csv(root/'outputs/hs300_long100/full/modes/tables/risk_portfolio_manifest.csv')
    w = manifest.pivot(index='ticker',columns='portfolio_id',values='weight').reindex(data['tickers']).fillna(0).to_numpy()
    assert w.shape == (100,201) and np.allclose(w.sum(axis=0),1)
    x=data['returns']; starts=data['starts']; decisions=data['decisions'][:,0]
    dates=pd.to_datetime(data['dates']); rows=[]
    names=['raw_gmv','shrink20_gmv','rmt_clipped_gmv','kl_static','kl_ewma','kl_gru','kl_tcn']
    risk_cov={name:{} for name in names}
    for name in names[:3]:
        for period,d in enumerate(decisions):
            risk_cov[name][period]=covariance(x,int(d),name)
    for name in names[3:]:
        audit=json.loads((out/'models'/name/'training_audit.json').read_text())
        for year in np.unique(dates[starts].year):
            ids=np.flatnonzero(dates[starts].year==year)
            cutoff=int(data['decisions'][ids].min())
            _,_,basis,residual,_=factor_task(x,cutoff)
            files=sorted((out/'models'/name).glob(f'{year}_seed*_factor_basis.npz'))
            vars=[]
            for file in files:
                saved=np.load(file)
                assert np.allclose(saved['basis'],basis,atol=1e-10)
                vars.append(saved['predicted_variance'].reshape(len(ids),2,10)[:,0])
            variance=np.mean(vars,axis=0)
            for j,period in enumerate(ids):
                d=decisions[period]
                dvar=np.maximum(residual[d-62:d+1].var(axis=0,ddof=1),1e-10)
                risk_cov[name][period]=(basis*variance[j])@basis.T+np.diag(dvar)
    for period,(start,d) in enumerate(zip(starts,decisions)):
        if start+HORIZON>len(x):
            continue
        test=x[start:start+HORIZON]-x[d-503:d+1].mean(axis=0)
        realized=np.mean((test@w)**2,axis=0)
        for name in names:
            cov=risk_cov[name][period]
            forecast=np.sum(w*(cov@w),axis=0)
            ratio=realized/forecast
            chol=np.linalg.cholesky(cov)
            whitened=np.linalg.solve(chol,test.T)
            score=(2*np.log(np.diag(chol)).sum()+np.mean(np.sum(whitened**2,axis=0)))/len(cov)
            rows.append({'model':name,'forecast_date':str(dates[d].date()),'test_start':str(dates[start].date()),
                'test_days':HORIZON,'mae_pp':np.mean(abs(np.sqrt(forecast)-np.sqrt(realized)))*100,
                'bias_pp':np.mean(np.sqrt(forecast)-np.sqrt(realized))*100,
                'qlike':np.mean(ratio-np.log(ratio)-1),'median_variance_ratio':np.median(ratio),
                'gaussian_covariance_score':score,'covariance_min_eigenvalue':np.linalg.eigvalsh(cov)[0]})
    folds=pd.DataFrame(rows);folds.to_csv(out/'risk_prediction_folds.csv',index=False)
    summary=folds.groupby('model',sort=False).agg(folds=('mae_pp','size'),mae_pp=('mae_pp','mean'),
        bias_pp=('bias_pp','mean'),qlike=('qlike','mean'),gaussian_score=('gaussian_covariance_score','mean'),
        median_variance_ratio=('median_variance_ratio','median')).reset_index()
    summary.to_csv(out/'risk_prediction_summary.csv',index=False)
    return summary


def figures(out,daily,summary,seeds,lag,annual,risk,ic):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    for font in ['/System/Library/Fonts/Hiragino Sans GB.ttc']:
        if Path(font).exists():
            font_manager.fontManager.addfont(font)
            plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout();fig.savefig(figdir/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(len(FAMILIES),1,figsize=(15,21),sharex=True)
    for ax,(title,names) in zip(axes,FAMILIES):
        for name in names:
            ax.plot(daily.index,(1+daily[name]).cumprod(),label=LABELS[name],lw=1.1)
        ax.plot(daily.index,(1+daily.rmt_clipped_gmv).cumprod(),color='black',lw=2.1,label='同口径 RMT clipped GMV')
        ax.plot(daily.index,(1+daily.legacy_rmt_clipped_gmv).cumprod(),color='grey',ls='--',lw=1.2,label='原有 RMT 曲线')
        ax.set(title=title,ylabel='净值（共同起点=1；对数轴）',yscale='log')
        from matplotlib.ticker import FuncFormatter, NullFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.legend(ncol=3,fontsize=9);ax.grid(alpha=.2)
    axes[-1].set_xlabel('交易日期')
    save(fig,'01_all_models_equity_vs_rmt')
    fig,axes=plt.subplots(1,3,figsize=(17,10),sharey=True)
    names=summary.model.tolist();positions=np.arange(len(names))
    for ax,column,title,scale in zip(axes,['terminal_wealth','sharpe_rf0','max_drawdown'],['期末净值','Sharpe（无风险利率0）','最大回撤（%）'],[1,1,100]):
        ax.barh(positions,summary[column]*scale,color=['#C96645' if n=='rmt_clipped_gmv' else '#2d668d' for n in names])
        ax.set_yticks(positions,[LABELS[n] for n in names]);ax.set_title(title);ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();save(fig,'02_all_models_metrics')
    fig,axes=plt.subplots(len(FAMILIES),1,figsize=(15,18),sharex=True)
    for ax,(title,names) in zip(axes,FAMILIES):
        for name in names:
            wealth=np.r_[1.,(1+daily[name]).cumprod().to_numpy()]
            dd=wealth/np.maximum.accumulate(wealth)-1
            ax.plot(daily.index,dd[1:]*100,label=LABELS[name],lw=1)
        w=np.r_[1.,(1+daily.rmt_clipped_gmv).cumprod().to_numpy()]
        ax.plot(daily.index,(w/np.maximum.accumulate(w)-1)[1:]*100,color='black',lw=2,label='RMT clipped GMV')
        ax.set(title=title,ylabel='回撤（%）');ax.legend(ncol=3,fontsize=9)
    save(fig,'03_all_models_drawdowns')
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    stochastic=[n for n in names if len(seeds[seeds.model==n])==3]
    stochastic=seeds.groupby('model').filter(lambda s:len(s)==3).model.drop_duplicates().tolist()
    for j,column in enumerate(['sharpe_rf0','terminal_wealth']):
        for i,name in enumerate(stochastic):
            vals=seeds[seeds.model==name][column]
            axes[j].scatter([i]*3,vals,s=35,color='#2d668d')
            axes[j].plot([i,i],[vals.min(),vals.max()],color='#2d668d')
            axes[j].scatter(i,summary.set_index('model').loc[name,column],marker='D',color='#C96645',s=45)
        axes[j].set_xticks(range(len(stochastic)),[LABELS[n] for n in stochastic],rotation=45,ha='right')
        axes[j].set_title('三个训练种子（蓝）与平均权重组合（橙）：'+column)
        axes[j].axhline(summary.set_index('model').loc['rmt_clipped_gmv',column],color='black',ls='--')
    save(fig,'04_random_seed_variation')
    fig,axes=plt.subplots(1,2,figsize=(15,10),sharey=True)
    for j,column in enumerate(['sharpe_rf0','terminal_wealth']):
        pivot=lag.pivot(index='model',columns='signal_lag_days',values=column).reindex(ALL)
        y=np.arange(len(pivot))
        axes[j].barh(y-.18,pivot[1],height=.35,label='延迟1日（主实验）')
        axes[j].barh(y+.18,pivot[2],height=.35,label='延迟2日')
        axes[j].set_yticks(y,[LABELS[n] for n in pivot.index]);axes[j].set_title(column);axes[j].legend()
    axes[0].invert_yaxis();save(fig,'05_execution_delay_sensitivity')
    grid=annual.pivot(index='model',columns='year',values='total_return').reindex(ALL)
    fig,ax=plt.subplots(figsize=(16,9));im=ax.imshow(grid*100,cmap='RdBu_r',vmin=-50,vmax=50,aspect='auto')
    ax.set_yticks(range(len(grid)),[LABELS[n] for n in grid.index]);ax.set_xticks(range(len(grid.columns)),grid.columns,rotation=45)
    fig.colorbar(im,ax=ax,label='年度收益（%；色标截断±50）');ax.set_title('所有模型年度收益：首尾年份不完整')
    save(fig,'06_annual_returns_heatmap')
    fig,axes=plt.subplots(1,3,figsize=(15,6))
    for ax,column,title in zip(axes,['mae_pp','qlike','gaussian_score'],['日波动预测MAE（百分点）','方差QLIKE','Gaussian协方差评分']):
        ax.bar(range(len(risk)),risk[column],color='#2d668d')
        ax.set_xticks(range(len(risk)),[LABELS[n] for n in risk.model],rotation=45,ha='right')
        ax.set_title(title+'；越小越好')
    save(fig,'07_out_of_sample_risk_prediction')
    fig,ax=plt.subplots(figsize=(12,5))
    if len(ic):
        sample=ic[ic.seed=='ensemble'].groupby('model').ic.mean().reindex([n for n in ALL if n in ic.model.unique()])
        ax.bar(range(len(sample)),sample,color='#2d668d');ax.set_xticks(range(len(sample)),[LABELS[n] for n in sample.index],rotation=40,ha='right')
        ax.axhline(0,color='grey');ax.set(ylabel='平均横截面Spearman IC',title='未来21日股票排名预测；未做独立性显著性假设')
    save(fig,'08_prediction_rank_information')
    fig,axes=plt.subplots(2,1,figsize=(15,9),sharex=True,gridspec_kw={'height_ratios':[1.4,1]})
    selected=['legacy_rmt_clipped_gmv','rmt_clipped_gmv','lightgbm','elastic_net','rnn','lstm','gru','transformer']
    for name in selected:
        color='black' if name=='rmt_clipped_gmv' else 'grey' if name=='legacy_rmt_clipped_gmv' else None
        width=2.4 if name=='rmt_clipped_gmv' else 1.2
        style='--' if name=='legacy_rmt_clipped_gmv' else '-'
        wealth=(1+daily[name]).cumprod()
        line=axes[0].plot(daily.index,wealth,label=LABELS[name],lw=width,ls=style,color=color)[0]
        axes[1].plot(daily.index,(wealth/wealth.cummax().clip(lower=1)-1)*100,lw=width,ls=style,color=line.get_color())
    axes[0].set(ylabel='净值（共同起点=1）',title='主要预测模型与原有/同口径RMT基线：完整4069个交易日')
    axes[1].set(ylabel='回撤（%）',xlabel='交易日期');axes[0].legend(ncol=4,fontsize=9)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'09_key_equity_and_drawdown')


def report(root,out):
    missing=[name for name in ALL if not (out/'models'/name/'targets.npz').exists()]
    if missing:
        raise ValueError(f'以下方法尚未完成，不能生成全部结果报告：{missing}')
    data=load(out);dates=pd.to_datetime(data['dates']);evaluation_dates=dates[data['starts'][0]:]
    daily=pd.DataFrame(index=evaluation_dates);seed_rows=[];lag_rows=[];rebalance_rows=[];weight_rows=[];ic_rows=[];cost_rows=[]
    checks={};train_records=[]
    for name in ALL:
        saved=np.load(out/'models'/name/'targets.npz')
        weights=saved['weights'];signals=saved['signals'];seeds=saved['seeds']
        if not np.isfinite(weights).all() or weights.min() < -1e-7 or weights.max() > .05+1e-7 or not np.allclose(weights.sum(axis=-1),1,atol=1e-6):
            raise AssertionError(f'Invalid weights: {name}')
        for lag_index in range(2):
            ensemble=weights[:,lag_index].mean(axis=0)
            values,rebalances=simulate(data,ensemble)
            row={'model':name,'signal_lag_days':lag_index+1,**metrics(values),
                 'average_turnover':rebalances.turnover.mean(),
                 'annualized_turnover':rebalances.turnover.sum()*ANNUAL/len(values),
                 'mean_effective_stocks':rebalances.effective_stocks.mean()}
            lag_rows.append(row)
            if lag_index==0:
                daily[name]=values
                rebalances.insert(0,'model',name);rebalance_rows.append(rebalances)
                for cost in [0.,.001,.0025,.005]:
                    cost_values,_=simulate(data,ensemble,cost)
                    cost_rows.append({'model':name,'cost_bps_per_half_L1_turnover':cost*10000,**metrics(cost_values)})
                for seed_index,seed in enumerate(seeds):
                    values,_=simulate(data,weights[seed_index,0])
                    seed_rows.append({'model':name,'seed':int(seed),**metrics(values)})
            table=pd.DataFrame(ensemble,columns=data['tickers'])
            table.insert(0,'rebalance_date',dates[data['starts']].strftime('%Y-%m-%d'))
            table.insert(0,'signal_lag_days',lag_index+1);table.insert(0,'model',name);weight_rows.append(table)
        if name in CLASSICAL or name in [n for n in NEURAL if n!='direct_weights']:
            for seed_index,seed in [(i,str(seed)) for i,seed in enumerate(seeds)]+[(None,'ensemble')]:
                prediction=signals[:,0].mean(axis=0) if seed_index is None else signals[seed_index,0]
                for period,start in enumerate(data['starts']):
                    d=data['decisions'][period,0]
                    if data['valid_targets'][d]:
                        pred=prediction[period]; target=data['targets'][d]
                        ic=float(stats.spearmanr(pred,target).statistic) if np.std(pred)>1e-12 else np.nan
                        ic_rows.append({'model':name,'seed':seed,'date':str(dates[start].date()),'ic':ic})
        audit=json.loads((out/'models'/name/'training_audit.json').read_text());train_records.extend(audit)
    legacy_path=root/'outputs/hs300_long100/full/rmt/tables/09_portfolio_daily_returns.csv'
    legacy=pd.read_csv(legacy_path,index_col=0,parse_dates=True).rmt_clipped_gmv.reindex(evaluation_dates)
    assert not legacy.isna().any()
    daily['legacy_rmt_clipped_gmv']=legacy
    daily.index.name='Date';daily.to_csv(out/'daily_net_returns.csv')
    nav=(1+daily).cumprod();nav.to_csv(out/'normalized_equity.csv')
    drawdown=nav/nav.cummax().clip(lower=1)-1;drawdown.to_csv(out/'drawdowns.csv')
    lag=pd.DataFrame(lag_rows);seeds=pd.DataFrame(seed_rows);ic=pd.DataFrame(ic_rows)
    summary=lag[lag.signal_lag_days==1].copy()
    summary=pd.concat([summary,pd.DataFrame([{'model':'legacy_rmt_clipped_gmv','signal_lag_days':0,**metrics(legacy)}])],ignore_index=True)
    summary['name_zh']=summary.model.map(LABELS)
    summary['delta_sharpe_vs_matched_rmt']=summary.sharpe_rf0-summary.set_index('model').loc['rmt_clipped_gmv','sharpe_rf0']
    summary.to_csv(out/'performance_summary.csv',index=False)
    seeds.to_csv(out/'seed_performance.csv',index=False);lag.to_csv(out/'execution_delay_sensitivity.csv',index=False)
    pd.concat(rebalance_rows).to_csv(out/'rebalance_turnover.csv',index=False)
    pd.concat(weight_rows).to_csv(out/'ensemble_target_weights.csv.gz',index=False,compression='gzip')
    ic.to_csv(out/'ranking_information_coefficients.csv',index=False)
    pd.DataFrame(cost_rows).to_csv(out/'transaction_cost_sensitivity.csv',index=False)
    annual=[];stress=[]
    for year,piece in daily.groupby(daily.index.year):
        for name in daily:
            annual.append({'model':name,'year':year,'total_return':np.prod(1+piece[name])-1,**metrics(piece[name])})
    annual=pd.DataFrame(annual);annual.to_csv(out/'annual_performance.csv',index=False)
    for label,start,end in [('2015年中','2015-06-01','2015-09-30'),('2020一季度','2020-01-01','2020-03-31'),
                            ('2022全年','2022-01-01','2022-12-31'),('2024秋季','2024-09-01','2024-10-31'),
                            ('2025四月','2025-04-01','2025-04-30')]:
        for name in daily:
            piece=daily.loc[start:end,name]
            stress.append({'period':label,'model':name,**metrics(piece)})
    pd.DataFrame(stress).to_csv(out/'stress_period_performance.csv',index=False)
    print('Evaluating uncertainty with 2000 paired circular-block resamples...',flush=True)
    bootstrap=bootstrap_comparison(daily);bootstrap.to_csv(out/'paired_bootstrap_vs_rmt.csv',index=False)
    print('Evaluating covariance forecasts on the same 201 fixed portfolios...',flush=True)
    risk=risk_evaluation(root,out,data)
    figures(out,daily,summary,seeds,lag,annual,risk,ic)
    print('Checking time boundaries, weight constraints, source hashes and all figures...',flush=True)
    expected_sources={f'ml_experiment/{name}':sha(root/'ml_experiment'/name) for name in ['data.py','models.py','run.py']}
    checks['all20_methods_complete']=len(ALL)==20 and all((out/'models'/n/'metadata.json').exists() for n in ALL)
    checks['all_model_code_hashes_match']=all(json.loads((out/'models'/n/'metadata.json').read_text())['code_sha256']==expected_sources for n in ALL)
    checks['all_training_labels_known_at_fit_time']=all(pd.Timestamp(r['max_final_label_date'])<=pd.Timestamp(r['fit_information_cutoff']) for r in train_records)
    checks['purged_training_validation_labels']=all(pd.Timestamp(r['train_max_label_date'])<pd.Timestamp(r['validation_first_feature_date']) for r in train_records)
    checks['all_model_fit_cutoffs_precede_forecast_information']=all(pd.Timestamp(r['fit_information_cutoff'])<=min(pd.to_datetime(r['forecast_information_dates'])) for r in train_records)
    checks['all_forecast_information_matches_exact_annual_decision_dates']=all(
        r['forecast_information_dates']==[str(dates[d].date()) for d in data['decisions'][dates[data['starts']].year==r['year']].ravel()]
        for r in train_records)
    checks['exact_execution_lag1_and_lag2']=bool(np.array_equal(data['starts'][:,None]-data['decisions']-1,np.tile([1,2],(len(data['starts']),1))))
    checks['all_target_weights_finite_sum1_and_capped']=True
    checks['all_daily_returns_finite_and_greater_than_minus1']=bool(np.isfinite(daily.to_numpy()).all() and (daily.to_numpy()>-1).all())
    checks['legacy_baseline_daily_returns_exactly_preserved']=bool(np.array_equal(daily.legacy_rmt_clipped_gmv.to_numpy(),legacy.to_numpy()))
    audit=json.loads((out/'input_audit.json').read_text())
    checks['clean_return_source_hash_unchanged']=sha(root/'outputs/hs300_long100/full/log_returns.csv')==audit['returns_sha256']
    checks['old_baseline_source_hash_unchanged']=sha(legacy_path)==audit['legacy_baseline_sha256']
    checks['all_risk_forecasts_positive_definite']=bool((pd.read_csv(out/'risk_prediction_folds.csv').covariance_min_eigenvalue>0).all())
    checks['all_methods_share_identical_evaluation_dates']=bool(len(daily)==len(data['returns'])-data['starts'][0] and not daily.isna().any().any())
    checks['all_stochastic_models_use_three_seeds']=all(len(np.load(out/'models'/n/'targets.npz')['seeds'])==3 for n in [
        'lightgbm','xgboost','rnn','lstm','gru','tcn','transformer','gnn_gru','direct_weights','kl_gru','kl_tcn'])
    for path in (out/'figures').glob('*.png'):
        with Image.open(path) as image:
            image.verify()
    checks['all_png_readable']=True
    from .checks import implementation_checks
    checks.update(implementation_checks(data,out))
    if not all(checks.values()):
        raise AssertionError([k for k,v in checks.items() if not v])
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','torch','lightgbm','xgboost','matplotlib']}
    dump(out/'validation.json',{'checks':checks,'training_records':len(train_records),'versions':versions,
         'python':sys.version,'report_sha256':sha(__file__),'seeds':[17,42,2026], 'bootstrap_repetitions':2000})
    write_report(root,out,daily,summary,seeds,lag,bootstrap,risk,ic,checks,train_records)
    package(root,out,summary,versions)


def write_report(root,out,daily,summary,seeds,lag,bootstrap,risk,ic,checks,train_records):
    primary=summary[summary.model!='legacy_rmt_clipped_gmv']
    ordered=primary.sort_values('terminal_wealth',ascending=False)
    best_nav=ordered.iloc[0];best_sharpe=primary.loc[primary.sharpe_rf0.idxmax()]
    baseline=summary.set_index('model').loc['rmt_clipped_gmv'];old=summary.set_index('model').loc['legacy_rmt_clipped_gmv']
    delay=lag.set_index(['model','signal_lag_days'])
    transformer2=delay.loc[('transformer',2)]
    lightgbm2=delay.loc[('lightgbm',2)]
    lstm_seeds=seeds[seeds.model=='lstm']
    risk_by_name=risk.set_index('model')
    show=summary[['name_zh','terminal_wealth','annual_return','annual_volatility','sharpe_rf0','max_drawdown','average_turnover']].copy()
    show[['annual_return','annual_volatility','max_drawdown']]*=100
    show.columns=['方法','期末净值','年化收益%','年化波动%','Sharpe','最大回撤%','平均调仓换手']
    show.to_csv(out/'主要结果中文.csv',index=False)
    interval=bootstrap[(bootstrap.block_days==21)&bootstrap.model.isin(['lightgbm','elastic_net','transformer','ridge','gru','lstm','kl_gru','kl_tcn'])].copy()
    interval['model']=interval.model.map(LABELS)
    count_significant=int(((bootstrap.block_days==21)&(bootstrap.holm_adjusted_p<.05)).sum())
    # A readable schema for all model hyperparameters and annual time boundaries.
    training_summary=pd.DataFrame([{k:r.get(k) for k in ['model','seed','year','fit_information_cutoff','max_final_label_date',
        'train_anchors','validation_anchors','final_fit_anchors','selected_epochs','selected_option','elapsed_seconds']} for r in train_records])
    training_summary.to_csv(out/'training_cutoffs_and_parameters.csv',index=False)
    content=['# 100股长历史数据：机器学习组合与RMT clipped GMV完整比较','',
        '## 1. 本次做了什么','',
        f'全部20种方法完成；共同评价区间为 **{daily.index[0].date()}—{daily.index[-1].date()}**，共 **{len(daily)}** 个日收益。每条曲线都从同一天、初始净值1开始。原输入为经过日历审计的100股、6001个日收益，未改动原始数据和既有RMT结果。',
        '方法包括：等权、原始GMV、20%收缩GMV、RMT clipped GMV、HRP、Ridge、Elastic Net、LightGBM、XGBoost、RNN、LSTM、GRU、TCN、小型分块Transformer、GNN＋GRU、直接权重学习、KL静态风险、KL历史均方风险、KL＋GRU风险、KL＋TCN风险。',
        '数据从2002年开始，机器学习需要先积累训练与验证历史，所以没有把2004年就开始的原RMT全区间净值直接与较晚启动的模型净值比较。2008年没有本次机器学习样本外结果。', '',
        '## 2. 两条RMT比较基线','',
        f'**原有RMT曲线：**直接读取之前保存的每日净收益，截取共同区间后重新连乘，期末净值 **{old.terminal_wealth:.4f}**、Sharpe **{old.sharpe_rf0:.4f}**。原文件逐字节哈希和截取后的逐日收益均已核对。它沿用原来的收盘成交理想化假设及此前持仓。',
        f'**同口径RMT clipped GMV：**按这次统一规则重算，期末净值 **{baseline.terminal_wealth:.4f}**、Sharpe **{baseline.sharpe_rf0:.4f}**。全部主要比较与区块重采样使用这条基线。',
        '统一执行规则：在d日收盘信息形成预测，留出完整一个交易日，于d+1收盘执行，从d+2的收盘到收盘收益开始持有；每21日调仓，504日风险估计窗口，股票目标权重0–5%、总和1、无现金/杠杆/做空，持仓中权重自然漂移。初始建仓不计费用，与原handoff口径一致。费用严格沿用原仓库公式：0.001×0.5×目标权重与调仓前漂移权重的L1差，并在调仓日乘法扣除；不是把10bps再乘到全部买卖金额上。年化242日，无风险利率0。额外延迟到d+2收盘的结果单独报告。', '',
        '## 3. 全部净值、Sharpe和回撤结果','',mdtable(show),'',
        f'本样本中期末净值最高的是 **{LABELS[best_nav.model]}（{best_nav.terminal_wealth:.3f}）**，Sharpe最高的是 **{LABELS[best_sharpe.model]}（{best_sharpe.sharpe_rf0:.3f}）**。这是对已完成实验的描述，不是用测试结果重新选择模型或超参数。', '',
        f'净值与回撤的取舍：LightGBM最大回撤为{abs(summary.set_index("model").loc["lightgbm","max_drawdown"]):.2%}，大于同口径RMT的{abs(baseline.max_drawdown):.2%}；Elastic Net则在提高这段样本净值的同时，将最大回撤降到{abs(summary.set_index("model").loc["elastic_net","max_drawdown"]):.2%}。这些都只是当前固定池的历史结果。','',
        '![所有模型与两条RMT基线的净值](figures/01_all_models_equity_vs_rmt.png)','',
        '![全部结果指标](figures/02_all_models_metrics.png)','',
        '## 4. 模型如何训练，哪些步骤没有使用未来信息','',
        '- 每年重新训练；每次取过去约六年成熟标签，最后约一年为时间验证集，前约五年为训练集。21日收益标签跨越验证边界的样本被剔除。选好参数/训练轮数后，再用截止当时已经成熟的标签重拟合。',
        '- 训练锚点与原策略的21日调仓网格一致，每年约55–56个训练日期、11个验证日期，最终重拟合约67个日期；每个日期包含100股。股票共享模型参数，未独立训练100个小网络。这些横截面样本相关，不能当作数千个独立市场状态。',
        '- 18个输入特征包括过去收益/动量、波动、回撤、市场暴露、相对收益、滚动KL载荷/能量/市场份额及残差收益。输入序列40日；每个时点的PCA和图只用此前252日。特征标准化只在对应训练集拟合。没有使用全样本PCA，也没有把2026行业标签回填到过去。',
        '- 所有收益预测模型学习未来21日的股票横截面排名。Ridge/Elastic Net在两个正则强度中选择；树模型在两个复杂度中选择，均只看验证误差。它们采用同一套末期值、40日均值和变化量特征。',
        '- RNN/LSTM/GRU为单层16维隐状态；TCN为16通道、4个因果膨胀卷积块；Transformer为5日一个分块、16维表示、2个注意力头、1层编码器，是受PatchTST启发的紧凑模型，不冒充原论文的完整复现。最多8轮训练、验证连续2轮未改善则停止，然后按选中轮数重新拟合。',
        '- GNN先用GRU编码每股历史，再用当时正的去市场相关关系做消息传递；直接权重模型用小型网络输出满足权重上限的组合，训练目标为均值—方差—换手惩罚，其训练目标是近似效用，最终净值按统一真实漂移规则重新计算。',
        '- 为分离收益预测和风险配置，所有收益预测器都取预测分数前30股，再用同一个20%对角收缩协方差求最小方差权重；并非把其余70股从输入或研究股票池删除。HRP基于同一收缩风险矩阵、单连接聚类和递归风险分配，最后投影到5%权重上限。',
        '- KL风险路线使用过去504日拟合10个因子；验证阶段和最终重拟合阶段分别拟合PCA，验证变换没有看到验证之后的数据。GRU/TCN预测正的未来因子均方风险，加上过去63日残差对角风险形成正定协方差，再求GMV。每年内载荷固定、风险随信息更新；同时保留不使用神经网络的KL静态和KL历史均方控制。',
        '- 11种随机模型均运行17、42、2026三个种子；主要曲线用三组目标权重的算术平均，再完整回测。没有挑选表现最好的种子。其他9个确定性方法使用一次拟合。', '',
        '## 5. 不确定性、执行延迟和交易费用','',
        '相对同口径RMT的Sharpe差异，使用共同日期的21日和63日循环块重采样各2000次，保留策略间配对关系。下表为21日块结果。区间是探索性描述；中心化bootstrap近似p值另做Holm多重比较校正，不能当作独立、未看过测试集上的确认性证据。', '',
        mdtable(interval[['model','delta_sharpe','ci025','ci975','holm_adjusted_p']]),'',
        f'21日块下，经过本轮全部比较的Holm校正后，p<0.05的差异共有 **{count_significant}** 个（包含可能显著变差的方法；正负方向看delta_sharpe）。不要仅凭最高净值或最高Sharpe宣称获得稳定超额收益。',
        '执行延迟1日/2日见 execution_delay_sensitivity.csv；成本0/10/25/50bps见 transaction_cost_sensitivity.csv。成本敏感性固定已训练的模型和目标权重，只改变回测扣费，不重新挑模型。', '',
        f'**执行时点是一个实质限制：**Transformer额外延迟一天后，期末净值降至{transformer2.terminal_wealth:.3f}、Sharpe降至{transformer2.sharpe_rf0:.3f}；LightGBM在该检验下为净值{lightgbm2.terminal_wealth:.3f}、Sharpe {lightgbm2.sharpe_rf0:.3f}。因此不能仅按主实验最高Sharpe把Transformer认定为最可靠方案。LSTM三个种子的期末净值从{lstm_seeds.terminal_wealth.min():.3f}到{lstm_seeds.terminal_wealth.max():.3f}，也说明单次训练结果不稳定。','',
        '![随机种子差异](figures/04_random_seed_variation.png)','',
        '![执行延迟敏感性](figures/05_execution_delay_sensitivity.png)','',
        '## 6. KL＋机器学习是否改善了风险预测','',
        '另外使用此前完全相同的201个固定等权组合（全体100股1个、随机10股100个、随机30股100个），在相同21日测试段比较预测协方差。所有预测只用当时信息；实现风险以训练均值为中心计算。MAE、QLIKE、Gaussian协方差评分均越低越好；校准比理想值约1。最终不足21日的测试段不进入这个评分。它与投资组合收益评价不同。', '',
        mdtable(risk), '',
        f'样本平均波动MAE：RMT为{risk_by_name.loc["rmt_clipped_gmv","mae_pp"]:.4f}个百分点，KL＋GRU为{risk_by_name.loc["kl_gru","mae_pp"]:.4f}，KL＋TCN为{risk_by_name.loc["kl_tcn","mae_pp"]:.4f}。后者相对RMT下降{1-risk_by_name.loc["kl_tcn","mae_pp"]/risk_by_name.loc["rmt_clipped_gmv","mae_pp"]:.1%}，相对不使用神经网络的KL历史均方控制下降{1-risk_by_name.loc["kl_tcn","mae_pp"]/risk_by_name.loc["kl_ewma","mae_pp"]:.1%}。这是该评价样本的点估计；两条KL神经网络组合的Sharpe仍低于RMT，不能将风险预测改善等同于投资收益改善。','',
        '![样本外风险预测比较](figures/07_out_of_sample_risk_prediction.png)','',
        '## 7. 本次实验的解释边界','',
        '- 数据是从固定研究池事后选出的长历史100股，存在存活者与选择偏差；没有重建每年的真实成分或历史可交易股票池。结果不能直接解释为真实历史可实现的投资收益。',
        '- 这一历史区间此前已经反复查看，现有比较属于探索性实验。模型结构与实验规则在本轮拟合前固定，但这不能把已查看过的历史恢复成真正未知的最终测试集。',
        '- 只有收盘数据支持的理想化成交；统一延迟可减少同收盘信息泄漏，但未模拟停牌、涨跌停无法成交、冲击成本、税费或实际开盘执行。因此这是一套可比较的研究回测。',
        '- 小型网络、月度成熟标签和最多8轮属于针对这份数据的一个明确实验设定，不是对每种算法做无限制调参。某个模型较差，不等于该模型家族必然无效；某个较好，也不等于实盘稳定更好。',
        '- 净值较高可能同时承担更大回撤；模型排名依指标、年份、种子和执行延迟变化。风险矩阵预测改善也不必然提高净收益。', '',
        f'## 8. 文件与校验','',
        f'通过 **{len(checks)}项**时间边界、权重、数据哈希和图像校验，共保存 **{len(train_records)}条**模型—年度—种子训练记录。validation.json记录版本和检查，training_cutoffs_and_parameters.csv列出每年数据截止日与所选参数；每个模型目录保留预测、种子权重、完整训练审计及模型检查点。',
        '- [离线图表和全部结果入口](index.html)',
        '- [中文主要结果表](主要结果中文.csv)、[完整指标](performance_summary.csv)、[每日净收益](daily_net_returns.csv)、[统一起点净值](normalized_equity.csv)。',
        '- [全部目标权重](ensemble_target_weights.csv.gz)、[种子结果](seed_performance.csv)、[相对RMT区间](paired_bootstrap_vs_rmt.csv)。',
        '- annual_performance.csv、stress_period_performance.csv包含完整逐年与预定压力时段表现。',
        '- 上一级目录的 hs300_long100_ml_results.zip 包含全部研究结果、模型检查点、代码快照、版本锁定和输入收益；98MB的可重建特征缓存prepared.npz不重复打包。', '',
        '在项目根目录复跑：','', '```bash',
        'python3 -m venv .cache/ml_venv', '.cache/ml_venv/bin/python -m pip install -r requirements_ml_lock.txt',
        '.cache/ml_venv/bin/python run_ml.py',
        '```','',
        '已有完成结果时会校验代码版本后复用，不重新训练；只更新报告使用 run_ml.py --report-only。换参数或代码请指定新的 --output，避免与本次结果混用。','']
    (out/'机器学习组合完整报告.md').write_text('\n'.join(content))


def package(root,out,summary,versions):
    inputs=out/'inputs';inputs.mkdir(exist_ok=True)
    for source in [root/'outputs/hs300_long100/full/log_returns.csv',
                   root/'outputs/hs300_long100/full/excluded_nontrading_dates.csv',
                   root/'outputs/hs300_long100/full/rmt/tables/09_portfolio_daily_returns.csv',
                   root/'outputs/hs300_long100/full/modes/tables/risk_portfolio_manifest.csv']:
        shutil.copy2(source,inputs/source.name)
    snapshot=out/'reproduction_code';snapshot.mkdir(exist_ok=True)
    for source in [root/'run_ml.py',root/'requirements_ml.txt',root/'requirements_ml_lock.txt',*sorted((root/'ml_experiment').glob('*.py'))]:
        target=snapshot/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    sections=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>机器学习组合与RMT比较</title><style>body{max-width:1450px;margin:30px auto;padding:0 20px;font:16px/1.6 system-ui;color:#18364b}a{color:#17618b}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:6px 9px;border-bottom:1px solid #ddd}figure{margin:30px 0}li{overflow-wrap:anywhere}</style>',
        '<h1>100股长历史数据：20种方法与RMT clipped GMV比较</h1>',
        '<p>统一样本外区间2010-01-29—2026-11-09；净值起点1。随机模型三种子平均权重。全部为探索性历史实验。</p>',
        '<p><a href="机器学习组合完整报告.md">完整中文报告</a> · <a href="performance_summary.csv">全部指标CSV</a> · <a href="normalized_equity.csv">每日净值CSV</a> · <a href="validation.json">校验</a></p>',
        pd.read_csv(out/'主要结果中文.csv').to_html(index=False,float_format=lambda x:f'{x:.3f}'),'<h2>全部图表</h2>']
    for path in sorted((out/'figures').glob('*.png')):
        rel=path.relative_to(out).as_posix();sections.append(f'<figure><a href="{rel}"><img loading="lazy" src="{rel}"></a><figcaption>{path.name}</figcaption></figure>')
    sections.append('<h2>数据与运行记录</h2><ul>')
    for path in sorted(out.glob('*')):
        if path.is_file() and path.name not in ['prepared.npz','index.html']:
            sections.append(f'<li><a href="{html.escape(path.name)}">{html.escape(path.name)}</a></li>')
    sections.append('</ul><h2>各模型检查点和训练记录</h2>')
    for name in ALL:
        sections.append(f'<details><summary>{LABELS[name]}</summary><ul>')
        for path in sorted((out/'models'/name).glob('*')):
            rel=path.relative_to(out).as_posix();sections.append(f'<li><a href="{rel}">{path.name}</a></li>')
        sections.append('</ul></details>')
    sections.append('</html>');(out/'index.html').write_text('\n'.join(sections))
    records=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name not in ['artifact_manifest.csv','prepared.npz'] and path.suffix!='.log':
            records.append({'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':sha(path)})
    pd.DataFrame(records).to_csv(out/'artifact_manifest.csv',index=False)
    archive=out.parent/'hs300_long100_ml_results.zip';temp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as pack:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name!='prepared.npz':
                pack.write(path,f'hs300_long100_ml_results/{path.relative_to(out)}')
    temp.replace(archive)
    with zipfile.ZipFile(archive) as pack:
        if pack.testzip() is not None:
            raise AssertionError('Archive CRC check failed')
    print(summary[['model','terminal_wealth','annual_return','sharpe_rf0','max_drawdown']].to_string(index=False),flush=True)
    print(f'ALL COMPLETE: {archive}; {archive.stat().st_size/1e6:.1f} MB',flush=True)
