"""Implementation checks independent of model selection and test performance."""
from __future__ import annotations
import numpy as np
import torch
from scipy.stats import rankdata
from .data import HORIZON, dump
from .models import ForecastNet, neural_predict, feature_scaler
from .run import simulate, gmv, covariance


def implementation_checks(data,out):
    checks={};x=data['returns'];f=data['features'];n=x.shape[1]
    for d in [1000,3000,5000]:
        # Recompute from an actual prefix, without making future rows available.
        prefix=x[:d+1]
        sample=prefix[-252:];sd=np.maximum(sample.std(axis=0,ddof=0),1e-9)
        z=(sample-sample.mean(axis=0))/sd
        c=z.T@z/len(z);vals,vec=np.linalg.eigh(c)
        v=vec[:,-1]
        if v.sum()<0:v=-v
        local=np.column_stack([v*np.sqrt(n),n*vec[:,-2]**2,n*np.sum(vec[:,-5:-1]**2,axis=1),
            z[-1]-(z[-1]@v)*v,np.full(n,(z[-1]@v)/np.sqrt(vals[-1])),np.full(n,vals[-1]/n)])
        checks[f'PCA_features_match_prefix_only_{d}']=bool(np.allclose(local,f[d,:,10:16],atol=1e-6))
        checks[f'rolling_momentum_and_volatility_match_prefix_{d}']=bool(
            np.allclose(prefix[-21:].sum(axis=0),f[d,:,2],atol=1e-7) and
            np.allclose(prefix[-21:].std(axis=0,ddof=1),f[d,:,5],atol=1e-7))
        future=x[d+2:d+HORIZON+2]
        target=(rankdata(future.sum(axis=0))-1)/(n-1)-.5
        checks[f'target_horizon_and_execution_gap_match_{d}']=bool(
            np.allclose(target,data['targets'][d],atol=1e-7) and
            np.allclose(np.expm1(future),data['future'][d],atol=1e-7))
    anchors=np.array([504,525,546])
    scaler=feature_scaler(f,anchors)
    perturbed=f.copy();perturbed[547:]=123456.
    changed=feature_scaler(perturbed,anchors)
    checks['training_scaler_invariant_to_all_future_feature_values']=bool(all(np.array_equal(a,b) for a,b in zip(scaler,changed)))
    del perturbed
    # These are checkpoints produced by this run, not untrusted downloaded objects.
    saved=torch.load(out/'models/gru/2010_seed17.pt',map_location='cpu',weights_only=False)
    model=ForecastNet('gru',f.shape[-1]);model.load_state_dict(saved['state_dict'])
    ids=np.arange(3);indices=data['decisions'][ids].ravel()
    prediction=neural_predict(model,saved['scaler'],f,indices).reshape(len(ids),2,-1)
    original=np.load(out/'models/gru/targets.npz')
    checks['saved_GRU_checkpoint_reproduces_actual_forecasts']=bool(np.allclose(prediction,original['signals'][0,:,:3].transpose(1,0,2),atol=1e-7))
    expected=gmv(covariance(x,int(data['decisions'][0,0]),'shrink20_gmv'),np.argsort(prediction[0,0],kind='stable')[-30:])
    checks['first_GRU_allocation_reproduced_from_checkpoint']=bool(np.allclose(expected,original['weights'][0,0,0],atol=1e-7))
    toy={'returns':np.column_stack([np.full(42,np.log1p(.01)),np.zeros(42)]),
         'starts':np.array([0,21]),'dates':np.array([str(i) for i in range(42)])}
    returns,rebalance=simulate(toy,np.array([[1.,0.],[0.,1.]]))
    checks['fee_and_holding_return_formula_matches_closed_form']=bool(
        np.allclose(returns[:21],.01) and np.isclose(returns[21],-.001) and
        np.allclose(returns[22:],0) and np.isclose(np.prod(1+returns),1.01**21*.999) and
        np.array_equal(rebalance.turnover.to_numpy(),[0.,1.]))
    returns,rebalance=simulate(toy,np.array([[.5,.5],[.5,.5]]))
    drift=np.array([1.01**21,1.]);drift/=drift.sum()
    checks['turnover_uses_drifted_weights_not_previous_targets']=bool(np.isclose(rebalance.turnover.iloc[1],.5*abs(drift-.5).sum()))
    if not all(checks.values()):
        raise AssertionError([key for key,value in checks.items() if not value])
    dump(out/'implementation_checks.json',checks)
    return checks
