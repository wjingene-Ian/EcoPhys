from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from .data import prepare, load, dump, sha, training_split, factor_task, HORIZON, ANNUAL
from .models import ALL, NEURAL, CLASSICAL, RISK, neural_fit, neural_predict, classic_fit_predict

SEEDS = [17, 42, 2026]
CAP = .05


def project(y, cap=CAP):
    low, high = y.min()-cap, y.max()
    for _ in range(60):
        middle = (low+high)/2
        if np.clip(y-middle, 0, cap).sum() > 1:
            low = middle
        else:
            high = middle
    return np.clip(y-(low+high)/2, 0, cap)


def gmv(covariance, allowed=None, initial=None):
    n = len(covariance)
    allowed = np.arange(n) if allowed is None else np.asarray(allowed)
    cov = covariance[np.ix_(allowed, allowed)].copy()
    scale = np.trace(cov)/len(cov)
    cov = cov/max(scale, 1e-12)+np.eye(len(cov))*1e-6
    start = np.full(len(cov), 1/len(cov)) if initial is None else project(initial[allowed])
    answer = minimize(lambda w: float(w@cov@w), start, jac=lambda w: 2*cov@w,
        method='SLSQP', bounds=[(0, CAP)]*len(cov),
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1, 'jac': lambda w: np.ones(len(w))}],
        options={'ftol': 1e-10, 'maxiter': 500})
    if not answer.success:
        raise RuntimeError(answer.message)
    w = np.zeros(n); w[allowed] = answer.x
    assert abs(w.sum()-1) < 1e-7 and w.min() >= -1e-9 and w.max() <= CAP+1e-8
    return w


def covariance(x, decision, kind):
    sample = x[decision-503:decision+1]
    cov = np.cov(sample, rowvar=False, ddof=1)
    if kind == 'raw_gmv':
        return cov
    if kind != 'rmt_clipped_gmv':
        return .8*cov+.2*np.diag(np.diag(cov))
    sd = sample.std(axis=0, ddof=1)
    c = cov/np.outer(sd, sd)
    vals, vec = np.linalg.eigh(c)
    mask = vals <= (1+np.sqrt(x.shape[1]/len(sample)))**2
    vals[mask] = vals[mask].mean()
    c = (vec*vals)@vec.T
    c /= np.sqrt(np.outer(np.diag(c), np.diag(c)))
    return c*np.outer(sd, sd)


def hrp(cov):
    corr = cov/np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    distance = np.sqrt(np.maximum(0, (1-np.clip(corr, -1, 1))/2))
    np.fill_diagonal(distance, 0)
    order = leaves_list(linkage(squareform(distance, checks=False), method='single'))
    weights = np.ones(len(cov)); groups = [order]
    def variance(indices):
        c = cov[np.ix_(indices, indices)]
        w = 1/np.diag(c); w /= w.sum()
        return w@c@w
    while groups:
        new = []
        for group in groups:
            if len(group) <= 1:
                continue
            a, b = group[:len(group)//2], group[len(group)//2:]
            va, vb = variance(a), variance(b)
            weights[a] *= vb/(va+vb); weights[b] *= va/(va+vb)
            new.extend([a, b])
        groups = new
    return project(weights)


def fit_model(name, data, out, max_epochs=8):
    destination = out/'models'/name; destination.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(Path(__file__).parents[1])): sha(p)
               for p in [Path(__file__).parent/name for name in ['data.py', 'models.py', 'run.py']]}
    result_path = destination/'targets.npz'
    if result_path.exists():
        previous = json.loads((destination/'metadata.json').read_text())
        if previous['code_sha256'] != sources or previous['max_epochs'] != max_epochs:
            raise ValueError(f'{name}: implementation changed; use a fresh output folder')
        print(f'{name}: reuse completed model', flush=True); return
    x, f, graph = data['returns'], data['features'], data['graph']
    starts, decisions = data['starts'], data['decisions']
    dates = pd.to_datetime(data['dates']); n = x.shape[1]
    years = dates[starts].year.to_numpy()
    deterministic = name in ['equal_weight', 'raw_gmv', 'shrink20_gmv', 'rmt_clipped_gmv',
                              'hrp', 'ridge', 'elastic_net', 'kl_static', 'kl_ewma']
    seeds = SEEDS[:1] if deterministic else SEEDS
    weights = np.zeros((len(seeds), 2, len(starts), n), dtype=np.float64)
    signals = np.zeros_like(weights)
    logs = []
    start_time = time.monotonic()
    for seed_index, seed in enumerate(seeds):
        prior = [None, None]
        for year in np.unique(years):
            period_ids = np.flatnonzero(years == year)
            predict_indices = decisions[period_ids].ravel()
            cutoff = int(predict_indices.min())
            train, validation, final = training_split(cutoff, len(x))
            record = {'model': name, 'seed': seed, 'year': int(year),
                      'fit_information_cutoff': str(dates[cutoff].date()),
                      'train_anchors': len(train), 'validation_anchors': len(validation), 'final_fit_anchors': len(final),
                      'max_final_label_date': str(dates[final.max()+HORIZON+1].date()),
                      'first_evaluation_date': str(dates[starts[period_ids[0]]].date()),
                      'validation_first_feature_date': str(dates[validation.min()].date()),
                      'train_max_label_date': str(dates[train.max()+HORIZON+1].date()),
                      'forecast_information_dates': [str(dates[d].date()) for d in predict_indices]}
            assert final.max()+HORIZON+1 <= cutoff < starts[period_ids[0]]-1
            tick = time.monotonic()
            if name in CLASSICAL:
                pred, model, scaler, selection = classic_fit_predict(name, f, data['targets'],
                    train, validation, final, predict_indices, seed)
                joblib.dump({'model': model, 'scaler': scaler, 'record': record}, destination/f'{year}_seed{seed}.joblib')
                record.update(selection)
            elif name in NEURAL:
                _, _, epochs, history = neural_fit(name, f, data['targets'], train, seed, graph,
                    data['future'], epochs=max_epochs, validation=validation)
                model, scaler, _, refit_history = neural_fit(name, f, data['targets'], final, seed,
                    graph, data['future'], epochs=epochs)
                pred = neural_predict(model, scaler, f, predict_indices, graph)
                torch.save({'state_dict': model.state_dict(), 'scaler': scaler, 'record': record},
                           destination/f'{year}_seed{seed}.pt')
                record.update(selected_epochs=epochs, validation_history=history, refit_history=refit_history)
            elif name in RISK:
                if name in ['kl_gru', 'kl_tcn']:
                    validation_basis_cutoff = int(train.max())
                    vf, vy, *_ = factor_task(x, validation_basis_cutoff)
                    architecture = name.removeprefix('kl_')
                    _, _, epochs, history = neural_fit(architecture, vf, vy, train, seed,
                        epochs=max_epochs, validation=validation)
                ff, fy, basis, residual, factor = factor_task(x, cutoff)
                if name in ['kl_gru', 'kl_tcn']:
                    model, scaler, _, refit_history = neural_fit(architecture, ff, fy, final, seed, epochs=epochs)
                    log_variance = neural_predict(model, scaler, ff, predict_indices)
                    pred = np.exp(np.clip(log_variance, -3, 3))
                    torch.save({'state_dict': model.state_dict(), 'scaler': scaler, 'basis': basis,
                                'record': record}, destination/f'{year}_seed{seed}.pt')
                    record.update(selected_epochs=epochs, validation_history=history, refit_history=refit_history)
                else:
                    pred = np.ones((len(predict_indices), 10))
                    if name == 'kl_ewma':
                        pred = np.asarray([np.mean(factor[d-62:d+1]**2, axis=0) for d in predict_indices])
                np.savez_compressed(destination/f'{year}_seed{seed}_factor_basis.npz', basis=basis,
                                    indices=predict_indices, predicted_variance=pred)
            else:
                pred = np.zeros((len(predict_indices), n))
            pred = pred.reshape(len(period_ids), 2, -1)
            for local, period in enumerate(period_ids):
                for lag in range(2):
                    decision = int(decisions[period, lag])
                    if name == 'equal_weight':
                        w = np.full(n, 1/n)
                    elif name == 'direct_weights':
                        w = pred[local, lag]
                    elif name in RISK:
                        dvar = np.maximum(residual[decision-62:decision+1].var(axis=0, ddof=1), 1e-10)
                        cov = (basis*pred[local, lag])@basis.T+np.diag(dvar)
                        w = gmv(cov, initial=prior[lag])
                    else:
                        cov = covariance(x, decision, name)
                        if name == 'hrp':
                            w = hrp(cov)
                        elif name in ['raw_gmv', 'shrink20_gmv', 'rmt_clipped_gmv']:
                            w = gmv(cov, initial=prior[lag])
                        else:
                            allowed = np.argsort(pred[local, lag], kind='stable')[-30:]
                            w = gmv(cov, allowed, prior[lag])
                    weights[seed_index, lag, period] = w
                    if pred.shape[-1] == n:
                        signals[seed_index, lag, period] = pred[local, lag]
                    prior[lag] = w.copy()
            record['elapsed_seconds'] = round(time.monotonic()-tick, 3)
            logs.append(record)
            dump(destination/'training_audit.json', logs)
            print(f'{name} seed={seed} year={year}: train={len(train)}, val={len(validation)}, '
                  f'epochs={record.get("selected_epochs", "-")}, {record["elapsed_seconds"]:.1f}s', flush=True)
    assert np.isfinite(weights).all() and np.allclose(weights.sum(axis=-1), 1, atol=1e-6)
    assert weights.min() >= -1e-7 and weights.max() <= CAP+1e-7
    np.savez_compressed(result_path, weights=weights, signals=signals, seeds=seeds, starts=starts, decisions=decisions)
    dump(destination/'metadata.json', {'model': name, 'seeds': seeds,
        'elapsed_seconds': time.monotonic()-start_time, 'code_sha256': sources, 'max_epochs': max_epochs,
        'years': np.unique(years).tolist(), 'target_shape': weights.shape,
        'ensemble': 'arithmetic mean of seed target weights; never select best test seed'})
    print(f'COMPLETE {name}: {time.monotonic()-start_time:.1f}s', flush=True)


def simulate(data, weights, cost=.001):
    simple = np.expm1(data['returns']); starts = data['starts']
    values, rows, current = [], [], None
    for period, start in enumerate(starts):
        target = weights[period]
        turnover = 0. if current is None else .5*np.abs(target-current).sum()
        current = target.copy()
        rows.append({'start': int(start), 'date': str(data['dates'][start]), 'turnover': turnover,
                     'maximum_weight': target.max(), 'effective_stocks': 1/np.sum(target**2)})
        for j, asset in enumerate(simple[start:min(start+HORIZON, len(simple))]):
            gross = float(current@asset)
            net = (1-turnover*cost)*(1+gross)-1 if j == 0 else gross
            values.append(net)
            current *= (1+asset)/(1+gross)
    return np.asarray(values), pd.DataFrame(rows)


def main(root):
    parser = argparse.ArgumentParser(description='Chronological ML portfolio comparison on hs300_long100')
    parser.add_argument('--output', type=Path, default=root/'outputs/hs300_long100/ml_portfolios')
    parser.add_argument('--models', default=','.join(ALL))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    output = args.output.resolve()
    if args.report_only:
        from .report import report
        report(root, output); return
    data = prepare(root, output)
    plan = {'status': 'fixed before fitting', 'models': ALL, 'seeds': SEEDS,
            'training': 'rolling past six years; final year chronological validation; purged 21-day labels plus execution day',
            'sampling': 'same 21-day anchors as benchmark; shared stock model, no independently trained 100 networks',
            'prediction_target': 'relative 21-day return rank, training labels normalized cross-sectionally',
            'evaluation_start_year': 2010, 'refit': 'annual, cutoff is earliest information date of both execution lags',
            'signal_lag_primary': 1, 'signal_lag_sensitivity': 2,
            'covariance_window': 504, 'rebalance_days': 21, 'cap': CAP, 'cost_bps_one_way': 10,
            'first_build_cost': 0, 'annualization': 242, 'cash': False,
            'allocation_for_all_return_predictors': 'top30 then long-only GMV with same 20% diagonal shrinkage covariance',
            'neural_hidden': 16, 'neural_layers': 1, 'max_epochs': args.max_epochs, 'early_stopping_patience': 2,
            'transformer': 'patch length5, lookback40, d_model16, heads2, one encoder; PatchTST-inspired, not an exact paper replication',
            'graph': 'positive market-removed 252-day correlation, self-loops, row normalized; no 2026 industry snapshot',
            'direct_weight_loss': 'annualized negative mean plus 1.5 variance plus target-turnover penalty; simplex/cap enforced',
            'KL_risk': '10 factors fitted only to past data; positive diagonal factor variances plus past63-day residual diagonal',
            'KL_variance_prediction_clip': 'exp(log_variance clipped to [-3,3])',
            'reporting': 'all methods, three stochastic seeds, mean seed weights, no test-based best seed selection',
            'interpretation': 'exploratory: fixed long-history cohort and previously examined evaluation period'}
    if not (output/'experiment_plan.json').exists():
        dump(output/'experiment_plan.json', plan)
    if args.prepare_only:
        print('Feature preparation complete', flush=True); return
    names = args.models.split(',')
    if any(name not in ALL for name in names):
        parser.error('Unknown model')
    for name in names:
        fit_model(name, data, output, args.max_epochs)
    if names == ALL:
        from .report import report
        report(root, output)
