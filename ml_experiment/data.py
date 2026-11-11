from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HORIZON, LOOKBACK, ANNUAL = 21, 40, 242
FEATURE_NAMES = ['return_1', 'momentum_5', 'momentum_21', 'momentum_63', 'momentum_120',
                 'volatility_21', 'volatility_63', 'drawdown_63', 'market_beta_252',
                 'relative_momentum_21', 'market_loading', 'second_mode_energy',
                 'group_mode_energy', 'market_removed_return', 'market_mode_score',
                 'market_variance_fraction', 'relative_volatility', 'zero_return_fraction_63']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def prepare(root, output):
    source = root/'outputs/hs300_long100/full/log_returns.csv'
    baseline = root/'outputs/hs300_long100/full/rmt/tables/09_portfolio_daily_returns.csv'
    output.mkdir(parents=True, exist_ok=True)
    cache = output/'prepared.npz'
    fingerprint = {'returns_sha256': sha(source), 'legacy_baseline_sha256': sha(baseline),
                   'feature_implementation_sha256': sha(__file__)}
    if cache.exists():
        previous = json.loads((output/'input_audit.json').read_text())
        if any(previous.get(k) != v for k, v in fingerprint.items()):
            raise ValueError('输入或特征代码已变，使用新的输出目录，不能复用旧缓存。')
        return load(output)
    frame = pd.read_csv(source, index_col=0, parse_dates=True)
    x = frame.to_numpy(); t, n = x.shape
    assert n == 100 and t == 6001 and np.isfinite(x).all()
    m = frame.mean(axis=1)
    std21, std63 = frame.rolling(21).std(), frame.rolling(63).std()
    marketvar = m.rolling(252).var().clip(lower=1e-12)
    beta = frame.rolling(252).cov(m).div(marketvar, axis=0)
    logprice = frame.cumsum()
    features = [frame, frame.rolling(5).sum(), frame.rolling(21).sum(),
                frame.rolling(63).sum(), frame.rolling(120).sum(), std21, std63,
                logprice-logprice.rolling(63).max(), beta,
                frame.rolling(21).sum().sub(m.rolling(21).sum(), axis=0)]
    base = np.stack([v.to_numpy() for v in features], axis=2)
    dynamic = np.zeros((t, n, 6), dtype=np.float32)
    graph = np.zeros((t, n, n), dtype=np.float16)
    for end in range(251, t):
        sample = x[end-251:end+1]
        sd = sample.std(axis=0, ddof=0)
        z = (sample-sample.mean(axis=0))/np.maximum(sd, 1e-9)
        c = z.T@z/len(z)
        vals, vec = np.linalg.eigh(c)
        v = vec[:, -1]
        if v.sum() < 0:
            v = -v
        rawscore = z[-1]@v
        residual = c-vals[-1]*np.outer(v, v)
        adjacency = np.maximum(residual, 0)
        np.fill_diagonal(adjacency, 1)
        adjacency /= adjacency.sum(axis=1, keepdims=True)
        graph[end] = adjacency
        dynamic[end, :, 0] = v*np.sqrt(n)
        dynamic[end, :, 1] = n*vec[:, -2]**2
        dynamic[end, :, 2] = n*np.sum(vec[:, -5:-1]**2, axis=1)
        dynamic[end, :, 3] = z[-1]-rawscore*v
        dynamic[end, :, 4] = rawscore/np.sqrt(vals[-1])
        dynamic[end, :, 5] = vals[-1]/n
        if end % 1000 == 0:
            print(f'causal 252-day PCA features: {end}/{t}', flush=True)
    relative_vol = std21.div(std21.mean(axis=1), axis=0).to_numpy()[..., None]
    zero = (frame.abs() < 1e-14).rolling(63).mean().to_numpy()[..., None]
    f = np.concatenate([base, dynamic, relative_vol, zero], axis=2).astype(np.float32)
    f[:251] = 0
    assert np.isfinite(f[251:]).all()
    targets = np.zeros((t, n), dtype=np.float32)
    future = np.zeros((t, HORIZON, n), dtype=np.float32)
    valid = np.zeros(t, dtype=bool)
    # Feature date d; execute one full day later; own returns d+2 ... d+22.
    for d in range(t-HORIZON-1):
        future[d] = np.expm1(x[d+2:d+HORIZON+2])
        score = x[d+2:d+HORIZON+2].sum(axis=0)
        targets[d] = (rankdata(score)-1)/(n-1)-.5
        valid[d] = True
    starts = np.arange(504, t, HORIZON)
    starts = starts[frame.index[starts].year >= 2010]
    # Both primary and extra-day delay forecasts use a model fit by the earlier date.
    decisions = np.stack([starts-2, starts-3], axis=1)
    np.savez_compressed(cache, returns=x, features=f, graph=graph, targets=targets,
                        future=future, valid_targets=valid, starts=starts, decisions=decisions,
                        dates=frame.index.strftime('%Y-%m-%d').to_numpy(dtype=str),
                        tickers=np.asarray(frame.columns, dtype=str))
    dump(output/'input_audit.json', {**fingerprint, 'rows': t, 'assets': n,
         'first_evaluation': str(frame.index[starts[0]].date()), 'last_evaluation': str(frame.index[-1].date()),
         'features': FEATURE_NAMES, 'pca_window': 252, 'no_full_sample_pca': True,
         'label': 'cross-sectional rank of 21-day log return, beginning d+2',
         'primary_signal_lag_days': 1, 'sensitivity_signal_lag_days': 2})
    return load(output)


def load(output):
    raw = np.load(Path(output)/'prepared.npz')
    return {key: raw[key] for key in raw.files}


def training_split(cutoff, t):
    # Monthly anchors share the old benchmark's 21-day rebalance grid.
    anchors = np.arange(504, t, HORIZON)-2
    available = anchors[(anchors >= cutoff-6*ANNUAL) & (anchors+HORIZON+1 <= cutoff)]
    validation = available[available >= cutoff-HORIZON-1-ANNUAL]
    train = available[available+HORIZON+1 < validation.min()]
    assert len(train) >= 35 and len(validation) >= 8
    assert np.max(train)+HORIZON+1 < validation.min()
    assert available.max()+HORIZON+1 <= cutoff
    return train, validation, available


def sequences(features, indices):
    indices = np.asarray(indices)
    positions = indices[:, None]+np.arange(-LOOKBACK+1, 1)[None, :]
    return features[positions].transpose(0, 2, 1, 3).copy()


def tabular(features, indices):
    seq = sequences(features, indices)
    return np.concatenate([seq[:, :, -1], seq.mean(axis=2), seq[:, :, -1]-seq[:, :, 0]], axis=-1)


def factor_task(x, cutoff, count=10):
    sample = x[cutoff-503:cutoff+1]
    mean, sd = sample.mean(axis=0), np.maximum(sample.std(axis=0, ddof=0), 1e-9)
    z = (sample-mean)/sd
    vals, vec = np.linalg.eigh(z.T@z/len(z))
    vals, vec = vals[-count:][::-1], vec[:, -count:][:, ::-1]
    for j in range(count):
        if vec[np.argmax(abs(vec[:, j])), j] < 0:
            vec[:, j] *= -1
    scores = ((x-mean)/sd)@vec/np.sqrt(vals)
    b = sd[:, None]*vec*np.sqrt(vals)
    residual = x-mean-scores@b.T
    a = pd.DataFrame(scores)
    feats = np.stack([scores, abs(scores), scores**2,
                      a.pow(2).rolling(5).mean().to_numpy(),
                      a.pow(2).rolling(21).mean().to_numpy(),
                      a.pow(2).rolling(63).mean().to_numpy()], axis=-1).astype(np.float32)
    feats[:62] = 0
    targets = np.zeros_like(scores, dtype=np.float32)
    for d in range(len(x)-HORIZON-1):
        targets[d] = np.log(np.maximum(np.mean(scores[d+2:d+HORIZON+2]**2, axis=0), 1e-5))
    return feats, targets, b, residual, scores
