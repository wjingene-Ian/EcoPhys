from __future__ import annotations
from copy import deepcopy
import time
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge, ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from .data import LOOKBACK, HORIZON, sequences, tabular

NEURAL = ['rnn', 'lstm', 'gru', 'tcn', 'transformer', 'gnn_gru', 'direct_weights']
CLASSICAL = ['ridge', 'elastic_net', 'lightgbm', 'xgboost']
RISK = ['kl_gru', 'kl_tcn', 'kl_static', 'kl_ewma']
ALL = ['equal_weight', 'raw_gmv', 'shrink20_gmv', 'rmt_clipped_gmv', 'hrp', *CLASSICAL, *NEURAL, *RISK]


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__(); self.padding = 2*dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation, padding=self.padding)
        self.dropout = nn.Dropout(.1)
    def forward(self, x):
        return x+self.dropout(torch.relu(self.conv(x)[..., :-self.padding]))


class ForecastNet(nn.Module):
    def __init__(self, kind, features, hidden=16):
        super().__init__(); self.kind = kind
        if kind in ['rnn', 'lstm', 'gru', 'gnn_gru']:
            cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU, 'gnn_gru': nn.GRU}[kind]
            self.encoder = cls(features, hidden, batch_first=True)
        elif kind == 'tcn':
            self.encoder = nn.Sequential(nn.Conv1d(features, hidden, 1),
                *[CausalBlock(hidden, d) for d in [1, 2, 4, 8]])
        elif kind == 'transformer':
            self.patch = nn.Linear(5*features, hidden)
            self.position = nn.Parameter(torch.randn(1, LOOKBACK//5, hidden)*.02)
            layer = nn.TransformerEncoderLayer(hidden, 2, dim_feedforward=32,
                dropout=.1, batch_first=True, activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        elif kind == 'direct_weights':
            self.encoder = nn.Sequential(nn.Linear(2*features, hidden), nn.Tanh())
        if kind == 'gnn_gru':
            self.graph_layer = nn.Sequential(nn.Linear(2*hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(hidden, 1))

    def forward(self, x, graph=None):
        batch, nodes, length, features = x.shape
        flat = x.reshape(batch*nodes, length, features)
        if self.kind in ['rnn', 'lstm', 'gru', 'gnn_gru']:
            h = self.encoder(flat)[0][:, -1]
        elif self.kind == 'tcn':
            h = self.encoder(flat.transpose(1, 2))[:, :, -1]
        elif self.kind == 'transformer':
            patches = flat.reshape(batch*nodes, length//5, features*5)
            h = self.encoder(self.patch(patches)+self.position).mean(dim=1)
        else:
            h = self.encoder(torch.cat([flat[:, -1], flat.mean(dim=1)], dim=-1))
        h = h.reshape(batch, nodes, -1)
        if self.kind == 'gnn_gru':
            h = self.graph_layer(torch.cat([h, torch.bmm(graph, h)], dim=-1))
        scores = self.head(h).squeeze(-1)
        if self.kind == 'direct_weights':
            p = torch.softmax(scores, dim=1)
            uniform = 1/nodes
            # Convexly mix softmax with equal weight; preserves gradient and 5% cap.
            scale = torch.clamp((.05-uniform)/(p.max(dim=1, keepdim=True).values-uniform+1e-9), max=1.)
            return uniform+scale*(p-uniform)
        return scores


def feature_scaler(features, anchors):
    lo, hi = anchors.min()-LOOKBACK+1, anchors.max()+1
    sample = features[lo:hi].reshape(-1, features.shape[-1]).astype(np.float64)
    mean = sample.mean(axis=0); sd = np.maximum(sample.std(axis=0), 1e-6)
    return mean.astype(np.float32), sd.astype(np.float32)


def neural_fit(kind, features, target, anchors, seed, graph=None, future=None,
               epochs=8, validation=None, scaler=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ForecastNet(kind, features.shape[-1])
    scaler = feature_scaler(features, anchors) if scaler is None else scaler
    mean, sd = scaler
    seq = np.clip((sequences(features, anchors)-mean)/sd, -8, 8)
    data = torch.from_numpy(seq)
    y = torch.from_numpy(target[anchors])
    adjacency = torch.from_numpy(graph[anchors].astype(np.float32)) if kind == 'gnn_gru' else None
    actual = torch.from_numpy(future[anchors]) if kind == 'direct_weights' else None
    if kind == 'direct_weights':
        previous = torch.from_numpy(np.clip((sequences(features, anchors-HORIZON)-mean)/sd, -8, 8))
    if validation is not None:
        vseq = torch.from_numpy(np.clip((sequences(features, validation)-mean)/sd, -8, 8))
        vy = torch.from_numpy(target[validation])
        va = torch.from_numpy(graph[validation].astype(np.float32)) if kind == 'gnn_gru' else None
        vf = torch.from_numpy(future[validation]) if kind == 'direct_weights' else None
        vp = torch.from_numpy(np.clip((sequences(features, validation-HORIZON)-mean)/sd, -8, 8)) if kind == 'direct_weights' else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    rng = np.random.default_rng(seed)
    best, best_epoch, patience, best_state = np.inf, 1, 0, None
    losses = []
    def loss(prediction, truth, returns=None, prior=None):
        if kind != 'direct_weights':
            return nn.functional.mse_loss(prediction, truth)
        portfolio = torch.einsum('bn,bdn->bd', prediction, returns)
        utility = -portfolio.mean()*242+1.5*portfolio.var(unbiased=False)*242
        turnover = .5*torch.abs(prediction-prior).sum(dim=1).mean()
        return utility+(.001*242/HORIZON)*turnover
    for epoch in range(epochs):
        model.train(); batch_losses = []
        for batch in np.array_split(rng.permutation(len(anchors)), max(1, int(np.ceil(len(anchors)/16)))):
            optimizer.zero_grad(set_to_none=True)
            pred = model(data[batch], None if adjacency is None else adjacency[batch])
            prior = model(previous[batch]).detach() if kind == 'direct_weights' else None
            value = loss(pred, y[batch], None if actual is None else actual[batch], prior)
            if not torch.isfinite(value):
                raise ValueError(f'Nonfinite training loss in {kind}')
            value.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); batch_losses.append(float(value.detach()))
        model.eval()
        with torch.no_grad():
            if validation is None:
                metric = float(np.mean(batch_losses))
            else:
                pred = model(vseq, va)
                prior = model(vp).detach() if kind == 'direct_weights' else None
                metric = float(loss(pred, vy, vf, prior))
        losses.append({'epoch': epoch+1, 'training_loss': np.mean(batch_losses), 'validation_loss': metric})
        if metric < best-1e-6:
            best, best_epoch, patience, best_state = metric, epoch+1, 0, deepcopy(model.state_dict())
        else:
            patience += 1
        if validation is not None and patience >= 2:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    return model, scaler, best_epoch if validation is not None else epochs, losses


def neural_predict(model, scaler, features, indices, graph=None):
    mean, sd = scaler; model.eval(); rows = []
    with torch.no_grad():
        for batch in np.array_split(np.asarray(indices), max(1, int(np.ceil(len(indices)/16)))):
            seq = torch.from_numpy(np.clip((sequences(features, batch)-mean)/sd, -8, 8))
            adjacency = torch.from_numpy(graph[batch].astype(np.float32)) if model.kind == 'gnn_gru' else None
            rows.append(model(seq, adjacency).numpy())
    return np.concatenate(rows)


def classic_fit_predict(kind, features, target, train, validation, final, indices, seed):
    scaler = feature_scaler(features, train)
    def transform(anchors, normalization):
        mean, sd = normalization
        normalized = np.clip((features-mean)/sd, -8, 8)
        array = tabular(normalized, anchors)
        return array.reshape(-1, array.shape[-1])
    train_x, val_x = transform(train, scaler), transform(validation, scaler)
    options = {'ridge': [1., 100.], 'elastic_net': [.001, .01],
               'lightgbm': [7, 15], 'xgboost': [2, 3]}[kind]
    def make(option):
        if kind == 'ridge':
            return Ridge(alpha=option)
        if kind == 'elastic_net':
            return ElasticNet(alpha=option, l1_ratio=.5, max_iter=5000, random_state=seed)
        if kind == 'lightgbm':
            return LGBMRegressor(n_estimators=150, num_leaves=option, learning_rate=.05,
                min_child_samples=100, reg_lambda=10., subsample=.8, subsample_freq=1,
                colsample_bytree=.8, n_jobs=1, random_state=seed, verbosity=-1)
        return XGBRegressor(n_estimators=150, max_depth=option, learning_rate=.05,
            min_child_weight=100, reg_lambda=10., subsample=.8, colsample_bytree=.8,
            n_jobs=1, random_state=seed, objective='reg:squarederror')
    best, chosen = np.inf, None
    metrics = []
    for option in options:
        model = make(option); model.fit(train_x, target[train].ravel())
        loss = np.mean((model.predict(val_x)-target[validation].ravel())**2)
        metrics.append({'option': option, 'validation_mse': float(loss)})
        if loss < best:
            best, chosen = loss, option
    scaler = feature_scaler(features, final)
    model = make(chosen); model.fit(transform(final, scaler), target[final].ravel())
    pred = model.predict(transform(indices, scaler)).reshape(len(indices), -1)
    return pred, model, scaler, {'selected_option': chosen, 'validation': metrics}
