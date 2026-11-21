"""Shared code for the leakage-free CWT–BiGRU lithofacies study.

Data are handled as a list of depth-ordered sequences ("wells" or contiguous intervals).
Every model labels every depth sample (sequence-to-sequence); there are no window-majority labels.
All normalisation statistics are fitted on training samples only.
"""
import numpy as np, pywt, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

import os
torch.set_num_threads(int(os.environ.get("LFI_THREADS", "2")))

# ------------------------------------------------------------------ CWT features
NS = 16                                    # number of scales
SCALES = np.geomspace(2, 64, NS)           # in samples (0.5 ft each -> 1 ft to 32 ft)
WAVELET = "cmor1.5-1.0"                    # complex Morlet, bandwidth B=1.5, centre frequency C=1.0
PAD = 64                                   # reflect padding (samples) against edge effects


CWT_MODE = "mag"                          # "mag" = log(1+|W|); "complex" = Re(W), Im(W) as separate channels


def cwt_mag(x):
    """CWT of one standardised log (1-D) -> (NS, len(x)) or (2*NS, len(x)); reflect-padded, then cropped."""
    s = np.pad(x, PAD, mode="reflect")
    co, _ = pywt.cwt(s, SCALES, WAVELET)
    co = co[:, PAD:-PAD]
    if CWT_MODE == "complex":
        co = co / np.sqrt(SCALES)[:, None]                      # scale-normalised real and imaginary parts
        return np.concatenate([co.real, co.imag], 0).astype(np.float32)
    return np.log1p(np.abs(co)).astype(np.float32)


class Scaler:
    """z-score on training samples; one mean/sd per channel (logs) or per channel×scale (CWT)."""
    def fit(self, arrays):                      # arrays: list of (..., T)
        cat = np.concatenate(arrays, axis=-1)
        self.mu = cat.mean(-1, keepdims=True); self.sd = cat.std(-1, keepdims=True) + 1e-8
        return self
    def __call__(self, a): return ((a - self.mu) / self.sd).astype(np.float32)


def build_features(seqs_X, train_mask, log_idx):
    """seqs_X: list of (T_i, F) raw inputs; train_mask: list of bool (T_i,) training samples.
    log_idx: which columns get CWT. Returns lists of raw (F, T) and cwt (len(log_idx), NS, T)."""
    raw = [x.T.astype(np.float32) for x in seqs_X]
    sc = Scaler().fit([r[:, m] for r, m in zip(raw, train_mask) if m.any()])
    raw = [sc(r) for r in raw]
    cw = [np.stack([cwt_mag(r[j]) for j in log_idx], 0) for r in raw]
    sc2 = Scaler().fit([c[..., m] for c, m in zip(cw, train_mask) if m.any()])
    cw = [sc2(c) for c in cw]
    return raw, cw


# ------------------------------------------------------------------ models
class SeqNet(nn.Module):
    """kind:
      raw_gru / raw_bigru      : logs -> (Bi)GRU
      cwt_cnn                  : CWT -> CNN (no recurrence)
      cwt_cnn_bigru            : CWT -> CNN -> BiGRU
      hyb_cnn                  : [logs, CWT-CNN] -> dense (no recurrence)
      hyb_gru / hyb_bigru      : [logs, CWT-CNN] -> GRU / BiGRU      (hyb_bigru = proposed)
      hyb_lstm                 : [logs, CWT-CNN] -> BiLSTM
    """
    def __init__(self, kind, n_raw, n_cwt, n_cls, H=64):
        super().__init__(); self.kind = kind
        use_cwt = kind.startswith("cwt") or kind.startswith("hyb")
        use_raw = kind.startswith("raw") or kind.startswith("hyb")
        self.use_cwt, self.use_raw = use_cwt, use_raw
        d = 0
        if use_raw:
            self.rp = nn.Linear(n_raw, 32); d += 32
        if use_cwt:
            self.cnn = nn.Sequential(                                       # conv over (scale, depth)
                nn.Conv2d(n_cwt, 16, (3, 3), padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)),
                nn.Conv2d(16, 16, (3, 3), padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)))
            nsc = NS * (2 if CWT_MODE == "complex" else 1)
            self.cp = nn.Sequential(nn.Dropout(0.5), nn.Linear(16 * (nsc // 4), 32)); d += 32
        self.drop = nn.Dropout(0.3)
        if kind.endswith("bigru"):   self.rnn = nn.GRU(d, H, batch_first=True, bidirectional=True); o = 2 * H
        elif kind.endswith("gru"):   self.rnn = nn.GRU(d, H, batch_first=True); o = H
        elif kind.endswith("lstm"):  self.rnn = nn.LSTM(d, H, batch_first=True, bidirectional=True); o = 2 * H
        else:                        self.rnn = None; self.mlp = nn.Sequential(nn.Linear(d, H), nn.ReLU()); o = H
        self.head = nn.Linear(o, n_cls)

    def forward(self, raw, cw):                 # raw (B,F,T), cw (B,C,NS,T)
        parts = []
        if self.use_raw: parts.append(self.rp(raw.transpose(1, 2)))
        if self.use_cwt: parts.append(self.cp(self.cnn(cw).permute(0, 3, 1, 2).flatten(2)))
        h = self.drop(torch.relu(torch.cat(parts, 2)))
        if self.rnn is not None: h, _ = self.rnn(h); h = self.drop(h)
        else: h = self.drop(self.mlp(h))
        return self.head(h)                     # (B,T,C)


def _predict(net, raw, cw, mc=0):
    """argmax prediction for one sequence; mc>0 -> MC-dropout mean probabilities."""
    r = torch.tensor(raw)[None]; c = torch.tensor(cw)[None]
    if mc:
        net.eval()
        for m in net.modules():
            if isinstance(m, nn.Dropout): m.train()
        with torch.no_grad(): P = torch.stack([torch.softmax(net(r, c)[0], -1) for _ in range(mc)]).numpy()
        net.eval(); return P                    # (mc, T, C)
    net.eval()
    with torch.no_grad(): return torch.softmax(net(r, c)[0], -1).numpy()


def train_seq(kind, raw, cw, y, tr_mask, va_mask, n_cls, seed, T=128, iters=1500, bs=16, patience=15, log=None, min_iters=0):
    """raw/cw/y/masks: lists per sequence. y = -1 where unlabelled. Returns trained net (best val checkpoint)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    yt = [np.where(m, yy, -1) for yy, m in zip(y, tr_mask)]
    cnt = np.bincount(np.concatenate([yy[yy >= 0] for yy in yt]), minlength=n_cls).astype(float)
    w = np.where(cnt > 0, 1 / np.sqrt(np.maximum(cnt, 1)), 0.0); w = torch.tensor(w / w[w > 0].mean(), dtype=torch.float32)
    net = SeqNet(kind, raw[0].shape[0], cw[0].shape[0], n_cls)
    opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w, ignore_index=-1)
    usable = [i for i, yy in enumerate(yt) if (yy >= 0).any() and len(yy) > T]
    lens = np.array([len(y[i]) for i in usable], float)
    best, state, wait, hist, best_it = -1, None, 0, [], 0
    for it in range(iters):
        net.train()
        ii = rng.choice(usable, bs, p=lens / lens.sum()); st = [rng.integers(0, len(y[i]) - T) for i in ii]
        rb = torch.tensor(np.stack([raw[i][:, s:s + T] for i, s in zip(ii, st)]))
        cb = torch.tensor(np.stack([cw[i][..., s:s + T] for i, s in zip(ii, st)]))
        yb = torch.tensor(np.stack([yt[i][s:s + T] for i, s in zip(ii, st)]))
        if (yb >= 0).sum() == 0: continue
        opt.zero_grad(); out = net(rb, cb); loss = lossf(out.reshape(-1, n_cls), yb.reshape(-1)); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if it % 10 == 9:
            yv, pv, ytr, ptr = [], [], [], []
            for i in range(len(y)):
                if va_mask[i].any() or (log is not None and tr_mask[i].any()):
                    p = _predict(net, raw[i], cw[i]).argmax(1)
                    yv.append(y[i][va_mask[i]]); pv.append(p[va_mask[i]])
                    if log is not None: ytr.append(y[i][tr_mask[i]]); ptr.append(p[tr_mask[i]])
            yv, pv = np.concatenate(yv), np.concatenate(pv)
            sc = accuracy_score(yv, pv) + f1_score(yv, pv, average="macro")
            if log is not None:
                ytr, ptr = np.concatenate(ytr), np.concatenate(ptr)
                hist.append((it + 1, float(loss), accuracy_score(ytr, ptr), accuracy_score(yv, pv)))
            if sc > best: best, wait, best_it, state = sc, 0, it + 1, {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience and it + 1 >= min_iters: break
    net.load_state_dict(state); net.eval(); net.best_it = best_it; net.hist = hist
    return net


def metrics(yt, yp, n_cls, adjacency=None):
    out = {"acc": accuracy_score(yt, yp), "macroF1": f1_score(yt, yp, average="macro", labels=list(range(n_cls)), zero_division=0),
           "weightedF1": f1_score(yt, yp, average="weighted", zero_division=0),
           "perclassF1": f1_score(yt, yp, average=None, labels=list(range(n_cls)), zero_division=0).tolist()}
    if adjacency is not None:
        out["adjacentAcc"] = float(np.mean([(p == t) or (p in adjacency.get(t, ())) for t, p in zip(yt, yp)]))
    return out
