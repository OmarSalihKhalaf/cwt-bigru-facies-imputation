"""Facies imputation in partly cored intervals (Karpur): CWT–BiGRU conditioned on the known facies around a gap.

Scenario: facies are known (core) everywhere except missing intervals of length G. Test gaps of exactly G ft are
placed every 5*G ft (fold k = blocks with index % 5 == k), so every gap has known facies above and below.
Labels inside test gaps are never used for training, early stopping, or as model input.

Usage: python3 -I impute.py <karpur.csv> <G_ft list comma> <n_offsets> <out_json>
"""
import sys, json, time, warnings, os
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, ".")
import core
from core import build_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
core.CWT_MODE = "complex"
CSV, GS, NOFF, OUT = sys.argv[1], [float(g) for g in sys.argv[2].split(",")], int(sys.argv[3]), sys.argv[4]

df = pd.read_csv(CSV)
grid = np.arange(df.depth.min(), df.depth.max() + 0.25, 0.5); L = len(grid)
X = np.stack([np.interp(grid, df.depth, df[c]) for c in ["caliper", "gamma", "density", "phi.N"]], 1)
classes = sorted(df.Facies.unique(), key=lambda s: int(s[1:])); C = len(classes)
lab = {round(d, 1): classes.index(f) for d, f in zip(df.depth, df.Facies)}
y = np.array([lab.get(round(d, 1), -1) for d in grid])


class CtxNet(nn.Module):
    """inputs: logs (raw), CWT (optional), known-facies one-hot + unknown flag. kind: ctx_raw_bigru | ctx_hyb_bigru | ctx_hyb_gru | ctx_lab_bigru"""
    def __init__(s, kind, n_raw, n_cwt, H=64):
        super().__init__(); s.kind = kind; d = 0
        s.use_raw = kind != "ctx_lab_bigru"; s.use_cwt = "hyb" in kind
        if s.use_raw: s.rp = nn.Linear(n_raw, 32); d += 32
        if s.use_cwt:
            s.cnn = nn.Sequential(nn.Conv2d(n_cwt, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)),
                                  nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)))
            s.cp = nn.Sequential(nn.Dropout(0.5), nn.Linear(16 * (2 * core.NS // 4), 32)); d += 32
        s.lp = nn.Linear(C + 1, 16); d += 16
        s.drop = nn.Dropout(0.3)
        bi = kind.endswith("bigru")
        s.rnn = nn.GRU(d, H, batch_first=True, bidirectional=bi); s.head = nn.Linear(H * (2 if bi else 1), C)
    def forward(s, raw, cw, lab):
        p = []
        if s.use_raw: p.append(s.rp(raw.transpose(1, 2)))
        if s.use_cwt: p.append(s.cp(s.cnn(cw).permute(0, 3, 1, 2).flatten(2)))
        p.append(s.lp(lab.transpose(1, 2)))
        h = s.drop(torch.relu(torch.cat(p, 2))); h, _ = s.rnn(h); return s.head(s.drop(h))


def lab_input(known):
    """known: int array (-1 unknown) -> (C+1, T) one-hot + unknown flag."""
    o = np.zeros((C + 1, len(known)), np.float32); k = known >= 0
    o[known[k], np.where(k)[0]] = 1; o[C, ~k] = 1; return o


def train_ctx(kind, raw, cw, y_avail, val_known, val_mask, seed, T=128, iters=1500, bs=16, patience=15, min_iters=300):
    """y_avail: labels usable for training (-1 elsewhere). Synthetic gaps are cut from y_avail during training."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    cnt = np.bincount(y_avail[y_avail >= 0], minlength=C).astype(float)
    w = np.where(cnt > 0, 1 / np.sqrt(np.maximum(cnt, 1)), 0); w = torch.tensor(w / w[w > 0].mean(), dtype=torch.float32)
    net = CtxNet(kind, raw.shape[0], cw.shape[0]); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w, ignore_index=-1)
    R, Cw = torch.tensor(raw), torch.tensor(cw)
    best, state, wait, bit = -1, None, 0, 0
    for it in range(iters):
        net.train(); rb, cb, lb, tb = [], [], [], []
        for _ in range(bs):
            s0 = rng.integers(0, L - T); ya = y_avail[s0:s0 + T].copy(); known = ya.copy(); tgt = np.full(T, -1)
            covered = 0
            while covered < 0.3 * T:                                   # synthetic gaps, 1-32 ft
                g = int(np.exp(rng.uniform(np.log(2), np.log(64)))); a = rng.integers(0, T - 1); b = min(T, a + g)
                tgt[a:b] = ya[a:b]; known[a:b] = -1; covered += b - a
            rb.append(R[:, s0:s0 + T]); cb.append(Cw[..., s0:s0 + T]); lb.append(torch.tensor(lab_input(known))); tb.append(torch.tensor(tgt))
        rb, cb, lb, tb = map(torch.stack, (rb, cb, lb, tb))
        if (tb >= 0).sum() == 0: continue
        opt.zero_grad(); out = net(rb, cb, lb); loss = lossf(out.reshape(-1, C), tb.reshape(-1)); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if it % 10 == 9:
            p = predict(net, raw, cw, val_known)
            sc = accuracy_score(y[val_mask], p[val_mask]) + f1_score(y[val_mask], p[val_mask], average="macro")
            if sc > best: best, wait, bit, state = sc, 0, it + 1, {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience and it + 1 >= min_iters: break
    net.load_state_dict(state); net.eval(); net.best_it = bit; return net


def predict(net, raw, cw, known, proba=False):
    net.eval()
    with torch.no_grad():
        o = torch.softmax(net(torch.tensor(raw)[None], torch.tensor(cw)[None], torch.tensor(lab_input(known))[None])[0], -1).numpy()
    return o if proba else o.argmax(1)


def nearest_fill(known, idx):
    kn = np.where(known >= 0)[0]; pos = np.searchsorted(kn, idx)
    lo = kn[np.clip(pos - 1, 0, len(kn) - 1)]; hi = kn[np.clip(pos, 0, len(kn) - 1)]
    return known[np.where(np.abs(idx - lo) <= np.abs(hi - idx), lo, hi)]


def ctx_feats(known, rawT):
    """RF baseline features: logs + facies just above / below the gap + relative position in the gap."""
    kn = np.where(known >= 0)[0]; idx = np.arange(L); pos = np.searchsorted(kn, idx)
    lo = kn[np.clip(pos - 1, 0, len(kn) - 1)]; hi = kn[np.clip(pos, 0, len(kn) - 1)]
    oa = np.eye(C)[known[lo]]; ob = np.eye(C)[known[hi]]
    gl = np.maximum(hi - lo, 1); rel = (idx - lo) / gl
    return np.hstack([rawT, oa, ob, rel[:, None], np.log1p(gl)[:, None]])


MODELS = ["nearest_fill", "rf_logs", "rf_logs_ctx", "ctx_lab_bigru", "ctx_raw_bigru", "ctx_hyb_gru", "ctx_hyb_bigru"]
res = {"G_ft": GS, "offsets": NOFF, "models": MODELS, "rows": []}; t0 = time.time()
for G in GS:
    n = int(round(2 * G))
    for off in range(NOFF):
        pred = {m: np.full(L, -1) for m in MODELS}
        blk = ((np.arange(L) + off * max(1, n // NOFF)) // n) % 5
        for k in range(5):
            te = blk == k; va = blk == (k + 1) % 5
            y_tr = np.where(~te & ~va, y, -1)                         # training labels
            known_test = np.where(~te, y, -1)                         # context at test time: all but the test gaps
            known_val = np.where(~te & ~va, y, -1)
            vmask = va & (y >= 0); tmask = te & (y >= 0)
            raw, cw = build_features([X], [y_tr >= 0], [0, 1, 2, 3]); raw, cw = raw[0], cw[0]
            idx = np.where(te)[0]
            pred["nearest_fill"][idx] = nearest_fill(known_test, idx)
            rf = RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=off, n_jobs=1)
            trm = (y >= 0) & ~te
            pred["rf_logs"][te] = rf.fit(raw.T[trm], y[trm]).predict(raw.T[te])
            # RF with context: training rows come from the other folds, each seen as if it were the gap
            Ftr, Ytr = [], []
            for j in range(5):
                if j == k: continue
                kj = np.where(~te & (blk != j), y, -1); mj = (blk == j) & (y >= 0)
                Ftr.append(ctx_feats(kj, raw.T)[mj]); Ytr.append(y[mj])
            rf2 = RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=off, n_jobs=1)
            pred["rf_logs_ctx"][te] = rf2.fit(np.vstack(Ftr), np.concatenate(Ytr)).predict(ctx_feats(known_test, raw.T)[te])
            for m in MODELS[3:]:
                net = train_ctx(m, raw, cw, y_tr, known_val, vmask, 100 * off + k)
                pred[m][te] = predict(net, raw, cw, known_test)[te]
        ok = (blk >= 0) & (y >= 0)
        row = {"G": G, "off": off, **{m: {"acc": accuracy_score(y[ok], pred[m][ok]),
                                          "macroF1": f1_score(y[ok], pred[m][ok], average="macro")} for m in MODELS}}
        res["rows"].append(row)
        print(f"G={G:>4} ft off={off}  " + "  ".join(f"{m}={row[m]['acc']:.3f}" for m in MODELS) + f"   ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(OUT, "w"))
