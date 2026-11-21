"""Facies imputation v2 — training gaps drawn with the same geometry as the test gaps.

Scenario (both cases): core facies are known everywhere in a well except missing intervals of exactly G ft.
  Karpur  : test gaps every 5G ft (fold k = blocks with index % 5 == k); the model is trained on the same well,
            with synthetic G-ft gaps cut only from known (non-test) intervals; test labels never seen.
  Hugoton : the model is trained on the 9 training wells with synthetic G-ft gaps; test gaps every 5G ft in the
            blind wells STUART and CRAWFORD, whose remaining facies serve as the known context.
Validation for early stopping: a fixed set of synthetic gaps whose labels are never used as training targets.

Usage: python3 -I impute2.py <case: karpur|hugoton> <G list> <n_offsets> <n_seeds> <out_json> [models]
"""
import sys, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, ".")
import core
from core import build_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
core.CWT_MODE = "complex"
CASE, GS, NOFF, NSEED, OUT = sys.argv[1], [float(g) for g in sys.argv[2].split(",")], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
MODELS = sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else ["nearest_fill", "rf_logs_ctx", "ctx_lab_bigru", "ctx_raw_bigru", "ctx_hyb_gru", "ctx_hyb_bigru"]

# ------------------------------------------------------------------ data as list of sequences
if CASE == "karpur":
    df = pd.read_csv("data/karpur.csv")
    grid = np.arange(df.depth.min(), df.depth.max() + 0.25, 0.5)
    Xs = [np.stack([np.interp(grid, df.depth, df[c]) for c in ["caliper", "gamma", "density", "phi.N"]], 1)]
    classes = sorted(df.Facies.unique(), key=lambda s: int(s[1:])); C = len(classes)
    lab = {round(d, 1): classes.index(f) for d, f in zip(df.depth, df.Facies)}
    Ys = [np.array([lab.get(round(d, 1), -1) for d in grid])]
    names = ["KARPUR"]; train_ids = [0]; test_ids = [0]; LOG_IDX = [0, 1, 2, 3]
else:
    src = open("hugoton.py").read().split("res = {")[0]
    sys.argv = [sys.argv[0], "data/seg2016", "none", "1", "/dev/null"]; hg = {}; exec(compile(src, "hg", "exec"), hg)
    names = hg["TRAIN"] + hg["BLIND"]; Xp = hg["prepare"](hg["TRAIN"], hg["BLIND"])
    Xs = [Xp[w] for w in names]; Ys = [np.where(hg["Y"][w] == 99, -1, hg["Y"][w]) for w in names]   # code 11 unscored
    C = 9; train_ids = list(range(9)); test_ids = [9, 10]; LOG_IDX = [0, 1, 2, 3, 4]
NSEQ = len(Xs)


class CtxNet(nn.Module):
    def __init__(s, kind, n_raw, n_cwt, H=64):
        super().__init__(); d = 0
        s.use_raw = kind != "ctx_lab_bigru"; s.use_cwt = "hyb" in kind
        if s.use_raw: s.rp = nn.Linear(n_raw, 32); d += 32
        if s.use_cwt:
            s.cnn = nn.Sequential(nn.Conv2d(n_cwt, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)),
                                  nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)))
            s.cp = nn.Sequential(nn.Dropout(0.5), nn.Linear(16 * (2 * core.NS // 4), 32)); d += 32
        s.lp = nn.Linear(2 * C + 3, 32); d += 32
        s.drop = nn.Dropout(0.3); bi = kind.endswith("bigru")
        s.rnn = nn.GRU(d, H, batch_first=True, bidirectional=bi); s.head = nn.Linear(H * (2 if bi else 1), C)
        s.alpha = nn.Parameter(torch.tensor(4.0))          # weight of the nearest-known-facies prior (skip connection)
    def forward(s, raw, cw, lab):
        p = []
        if s.use_raw: p.append(s.rp(raw.transpose(1, 2)))
        if s.use_cwt: p.append(s.cp(s.cnn(cw).permute(0, 3, 1, 2).flatten(2)))
        h = torch.relu(torch.cat(p + [s.lp(lab.transpose(1, 2))], 2)) if p else torch.relu(s.lp(lab.transpose(1, 2)))
        if p:                                      # dropout on log/CWT paths only; bounding-facies path kept intact
            nf = sum(x.shape[2] for x in p); h = torch.cat([s.drop(h[..., :nf]), h[..., nf:]], 2)
        h, _ = s.rnn(h)
        L_ = lab.transpose(1, 2); above, below, rel = L_[..., :C], L_[..., C:2 * C], L_[..., 2 * C:2 * C + 1]
        prior = torch.where(rel <= 0.5, above, below)      # nearest known facies (copy rule) as a one-hot prior
        return s.head(s.drop(h)) + s.alpha * prior


def lab_input(known):
    """per depth: one-hot of the facies directly above and directly below the nearest known samples,
    relative position between them, gap length (log), and an unknown flag  -> (2C+3, T)."""
    L = len(known); kn = np.where(known >= 0)[0]; idx = np.arange(L)
    if len(kn) == 0: return np.zeros((2 * C + 3, L), np.float32)
    pos = np.searchsorted(kn, idx, side="right")
    lo = kn[np.clip(pos - 1, 0, len(kn) - 1)]; hi = kn[np.clip(np.searchsorted(kn, idx), 0, len(kn) - 1)]
    o = np.zeros((2 * C + 3, L), np.float32)
    o[known[lo], idx] = 1; o[C + known[hi], idx] = 1
    gl = np.maximum(hi - lo, 1); o[2 * C] = np.clip((idx - lo) / gl, 0, 1); o[2 * C + 1] = np.log1p(gl) / 5; o[2 * C + 2] = known < 0
    return o


def place_gaps(known, n, rng, frac=0.2, max_tries=2000):
    """random non-overlapping gaps of n samples lying fully inside known samples, flanked by known samples."""
    L = len(known); g = np.zeros(L, bool); tries = 0
    while g.sum() < frac * (known >= 0).sum() and tries < max_tries:
        tries += 1; a = rng.integers(1, max(2, L - n - 1)); b = a + n
        if b >= L: continue
        if (known[a - 1:b + 1] >= 0).all() and not g[max(0, a - n):b + n].any(): g[a:b] = True
    return g


def train_ctx(kind, raws, cws, knowns, n, val_gaps, seed, T=None, iters=1500, bs=16, patience=15, min_iters=300):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    T = int(max(128, 4 * n)); T = min(T, min(len(k) for k in knowns) - 2)
    allk = np.concatenate([k[k >= 0] for k in knowns])
    cnt = np.bincount(allk, minlength=C).astype(float)
    w = np.where(cnt > 0, 1 / np.sqrt(np.maximum(cnt, 1)), 0); w = torch.tensor(w / w[w > 0].mean(), dtype=torch.float32)
    net = CtxNet(kind, raws[0].shape[0], cws[0].shape[0]); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)               # imputation targets accuracy: unweighted loss
    R = [torch.tensor(r) for r in raws]; Cw = [torch.tensor(c) for c in cws]
    lens = np.array([len(k) for k in knowns], float)
    best, state, wait, bit = -1, None, 0, 0
    for it in range(iters):
        net.train(); rb, cb, lb, tb = [], [], [], []
        for _ in range(bs):
            i = rng.choice(len(knowns), p=lens / lens.sum()); s0 = rng.integers(0, len(knowns[i]) - T)
            kn = knowns[i][s0:s0 + T].copy(); gaps = place_gaps(kn, n, rng, frac=0.25, max_tries=40)
            tgt = np.where(gaps & ~val_gaps[i][s0:s0 + T], kn, -1); kin = np.where(gaps | val_gaps[i][s0:s0 + T], -1, kn)
            rb.append(R[i][:, s0:s0 + T]); cb.append(Cw[i][..., s0:s0 + T]); lb.append(torch.tensor(lab_input(kin))); tb.append(torch.tensor(tgt))
        rb, cb, lb, tb = map(torch.stack, (rb, cb, lb, tb))
        if (tb >= 0).sum() == 0: continue
        opt.zero_grad(); loss = lossf(net(rb, cb, lb).reshape(-1, C), tb.reshape(-1)); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if it % 10 == 9:
            yt, yp = [], []
            for i in range(len(knowns)):
                if not val_gaps[i].any(): continue
                p = predict(net, raws[i], cws[i], np.where(val_gaps[i], -1, knowns[i]))
                yt.append(knowns[i][val_gaps[i]]); yp.append(p[val_gaps[i]])
            yt, yp = np.concatenate(yt), np.concatenate(yp)
            sc = accuracy_score(yt, yp) + f1_score(yt, yp, average="macro")
            if sc > best: best, wait, bit, state = sc, 0, it + 1, {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience and it + 1 >= min_iters: break
    net.load_state_dict(state); net.eval(); net.best_it = bit; return net


def predict(net, raw, cw, known):
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(raw)[None], torch.tensor(cw)[None], torch.tensor(lab_input(known))[None])[0].argmax(1).numpy()


def nearest_fill(known, idx):
    kn = np.where(known >= 0)[0]; pos = np.searchsorted(kn, idx)
    lo = kn[np.clip(pos - 1, 0, len(kn) - 1)]; hi = kn[np.clip(pos, 0, len(kn) - 1)]
    return known[np.where(np.abs(idx - lo) <= np.abs(hi - idx), lo, hi)]


def ctx_feats(known, rawT):
    L = len(known); kn = np.where(known >= 0)[0]; idx = np.arange(L); pos = np.searchsorted(kn, idx)
    lo = kn[np.clip(pos - 1, 0, len(kn) - 1)]; hi = kn[np.clip(pos, 0, len(kn) - 1)]
    oa = np.eye(C)[known[lo]]; ob = np.eye(C)[known[hi]]; gl = np.maximum(hi - lo, 1); rel = (idx - lo) / gl
    return np.hstack([rawT, oa, ob, rel[:, None], np.log1p(gl)[:, None]])


import os
OFF0 = int(os.environ.get("OFF0", "0"))
res = {"case": CASE, "G_ft": GS, "models": MODELS, "rows": []}; t0 = time.time()
for G in GS:
    n = int(round(2 * G))
    for off in range(OFF0, OFF0 + NOFF):
        nfold = 5; net_cache = {}
        preds = {m: [np.full(len(Ys[i]), -1) for i in range(NSEQ)] for m in MODELS}
        testmask = [np.zeros(len(Ys[i]), bool) for i in range(NSEQ)]
        for k in range(nfold):
            # test gaps
            tm = [np.zeros(len(Ys[i]), bool) for i in range(NSEQ)]
            for i in test_ids:
                blk = ((np.arange(len(Ys[i])) + off * max(1, n // max(NOFF, 1))) // n) % 5
                tm[i] = (blk == k) & (np.arange(len(Ys[i])) >= n) & (np.arange(len(Ys[i])) < len(Ys[i]) - n)
                testmask[i] |= tm[i]
            # what the model may learn from: Karpur -> same well minus test gaps; Hugoton -> training wells only
            fit_ids = train_ids
            knowns_fit = [np.where(tm[i], -1, Ys[i]) for i in fit_ids]
            rng = np.random.default_rng(1000 + off * 10 + k)
            val_gaps = [place_gaps(kk, n, rng, frac=0.15) for kk in knowns_fit]
            tr_samples = [(kk >= 0) for kk in knowns_fit]
            raws, cws = build_features([Xs[i] for i in range(NSEQ)], [np.isin(i, fit_ids) & (np.where(tm[i], -1, Ys[i]) >= 0) for i in range(NSEQ)], LOG_IDX)
            # RF with bounding-facies features: rows built from synthetic gaps in the fitting data
            Ftr, Ytr = [], []
            for rep in range(3):
                for j, i in enumerate(fit_ids):
                    g = place_gaps(knowns_fit[j], n, rng, frac=0.25)
                    Ftr.append(ctx_feats(np.where(g, -1, knowns_fit[j]), raws[i].T)[g]); Ytr.append(knowns_fit[j][g])
            rf = RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=off, n_jobs=1).fit(np.vstack(Ftr), np.concatenate(Ytr))
            for i in test_ids:
                kn_test = np.where(tm[i], -1, Ys[i]); idx = np.where(tm[i])[0]
                if len(idx) == 0: continue
                if "nearest_fill" in preds: preds["nearest_fill"][i][idx] = nearest_fill(kn_test, idx)
                if "rf_logs_ctx" in preds: preds["rf_logs_ctx"][i][idx] = rf.predict(ctx_feats(kn_test, raws[i].T)[idx])
            for m in MODELS:
                if not m.startswith("ctx"): continue
                for sd in range(NSEED):
                    tt = time.time()
                    key = (m, sd)
                    if CASE == "hugoton" and key in net_cache: net = net_cache[key]
                    else:
                        net = train_ctx(m, [raws[i] for i in fit_ids], [cws[i] for i in fit_ids], knowns_fit, n, val_gaps, 7 * sd + 100 * off + (k if CASE == "karpur" else 0))
                        if CASE == "hugoton": net_cache[key] = net
                    print(f"   fold {k} {m} seed {sd}: best_it {net.best_it}, {time.time()-tt:.0f}s", file=sys.stderr, flush=True)
                    for i in test_ids:
                        p = predict(net, raws[i], cws[i], np.where(tm[i], -1, Ys[i]))
                        preds.setdefault(m + "#votes", {}).setdefault(i, []).append(p)
                    if m == "ctx_hyb_bigru" and sd == 0:                      # permutation (depth-shift) importance per log
                        rs = np.random.default_rng(5)
                        for j in range(len(LOG_IDX)):
                            for i in test_ids:
                                kn_t = np.where(tm[i], -1, Ys[i]); msk = tm[i] & (Ys[i] >= 0)
                                if msk.sum() == 0: continue
                                if not msk.any(): continue
                                base = np.mean(predict(net, raws[i], cws[i], kn_t)[msk] == Ys[i][msk])
                                sh = int(rs.integers(len(Ys[i]) // 4, 3 * len(Ys[i]) // 4))
                                r2 = raws[i].copy(); c2 = cws[i].copy()
                                r2[LOG_IDX[j]] = np.roll(r2[LOG_IDX[j]], sh); c2[j] = np.roll(c2[j], sh, axis=-1)
                                drop = base - np.mean(predict(net, r2, c2, kn_t)[msk] == Ys[i][msk])
                                res.setdefault("perm_importance", []).append({"G": G, "off": off, "fold": k, "well": names[i], "log": j, "drop": float(drop), "n": int(msk.sum())})
        row = {"G": G, "off": off}
        for m in MODELS:
            yt = np.concatenate([Ys[i][testmask[i] & (Ys[i] >= 0)] for i in test_ids])
            if m.startswith("ctx"):
                accs = [];
                for sd in range(NSEED):
                    yp = []
                    for i in test_ids:
                        P = np.zeros(len(Ys[i]), int)
                        for kk, pv in enumerate(preds[m + "#votes"][i][sd::NSEED]): pass
                        yp.append(None)
                    accs.append(None)
                # combine: each fold contributed NSEED predictions per test well in order (fold-major, seed-minor)
                per_seed = []
                for sd in range(NSEED):
                    yp = []
                    for i in test_ids:
                        P = np.full(len(Ys[i]), -1)
                        votes = preds[m + "#votes"][i]
                        for f in range(nfold):
                            pv = votes[f * NSEED + sd]
                            blk = ((np.arange(len(Ys[i])) + off * max(1, n // max(NOFF, 1))) // n) % 5
                            fm = (blk == f) & testmask[i]
                            P[fm] = pv[fm]
                        yp.append(P[testmask[i] & (Ys[i] >= 0)])
                    per_seed.append(np.concatenate(yp))
                accs = [accuracy_score(yt, yp) for yp in per_seed]
                f1s = [f1_score(yt, yp, average="macro") for yp in per_seed]
                vote = np.array([np.bincount(c, minlength=C).argmax() for c in np.stack(per_seed).T])
                row[m] = {"acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)), "macroF1": float(np.mean(f1s)),
                          "acc_vote": accuracy_score(yt, vote), "macroF1_vote": f1_score(yt, vote, average="macro"),
                          "pred_seeds": [p.tolist() for p in per_seed]}
            else:
                yp = np.concatenate([preds[m][i][testmask[i] & (Ys[i] >= 0)] for i in test_ids])
                row[m] = {"acc": accuracy_score(yt, yp), "macroF1": f1_score(yt, yp, average="macro"), "pred": yp.tolist()}
        row["n_test"] = int(len(yt)); row["y_true"] = yt.tolist(); res["rows"].append(row)
        json.dump(res, open(OUT, "w"))
        print(f"{CASE} G={G:>4} ft off={off} n={len(yt)}  " + "  ".join(
            f"{m}={row[m]['acc']:.3f}" + (f"(vote {row[m]['acc_vote']:.3f})" if 'acc_vote' in row[m] else "") for m in MODELS) + f"  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(OUT, "w"))
