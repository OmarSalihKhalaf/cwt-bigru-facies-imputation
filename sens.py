"""CWT parameter sensitivity with a random forest on logs + CWT channels (fast, no network training).
Karpur: blocked CV (8-ft blocks, 4-sample guard, 5 folds x 3 repeats). Hugoton: train 9 wells -> blind STUART+CRAWFORD (3 seeds).
Usage: python3 -I sens.py <out_json>
"""
import sys, json, warnings, numpy as np, pandas as pd, pywt
sys.path.insert(0, ".")
import core
from core import build_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
OUT = sys.argv[1]

# Karpur
df = pd.read_csv("data/karpur.csv"); grid = np.arange(df.depth.min(), df.depth.max() + 0.25, 0.5); L = len(grid)
XK = np.stack([np.interp(grid, df.depth, df[c]) for c in ["caliper", "gamma", "density", "phi.N"]], 1)
cls = sorted(df.Facies.unique(), key=lambda s: int(s[1:])); lab = {round(d, 1): cls.index(f) for d, f in zip(df.depth, df.Facies)}
yK = np.array([lab.get(round(d, 1), -1) for d in grid])
# Hugoton
src = open("hugoton.py").read().split("res = {")[0]
argv0 = sys.argv; sys.argv = [sys.argv[0], "data/seg2016", "none", "1", "/dev/null"]; hg = {}; exec(compile(src, "hg", "exec"), hg); sys.argv = argv0
TR, BL = hg["TRAIN"], hg["BLIND"]; XH = hg["prepare"](TR, BL); YH = hg["Y"]


def set_cfg(mode, ns, B, amax):
    core.CWT_MODE = mode; core.NS = ns; core.SCALES = np.geomspace(2, amax, ns); core.WAVELET = f"cmor{B}-1.0"; core.PAD = int(amax)


def karpur_rf(cwt=True):
    accs, f1s = [], []
    for rep in range(3):
        rng = np.random.default_rng(100 + rep); nb = int(np.ceil(L / 16)); f = np.arange(nb) % 5; rng.shuffle(f); fold = f[np.arange(L) // 16]
        pred = np.full(L, -1)
        for k in range(5):
            near = np.convolve((fold == k).astype(float), np.ones(9), "same") > 0
            te = (fold == k) & (yK >= 0); tr = (fold != k) & (yK >= 0) & ~near
            raw, cw = build_features([XK], [tr], [0, 1, 2, 3]); A = np.hstack([raw[0].T, cw[0].reshape(-1, L).T]) if cwt else raw[0].T
            pred[te] = RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=rep, n_jobs=1).fit(A[tr], yK[tr]).predict(A[te])
        ok = yK >= 0; accs.append(accuracy_score(yK[ok], pred[ok])); f1s.append(f1_score(yK[ok], pred[ok], average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))


def hug_rf(cwt=True):
    wl = TR + BL
    raw, cw = build_features([XH[w] for w in wl], [np.full(len(XH[w]), w in TR) for w in wl], [0, 1, 2, 3, 4])
    A = {w: (np.hstack([raw[i].T, cw[i].reshape(-1, cw[i].shape[-1]).T]) if cwt else raw[i].T) for i, w in enumerate(wl)}
    accs, f1s = [], []
    yt = np.concatenate([YH[w] for w in BL])
    for s in range(3):
        m = RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=s, n_jobs=1)
        m.fit(np.concatenate([A[w] for w in TR]), np.concatenate([YH[w] for w in TR]))
        yp = np.concatenate([m.predict(A[w]) for w in BL]); accs.append(accuracy_score(yt, yp)); f1s.append(f1_score(yt, yp, average="macro", labels=list(range(9))))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))


rows = []
base = ("complex", 16, 1.5, 64)
configs = [("no CWT", None)] + [(f"{m}", (m, 16, 1.5, 64)) for m in ["mag", "complex"]] + \
          [(f"NS={n}", ("complex", n, 1.5, 64)) for n in [8, 32]] + \
          [(f"B={b} (omega0={2*np.pi*np.sqrt(b/2):.1f})", ("complex", 16, b, 64)) for b in [0.5, 3.0]] + \
          [(f"max scale={a} ({a*0.5:.0f} ft)", ("complex", 16, 1.5, a)) for a in [32, 128]]
for name, cfg in configs:
    if cfg is None: k = karpur_rf(False); h = hug_rf(False)
    else: set_cfg(*cfg); k = karpur_rf(); h = hug_rf()
    rows.append({"config": name, "karpur": k, "hugoton": h})
    print(f"{name:32s} Karpur acc {k[0]:.3f}±{k[1]:.3f} mF1 {k[2]:.3f} | Hugoton blind acc {h[0]:.3f}±{h[1]:.3f} mF1 {h[2]:.3f}", flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
