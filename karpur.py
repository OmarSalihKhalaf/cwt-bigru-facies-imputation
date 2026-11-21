"""Case 1 — Karpur dataset (single well): leakage-free blocked cross-validation.
Usage: python3 -I karpur.py <csv> <block> <n_repeats> <out_json> [models]
Depth blocks of <block> samples are assigned to 5 folds; a guard of 4 samples on each side of every
test block is removed from training and validation; the next fold is the inner validation set.
"""
import sys, json, time, warnings
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from core import build_features, train_seq, _predict, metrics
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
warnings.filterwarnings("ignore")

CSV, BLOCK, NREP, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
LOGS = ["caliper", "gamma", "density", "phi.N"]; K, GAP = 5, 4
MODELS_ALL = ["rf_point", "hgb_point", "rf_context", "rf_cwt", "raw_gru", "raw_bigru", "cwt_cnn_bigru", "hyb_cnn", "hyb_gru", "hyb_lstm", "hyb_bigru"]
MODELS = sys.argv[5].split(",") if len(sys.argv) > 5 else MODELS_ALL

df = pd.read_csv(CSV)
grid = np.arange(df.depth.min(), df.depth.max() + 0.25, 0.5); L = len(grid)
X = np.stack([np.interp(grid, df.depth, df[c]) for c in LOGS], 1)      # 14 one-foot gaps linearly filled (inputs only)
classes = sorted(df.Facies.unique(), key=lambda s: int(s[1:])); C = len(classes)
lab = {round(d, 1): classes.index(f) for d, f in zip(df.depth, df.Facies)}
y = np.array([lab.get(round(d, 1), -1) for d in grid])                  # -1 = gap level, never scored


def folds(seed):
    rng = np.random.default_rng(seed); nb = int(np.ceil(L / BLOCK)); f = np.arange(nb) % K; rng.shuffle(f)
    return f[np.arange(L) // BLOCK]


def ctx(A, k=4):
    P = np.pad(A, ((k, k), (0, 0)), mode="edge"); return np.hstack([P[i:i + len(A)] for i in range(2 * k + 1)])


res = {"block": BLOCK, "gap": GAP, "folds": K, "repeats": NREP, "classes": classes, "models": MODELS, "runs": []}
t0 = time.time()
for rep in range(NREP):
    fold = folds(100 + rep); pred = {m: np.full(L, -1) for m in MODELS}; info = {m: [] for m in MODELS}
    for k in range(K):
        near = np.convolve((fold == k).astype(float), np.ones(2 * GAP + 1), "same") > 0
        te = (fold == k) & (y >= 0)
        va = (fold == (k + 1) % K) & (y >= 0) & ~near
        tr = (fold != k) & (fold != (k + 1) % K) & (y >= 0) & ~near
        trva = tr | va
        raw, cw = build_features([X], [tr], log_idx=[0, 1, 2, 3]); raw, cw = raw[0], cw[0]
        A = {"rf_point": raw.T, "hgb_point": raw.T, "rf_context": ctx(raw.T),
             "rf_cwt": np.hstack([raw.T, cw.reshape(-1, L).T])}
        for m in MODELS:
            if m in A:
                clf = (HistGradientBoostingClassifier(random_state=rep) if m == "hgb_point" else
                       RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=rep, n_jobs=2))
                pred[m][te] = clf.fit(A[m][trva], y[trva]).predict(A[m][te])
            else:
                net = train_seq(m, [raw], [cw], [y], [tr], [va], C, 1000 * rep + k)
                pred[m][te] = _predict(net, raw, cw).argmax(1)[te]; info[m].append(net.best_it)
    ok = y >= 0
    res["runs"].append({"rep": rep, "scores": {m: metrics(y[ok], pred[m][ok], C) for m in MODELS},
                        "best_iter": info, "pred": {m: pred[m].tolist() for m in MODELS}})
    print(f"repeat {rep} done {time.time()-t0:.0f}s", flush=True)
res["y"] = y.tolist(); res["depth"] = grid.tolist()
json.dump(res, open(OUT, "w"))
print(f"\nKarpur, block={BLOCK} ({BLOCK*0.5:.0f} ft), guard={GAP}, {K}-fold x {NREP}")
print(f"{'model':15s} {'accuracy':>14s} {'macro-F1':>14s}")
for m in MODELS:
    a = [r["scores"][m]["acc"] for r in res["runs"]]; f = [r["scores"][m]["macroF1"] for r in res["runs"]]
    print(f"{m:15s} {np.mean(a):.3f} ± {np.std(a):.3f}  {np.mean(f):.3f} ± {np.std(f):.3f}")
