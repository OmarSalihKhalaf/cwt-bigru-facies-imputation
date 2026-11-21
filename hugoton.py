"""Case 2 — Hugoton–Panoma (Council Grove Group): blind-well and leave-one-well-out tests.
Usage: python3 -I hugoton.py <repo_dir> <mode: blind|lowo|time> <n_seeds> <out_json> [models,comma,separated]
"""
import sys, json, time, warnings
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from core import build_features, train_seq, _predict, metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier
warnings.filterwarnings("ignore")

REPO, MODE, NSEED, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
LOGS = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]; GEO = ["NM_M", "RELPOS"]; FEATS = LOGS + GEO
C = 9
ADJ = {0: (1,), 1: (0, 2), 2: (1,), 3: (4,), 4: (3, 5), 5: (4, 6), 6: (5, 7), 7: (5, 6, 8), 8: (6, 7)}   # Hall (2016) adjacency
TRAIN = ["SHRIMPLIN", "SHANKLE", "LUKE G U", "CROSS H CATTLE", "NOLAN", "NEWBY", "CHURCHMAN BIBLE", "ALEXANDER D", "KIMZEY A"]
BLIND = ["STUART", "CRAWFORD"]

fv = pd.read_csv(f"{REPO}/facies_vectors.csv"); fv = fv[fv["Well Name"] != "Recruit F9"]
nf = pd.read_csv(f"{REPO}/nofacies_data.csv"); bl = pd.read_csv(f"{REPO}/blind_stuart_crawford_core_facies.csv")
bw = nf.merge(bl, left_on=["Well Name", "Depth"], right_on=["WellName", "Depth.ft"]).rename(columns={"LithCode": "Facies"})
D = {w: fv[fv["Well Name"] == w].sort_values("Depth").reset_index(drop=True) for w in TRAIN}
D.update({w: bw[bw["Well Name"] == w].sort_values("Depth").reset_index(drop=True) for w in BLIND})
WELLS = TRAIN + BLIND
Y = {w: np.where(D[w].Facies.values <= 9, D[w].Facies.values - 1, 99) for w in WELLS}   # code 11 -> 99 (never predicted)

MODELS_ALL = ["rf_point", "hgb_point", "rf_context", "rf_cwt", "raw_gru", "raw_bigru", "cwt_cnn_bigru", "hyb_cnn", "hyb_gru", "hyb_lstm", "hyb_bigru"]
MODELS = sys.argv[5].split(",") if len(sys.argv) > 5 else MODELS_ALL


def prepare(train_wells, eval_wells):
    """PE imputation fitted on training wells only; returns X dict (T,F)."""
    X = {w: D[w][FEATS].values.astype(float).copy() for w in train_wells + eval_wells}
    have = [w for w in train_wells if not np.isnan(X[w][:, 4]).any()]
    A = np.concatenate([X[w] for w in have])
    reg = RandomForestRegressor(200, min_samples_leaf=3, random_state=0, n_jobs=2).fit(A[:, [0, 1, 2, 3, 5, 6]], A[:, 4])
    for w in X:
        m = np.isnan(X[w][:, 4])
        if m.any(): X[w][m, 4] = reg.predict(X[w][m][:, [0, 1, 2, 3, 5, 6]])
    return X


def ctx(A, k=4):
    P = np.pad(A, ((k, k), (0, 0)), mode="edge"); return np.hstack([P[i:i + len(A)] for i in range(2 * k + 1)])


def run_split(train_wells, test_wells, seed, models):
    """Returns {model: {well: predictions}} and nn best-iteration info."""
    wl = train_wells + test_wells
    X = prepare(train_wells, test_wells)
    rng = np.random.default_rng(seed)
    val_well = train_wells[seed % len(train_wells)]                     # inner validation well for early stopping
    fit_w = [w for w in train_wells if w != val_well]
    trm = [np.full(len(X[w]), w in fit_w) for w in wl]
    vam = [np.full(len(X[w]), w == val_well) for w in wl]
    raw, cw = build_features([X[w] for w in wl], [np.full(len(X[w]), w in train_wells) for w in wl], log_idx=[0, 1, 2, 3, 4])
    y = [Y[w] for w in wl]
    P = {}; info = {}
    trX = np.concatenate([X[w] for w in train_wells]); trY = np.concatenate([Y[w] for w in train_wells])
    for m in models:
        P[m] = {}
        if m in ("rf_point", "hgb_point", "rf_context", "rf_cwt"):
            if m == "rf_context":
                A = {w: ctx(np.hstack([r.T for r in [raw[i]]])) for i, w in enumerate(wl)}
            elif m == "rf_cwt":
                A = {w: np.hstack([raw[i].T, cw[i].reshape(-1, cw[i].shape[-1]).T]) for i, w in enumerate(wl)}
            else:
                A = {w: raw[i].T for i, w in enumerate(wl)}
            Atr = np.concatenate([A[w] for w in train_wells])
            clf = (HistGradientBoostingClassifier(random_state=seed) if m == "hgb_point"
                   else RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=seed, n_jobs=2))
            clf.fit(Atr, trY)
            for w in test_wells: P[m][w] = clf.predict(A[w])
        else:
            t0 = time.time()
            net = train_seq(m, raw, cw, y, trm, vam, C, seed)
            info[m] = {"best_iter": net.best_it, "sec": round(time.time() - t0, 1)}
            for i, w in enumerate(wl):
                if w in test_wells: P[m][w] = _predict(net, raw[i], cw[i]).argmax(1)
    return P, info


def score(P, wells):
    yt = np.concatenate([Y[w] for w in wells]); yp = np.concatenate([P[w] for w in wells])
    return metrics(yt, yp, C, ADJ)


res = {"mode": MODE, "seeds": NSEED, "models": MODELS, "runs": []}
t0 = time.time()
if MODE == "time":
    P, info = run_split(TRAIN, BLIND, 0, MODELS); print(info, {m: score(P[m], BLIND)["acc"] for m in P}); sys.exit()
if MODE == "blind":
    for s in range(NSEED):
        P, info = run_split(TRAIN, BLIND, s, MODELS)
        run = {"seed": s, "info": info, "scores": {m: {"both": score(P[m], BLIND), **{w: score({w: P[m][w]}, [w]) for w in BLIND}} for m in MODELS},
               "pred": {m: {w: P[m][w].tolist() for w in BLIND} for m in MODELS}}
        res["runs"].append(run); print(f"seed {s} done {time.time()-t0:.0f}s", flush=True)
else:   # leave-one-well-out over the 9 training wells
    for s in range(NSEED):
        pred = {m: {} for m in MODELS}
        for k, held in enumerate(TRAIN):
            tw = [w for w in TRAIN if w != held]
            P, info = run_split(tw, [held], s * 10 + k, MODELS)
            for m in MODELS: pred[m][held] = P[m][held]
            print(f"seed {s} held {held} {time.time()-t0:.0f}s", flush=True)
        res["runs"].append({"seed": s, "scores": {m: {"pooled": score(pred[m], TRAIN), **{w: score({w: pred[m][w]}, [w]) for w in TRAIN}} for m in MODELS}})
json.dump(res, open(OUT, "w"))

key = "both" if MODE == "blind" else "pooled"
print(f"\n{MODE}: {NSEED} seeds — mean ± SD")
print(f"{'model':15s} {'accuracy':>14s} {'macro-F1':>14s} {'adjacent acc':>14s}")
for m in MODELS:
    a = [r["scores"][m][key]["acc"] for r in res["runs"]]; f = [r["scores"][m][key]["macroF1"] for r in res["runs"]]
    j = [r["scores"][m][key]["adjacentAcc"] for r in res["runs"]]
    print(f"{m:15s} {np.mean(a):.3f} ± {np.std(a):.3f}  {np.mean(f):.3f} ± {np.std(f):.3f}  {np.mean(j):.3f} ± {np.std(j):.3f}")
