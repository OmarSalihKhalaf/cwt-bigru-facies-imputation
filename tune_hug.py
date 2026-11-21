"""Model-design selection on the 9 TRAINING wells only (leave-one-well-out); blind wells untouched.
Usage: python3 -I tune_hug.py <repo> <out_json> <variant,...>
variants: rfctx | raw_bigru:<proto> | hyb_bigru:<mode>:<proto>   proto = p1 (1 val well, no min) | p2 (2 val wells, min 400 iters)
"""
import sys, json, time, warnings, numpy as np
sys.path.insert(0, ".")
import core
from core import build_features, train_seq, _predict, metrics
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
sys.argv_saved = list(sys.argv)
REPO, OUT, VARS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
sys.argv = [sys.argv[0], REPO, "none", "1", "/dev/null"]
import importlib.util
spec = importlib.util.spec_from_file_location("hg", "hugoton.py")
src = open("hugoton.py").read().split("res = {")[0]        # reuse data loading + helpers only
hg = {}; exec(compile(src, "hugoton_head", "exec"), hg)
TRAIN, Y, C, ADJ = hg["TRAIN"], hg["Y"], hg["C"], hg["ADJ"]
res = {}; t0 = time.time()
for v in VARS:
    parts = v.split(":"); pred = {}
    for k, held in enumerate(TRAIN):
        tw = [w for w in TRAIN if w != held]; wl = tw + [held]
        X = hg["prepare"](tw, [held])
        if parts[0] == "rfctx":
            core.CWT_MODE = "mag"
            raw, cw = build_features([X[w] for w in wl], [np.full(len(X[w]), w in tw) for w in wl], [0, 1, 2, 3, 4])
            A = [hg["ctx"](r.T) for r in raw]
            clf = RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=k, n_jobs=2)
            clf.fit(np.concatenate(A[:-1]), np.concatenate([Y[w] for w in tw])); pred[held] = clf.predict(A[-1])
            continue
        kind = parts[0]; core.CWT_MODE = parts[1] if kind.startswith("hyb") else "mag"; proto = parts[-1]
        raw, cw = build_features([X[w] for w in wl], [np.full(len(X[w]), w in tw) for w in wl], [0, 1, 2, 3, 4])
        nval = 1 if proto == "p1" else 2
        val = [tw[(k + j) % len(tw)] for j in range(nval)]
        trm = [np.full(len(X[w]), (w in tw) and (w not in val)) for w in wl]
        vam = [np.full(len(X[w]), w in val) for w in wl]
        net = train_seq(kind, raw, cw, [Y[w] for w in wl], trm, vam, C, k, min_iters=0 if proto == "p1" else 400)
        pred[held] = _predict(net, raw[-1], cw[-1]).argmax(1)
    yt = np.concatenate([Y[w] for w in TRAIN]); yp = np.concatenate([pred[w] for w in TRAIN])
    res[v] = {"pooled": metrics(yt, yp, C, ADJ), "perwell": {w: float(np.mean(pred[w] == Y[w])) for w in TRAIN}}
    print(f"{v:28s} LOWO acc {res[v]['pooled']['acc']:.3f}  macroF1 {res[v]['pooled']['macroF1']:.3f}  adj {res[v]['pooled']['adjacentAcc']:.3f}   ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(OUT, "w"))
