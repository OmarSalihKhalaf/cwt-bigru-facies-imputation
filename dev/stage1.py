"""Stage 1: leakage-free blocked-CV benchmark on the single-well dataset.
Per-depth (sequence-to-sequence) facies labelling; held-out depth blocks have labels hidden.
Usage: python3 -I stage1.py <csv> <block_size> <n_repeats> <out_json>
"""
import sys, json, time, warnings
import numpy as np, pandas as pd, pywt, torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

CSV, BLOCK, NREP, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
LOGS = ["caliper", "gamma", "density", "phi.N"]
K, GAP, NS = 5, 4, 24

# ---------------- data on a regular 0.5-ft grid ----------------
df = pd.read_csv(CSV)
grid = np.arange(df.depth.min(), df.depth.max() + 0.25, 0.5)
L = len(grid)
X = np.stack([np.interp(grid, df.depth, df[c]) for c in LOGS], 1)          # (L,4), 14 levels interpolated
classes = sorted(df.Facies.unique(), key=lambda s: int(s[1:]))
lab = {round(d, 1): classes.index(f) for d, f in zip(df.depth, df.Facies)}
y = np.array([lab.get(round(d, 1), -1) for d in grid])                      # -1 = no label (gap level)
C = len(classes)

def folds(seed):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(L / BLOCK)); f = np.arange(nb) % K; rng.shuffle(f)
    return f[np.arange(L) // BLOCK]

def masks(fold, k):
    te = (fold == k) & (y >= 0)
    near = np.convolve((fold == k).astype(float), np.ones(2 * GAP + 1), "same") > 0   # guard zone
    va = (fold == (k + 1) % K) & (y >= 0) & ~near
    tr = (fold != k) & (fold != (k + 1) % K) & (y >= 0) & ~near
    return tr, va, te

def zfit(A, m):  # standardise with statistics from training depths only
    mu, sd = A[m].mean(0), A[m].std(0) + 1e-8
    return (A - mu) / sd

def cwt_feats(Xz, tr):
    scales = np.geomspace(2, 64, NS); pad = 64; out = []
    for j in range(Xz.shape[1]):
        s = np.pad(Xz[:, j], pad, mode="reflect")
        co, _ = pywt.cwt(s, scales, "cmor1.5-1.0")
        out.append(np.log1p(np.abs(co[:, pad:-pad])))                      # (NS, L) magnitude
    F = np.stack(out, 0)                                                    # (4, NS, L)
    mu = F[:, :, tr].mean(2, keepdims=True); sd = F[:, :, tr].std(2, keepdims=True) + 1e-8
    return ((F - mu) / sd).astype(np.float32)

# ---------------- models ----------------
class Net(nn.Module):
    def __init__(s, kind):
        super().__init__(); s.kind = kind; H = 64
        if kind in ("raw_bigru", "raw_gru"):
            s.inp = nn.Linear(4, H)
        elif kind == "hyb_bigru":
            s.cnn = nn.Sequential(
                nn.Conv2d(4, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d((4, 1)),
                nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d((2, None)))
            s.cproj = nn.Sequential(nn.Dropout(0.5), nn.Linear(16, 16)); s.rproj = nn.Linear(4, H - 16)
        elif kind == "cwt_mlp":
            s.inp = nn.Sequential(nn.Linear(4 * NS, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, H))
        else:
            s.cnn = nn.Sequential(
                nn.Conv2d(4, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 1)))
            s.inp = nn.Linear(32 * (NS // 4), H)
        bi = kind in ("raw_bigru", "cwt_cnn_bigru", "hyb_bigru")
        s.rnn = nn.GRU(H, H, batch_first=True, bidirectional=bi) if (bi or kind in ("cwt_cnn_gru", "raw_gru")) else None
        s.drop = nn.Dropout(0.3)
        s.head = nn.Linear(H * (2 if bi else 1), C)
    def forward(s, raw, cw):                       # raw (B,T,4); cw (B,4,NS,T)
        if s.kind in ("raw_bigru", "raw_gru"): h = s.inp(raw)
        elif s.kind == "hyb_bigru": h = torch.cat([s.rproj(raw), s.cproj(s.cnn(cw).permute(0, 3, 1, 2).flatten(2))], 2)
        elif s.kind == "cwt_mlp": h = s.inp(cw.permute(0, 3, 1, 2).flatten(2))
        else: h = s.inp(s.cnn(cw).permute(0, 3, 1, 2).flatten(2))
        h = s.drop(torch.relu(h))
        if s.rnn is not None: h, _ = s.rnn(h); h = s.drop(h)
        return s.head(h)

def train_nn(kind, Xz, F, tr, va, te, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    raw = torch.tensor(Xz, dtype=torch.float32)[None]; cw = torch.tensor(F)[None]
    yt = torch.tensor(np.where(tr, y, -1)); yv = np.where(va)[0]
    cnt = np.bincount(y[tr], minlength=C).astype(float)
    w = torch.tensor(np.where(cnt > 0, 1 / np.sqrt(np.maximum(cnt, 1)), 0), dtype=torch.float32); w = w / w[w > 0].mean()
    net = Net(kind); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w, ignore_index=-1)
    best, bstate, wait, T = -1, None, 0, 192
    for ep in range(200):
        net.train(); st = rng.integers(0, L - T, 8)
        rb = torch.cat([raw[:, s:s + T] for s in st]); cb = torch.cat([cw[:, :, :, s:s + T] for s in st])
        yb = torch.stack([yt[s:s + T] for s in st])
        if (yb >= 0).sum() == 0: continue
        opt.zero_grad(); out = net(rb, cb); loss = lossf(out.reshape(-1, C), yb.reshape(-1)); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad(): p = net(raw, cw)[0].argmax(1).numpy()
        sc = f1_score(y[yv], p[yv], average="macro") + accuracy_score(y[yv], p[yv])
        if sc > best: best, wait, bstate, bep = sc, 0, {k: v.clone() for k, v in net.state_dict().items()}, ep
        else:
            wait += 1
            if wait >= 30: break
    net.load_state_dict(bstate); net.eval()
    with torch.no_grad(): p = net(raw, cw)[0].argmax(1).numpy()
    return p, bep

def ctx(A, k=4):
    P = np.pad(A, ((k, k), (0, 0)), mode="edge")
    return np.hstack([P[i:i + len(A)] for i in range(2 * k + 1)])

NN = ["raw_gru", "raw_bigru", "hyb_bigru", "cwt_mlp", "cwt_cnn", "cwt_cnn_gru", "cwt_cnn_bigru"]
SK = ["rf_point", "hgb_point", "rf_context9", "rf_cwt_point"]
if len(sys.argv) > 5:
    sel = sys.argv[5].split(","); NN = [m for m in NN if m in sel]; SK = [m for m in SK if m in sel]
res = {m: [] for m in SK + NN}; perclass = {m: [] for m in SK + NN}; epochs = {m: [] for m in NN}
t0 = time.time()
for rep in range(NREP):
    fold = folds(100 + rep); pred = {m: np.full(L, -1) for m in SK + NN}
    for k in range(K):
        tr, va, te = masks(fold, k); trva = tr | va
        Xz = zfit(X, tr); F = cwt_feats(Xz, tr); Fp = F.reshape(4 * NS, L).T
        for m, A in [("rf_point", Xz), ("rf_context9", ctx(Xz)), ("rf_cwt_point", Fp)]:
            if m not in SK: continue
            pred[m][te] = RandomForestClassifier(300, class_weight="balanced_subsample", random_state=rep, n_jobs=2).fit(A[trva], y[trva]).predict(A[te])
        if "hgb_point" in SK: pred["hgb_point"][te] = HistGradientBoostingClassifier(random_state=rep).fit(Xz[trva], y[trva]).predict(Xz[te])
        for m in NN:
            p, bep = train_nn(m, Xz, F, tr, va, te, 1000 * rep + k); pred[m][te] = p[te]; epochs[m].append(bep)
    ok = y >= 0
    for m in SK + NN:
        res[m].append([accuracy_score(y[ok], pred[m][ok]), f1_score(y[ok], pred[m][ok], average="macro"),
                       f1_score(y[ok], pred[m][ok], average="weighted")])
        perclass[m].append(f1_score(y[ok], pred[m][ok], average=None, labels=list(range(C))).tolist())
    print(f"repeat {rep} done, {time.time()-t0:.0f}s", flush=True)

summ = {m: {"acc": [float(np.mean([r[0] for r in v])), float(np.std([r[0] for r in v]))],
            "macroF1": [float(np.mean([r[1] for r in v])), float(np.std([r[1] for r in v]))],
            "weightedF1": [float(np.mean([r[2] for r in v])), float(np.std([r[2] for r in v]))],
            "perclassF1": dict(zip(classes, np.mean(perclass[m], 0).round(3).tolist())),
            "mean_best_epoch": float(np.mean(epochs[m])) if m in epochs else None} for m, v in res.items()}
json.dump({"block": BLOCK, "gap": GAP, "folds": K, "repeats": NREP, "L": L, "labelled": int((y >= 0).sum()),
           "classes": classes, "summary": summ, "raw": res}, open(OUT, "w"), indent=1)
print(f"\nblock={BLOCK} samples ({BLOCK*0.5:.0f} ft), guard={GAP}, {K}-fold x {NREP} repeats; pooled out-of-fold predictions")
print(f"{'model':16s} {'accuracy':>14s} {'macro-F1':>14s} {'weighted-F1':>14s}")
for m, s in summ.items():
    print(f"{m:16s} {s['acc'][0]:.3f} ± {s['acc'][1]:.3f}  {s['macroF1'][0]:.3f} ± {s['macroF1'][1]:.3f}  {s['weightedF1'][0]:.3f} ± {s['weightedF1'][1]:.3f}")
print("\nper-class F1 (mean over repeats):")
print(pd.DataFrame({m: s["perclassF1"] for m, s in summ.items()}).T.to_string())
