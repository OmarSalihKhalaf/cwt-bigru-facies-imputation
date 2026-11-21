"""Re-implementation of the submitted CWT notebook (window-majority CNN) and the same model under a
non-overlapping depth split. Purpose: show where the 94% comes from.
Usage: python3 -I reproduce_original.py <karpur.csv> <n_seeds> [out_json]
"""
import sys, numpy as np, pandas as pd, pywt, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.set_num_threads(2)
CSV, NS = sys.argv[1], int(sys.argv[2])
df = pd.read_csv(CSV)
X = StandardScaler().fit_transform(df[["caliper", "gamma", "density", "phi.N"]].values)     # as in notebook: whole well
cw = np.stack([np.abs(pywt.cwt(X[:, j], np.arange(1, 64), "morl")[0]) for j in range(4)], 0)  # (4, 63, N)
cw = (cw - cw.mean()) / cw.std()
classes = sorted(df.Facies.unique()); yall = np.array([classes.index(f) for f in df.Facies])
W, STEP = 128, 8
starts = np.arange(0, len(df) - W, STEP)
Xw = np.stack([cw[:, :, s:s + W] for s in starts]).astype(np.float32)
yw = np.array([np.bincount(yall[s:s + W], minlength=len(classes)).argmax() for s in starts])
print(f"{len(starts)} windows; window-majority classes present: {sorted(set(classes[i] for i in yw))}")


class CNN(nn.Module):
    def __init__(s, C):
        super().__init__()
        blk = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
        s.f = nn.Sequential(*blk(4, 32), *blk(32, 64), *blk(64, 128), nn.AdaptiveAvgPool2d((8, 8)))
        s.h = nn.Sequential(nn.Flatten(), nn.Linear(128 * 64, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, C))
    def forward(s, x): return s.h(s.f(x))


def fit_eval(tr, te, seed):
    torch.manual_seed(seed); net = CNN(len(classes)); opt = torch.optim.Adam(net.parameters(), 1e-3)
    xt, ytt = torch.tensor(Xw[tr]), torch.tensor(yw[tr]); rng = np.random.default_rng(seed); best = 0
    for ep in range(20):
        net.train()
        for b in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // 32)):
            opt.zero_grad(); nn.functional.cross_entropy(net(xt[b]), ytt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): acc = (net(torch.tensor(Xw[te])).argmax(1).numpy() == yw[te]).mean()
        best = max(best, acc)                                # notebook kept the best epoch on this same set
    return best, len(te)


idx = np.arange(len(starts)); OUTJ = {"random_split": [], "depth_block_split": [], "windows": int(len(starts))}
for name in ["random split (as submitted)", "depth-block split, no window overlap"]:
    accs = []
    for s in range(NS):
        if name.startswith("random"):
            tr, te = train_test_split(idx, test_size=0.2, random_state=42 + s, stratify=yw)
        else:
            k = len(idx) // 5; b = s % 5                       # test = one contiguous fifth of the well
            te = idx[b * k:(b + 1) * k]
            lo, hi = starts[te[0]], starts[te[-1]] + W
            tr = np.array([i for i in idx if starts[i] + W <= lo or starts[i] >= hi])   # no shared depth samples
        a, n = fit_eval(tr, te, s); accs.append(a)
        OUTJ["random_split" if name.startswith("random") else "depth_block_split"].append(round(float(a), 3))
        OUTJ["n_test_random" if name.startswith("random") else "n_test_block"] = int(n)
        print(f"  {name}, seed {s}: best-epoch accuracy {a:.3f} on {n} windows", flush=True)
    print(f"{name}: mean {np.mean(accs):.3f}\n")
if len(sys.argv) > 3:
    import json; json.dump(OUTJ, open(sys.argv[3], "w"), indent=1)
