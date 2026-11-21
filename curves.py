"""Training history of the proposed imputation network (Case 2, G = 8 ft, seed 0), run for the full 1,500
iterations without stopping, recording training loss, validation-gap accuracy and blind-well-gap accuracy.
The checkpoint that early stopping would select (best validation score) is marked; test accuracy is recorded
for display only and plays no part in selection.
Usage: python3 -I curves.py <out_json>
"""
import sys, json, numpy as np, torch, torch.nn as nn
OUT = sys.argv[1]
sys.argv = [sys.argv[0], "hugoton", "8", "1", "1", "/dev/null", ""]
src = open("impute2.py").read().split('res = {"case"')[0]
g = {"__name__": "x"}; exec(compile(src, "impute2_head", "exec"), g)
np_ = np; C, Xs, Ys, LOG_IDX = g["C"], g["Xs"], g["Ys"], g["LOG_IDX"]; train_ids, test_ids = g["train_ids"], g["test_ids"]
n = 16; off = 0
tm = [np.zeros(len(y), bool) for y in Ys]
for i in test_ids:
    blk = ((np.arange(len(Ys[i])) + off) // n) % 5
    tm[i] = (blk == 2) & (np.arange(len(Ys[i])) >= n) & (np.arange(len(Ys[i])) < len(Ys[i]) - n)
knowns = [Ys[i] for i in train_ids]
rng = np.random.default_rng(1000); val_gaps = [g["place_gaps"](k, n, rng, frac=0.15) for k in knowns]
raws, cws = g["build_features"]([Xs[i] for i in range(len(Xs))], [np.isin(i, train_ids) & (Ys[i] >= 0) for i in range(len(Xs))], LOG_IDX)
torch.manual_seed(0); rng = np.random.default_rng(0)
net = g["CtxNet"]("ctx_hyb_bigru", raws[0].shape[0], cws[0].shape[0]); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss(ignore_index=-1); T = 128; R = [torch.tensor(raws[i]) for i in train_ids]; Cw = [torch.tensor(cws[i]) for i in train_ids]
lens = np.array([len(k) for k in knowns], float); hist = []; run_loss = []
from sklearn.metrics import accuracy_score, f1_score
for it in range(1500):
    net.train(); rb, cb, lb, tb = [], [], [], []
    for _ in range(16):
        i = rng.choice(len(knowns), p=lens / lens.sum()); s0 = rng.integers(0, len(knowns[i]) - T)
        kn = knowns[i][s0:s0 + T].copy(); gaps = g["place_gaps"](kn, n, rng, frac=0.25, max_tries=40)
        tgt = np.where(gaps & ~val_gaps[i][s0:s0 + T], kn, -1); kin = np.where(gaps | val_gaps[i][s0:s0 + T], -1, kn)
        rb.append(R[i][:, s0:s0 + T]); cb.append(Cw[i][..., s0:s0 + T]); lb.append(torch.tensor(g["lab_input"](kin))); tb.append(torch.tensor(tgt))
    rb, cb, lb, tb = map(torch.stack, (rb, cb, lb, tb))
    opt.zero_grad(); loss = lossf(net(rb, cb, lb).reshape(-1, C), tb.reshape(-1)); loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); run_loss.append(float(loss))
    if it % 10 == 9:
        yt, yp = [], []
        for j, k in enumerate(knowns):
            p = g["predict"](net, raws[train_ids[j]], cws[train_ids[j]], np.where(val_gaps[j], -1, k)); yt.append(k[val_gaps[j]]); yp.append(p[val_gaps[j]])
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        vacc = accuracy_score(yt, yp); vsc = vacc + f1_score(yt, yp, average="macro")
        tt, tp = [], []
        for i in test_ids:
            m = tm[i] & (Ys[i] >= 0); p = g["predict"](net, raws[i], cws[i], np.where(tm[i], -1, Ys[i])); tt.append(Ys[i][m]); tp.append(p[m])
        tacc = accuracy_score(np.concatenate(tt), np.concatenate(tp))
        hist.append({"iter": it + 1, "train_loss": float(np.mean(run_loss[-10:])), "val_acc": vacc, "val_score": vsc, "test_acc": tacc, "alpha": float(net.alpha)})
best = max(hist, key=lambda h: h["val_score"])
json.dump({"hist": hist, "selected_iter": best["iter"]}, open(OUT, "w"), indent=1)
print("selected iteration", best["iter"], "val acc", round(best["val_acc"], 3), "test acc", round(best["test_acc"], 3))
