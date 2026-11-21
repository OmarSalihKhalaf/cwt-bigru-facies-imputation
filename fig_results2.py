"""Fig. 7 (example interval) and Fig. 8 (confusion matrices) from the Case 2 full-coverage imputation run, G = 8 ft.
Usage: python3 -I fig_results2.py <outdir>
"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
OUT = sys.argv[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "savefig.dpi": 400})
REPO = "data/seg2016"
nf = pd.read_csv(f"{REPO}/nofacies_data.csv"); bl = pd.read_csv(f"{REPO}/blind_stuart_crawford_core_facies.csv")
bw = nf.merge(bl, left_on=["Well Name", "Depth"], right_on=["WellName", "Depth.ft"])
D = {w: bw[bw["Well Name"] == w].sort_values("Depth").reset_index(drop=True) for w in ["STUART", "CRAWFORD"]}
Y = {w: np.where(D[w].LithCode.values <= 9, D[w].LithCode.values - 1, -1) for w in D}
res = json.load(open("results/imp_full_hugoton.json"))
row = [r for r in res["rows"] if r["G"] == 8.0][0]; n = 16
# positions of scored samples, in the order used by impute3.py (STUART then CRAWFORD, depth order)
pos = {w: np.where((np.arange(len(Y[w])) >= n) & (np.arange(len(Y[w])) < len(Y[w]) - n) & (Y[w] >= 0))[0] for w in ["STUART", "CRAWFORD"]}
cut = len(pos["STUART"]); yt = np.array(row["y_true"])
assert len(yt) == cut + len(pos["CRAWFORD"])
vote = np.array([np.bincount(c, minlength=9).argmax() for c in np.stack(row["ctx_hyb_bigru"]["pred_seeds"]).T])
nfill = np.array(row["nearest_fill"]["pred"])
ABB = ["SS", "CSiS", "FSiS", "SiSh", "MS", "WS", "D", "PS", "BS"]
COL = ["#F4D03F", "#F5B041", "#DC7633", "#6E2C00", "#1B4F72", "#2E86C1", "#AED6F1", "#A569BD", "#196F3D"]   # palette of Hall (2016)

# ---------------- Fig. 7 example interval (STUART, layout position 0 gaps)
w = "STUART"; z = D[w].Depth.values; yS = Y[w]
full_v = np.full(len(yS), -1); full_n = np.full(len(yS), -1)
full_v[pos[w]] = vote[:cut]; full_n[pos[w]] = nfill[:cut]
blk = (np.arange(len(yS)) // n) % 5; gap = (blk == 0) & (np.arange(len(yS)) >= n) & (np.arange(len(yS)) < len(yS) - n)
lo, hi = 2870, 2990; sel = (z >= lo) & (z <= hi)
fig, axs = plt.subplots(1, 4, figsize=(6.0, 6.2), sharey=True, gridspec_kw=dict(width_ratios=[1.6, 0.7, 0.7, 0.7], wspace=0.08))
axs[0].plot(D[w].GR.values[sel], z[sel], color="#1a1a1a", lw=0.8); axs[0].set_xlabel("GR, API"); axs[0].set_ylabel("Depth, ft"); axs[0].invert_yaxis()


def column(ax, vals, title):
    for i in np.where(sel)[0][:-1]:
        v = vals[i]
        if v >= 0: ax.fill_betweenx([z[i], z[i + 1]], 0, 1, color=COL[v], lw=0)
    ax.set_xticks([]); ax.set_title(title, fontsize=8.5)


column(axs[1], yS, "Core")
column(axs[2], np.where(gap, full_n, yS), "Nearest\ncopy")
column(axs[3], np.where(gap, full_v, yS), "Proposed")
for ax in axs:
    for i in np.where(sel & gap)[0]:
        pass
    edges = np.where(np.diff(np.r_[0, (sel & gap).astype(int), 0]) != 0)[0]
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axhspan(z[a], z[min(b, len(z) - 1)], color="black", alpha=0.08 if ax is axs[0] else 0, lw=0)
        if ax is not axs[0]: ax.plot([0, 1, 1, 0, 0], [z[a], z[a], z[min(b, len(z) - 1)], z[min(b, len(z) - 1)], z[a]], color="black", lw=0.8)
fig.legend(handles=[Patch(color=COL[i], label=f"{i + 1} {ABB[i]}") for i in range(9)], loc="lower center", ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.savefig(f"{OUT}/Fig08_example_interval.png", bbox_inches="tight"); plt.close(fig)
acc_ex = (np.mean(full_v[sel & gap & (yS >= 0)] == yS[sel & gap & (yS >= 0)]), np.mean(full_n[sel & gap & (yS >= 0)] == yS[sel & gap & (yS >= 0)]))
print("example interval accuracy (proposed, nearest):", acc_ex)

# ---------------- Fig. 8 confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.4))
for ax, p, title in [(axs[0], nfill, f"Nearest-facies copy ({np.mean(nfill == yt):.1%})"), (axs[1], vote, f"Proposed network ({np.mean(vote == yt):.1%})")]:
    M = np.zeros((9, 9))
    for a, b in zip(yt, p): M[a, b] += 1
    R = M / np.maximum(M.sum(1, keepdims=True), 1) * 100
    ax.imshow(R, cmap="Blues", vmin=0, vmax=100)
    for i in range(9):
        for j in range(9):
            if M[i, j] > 0: ax.text(j, i, f"{R[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if R[i, j] > 55 else "#1a1a1a")
    ax.set_xticks(range(9)); ax.set_yticks(range(9)); ax.set_xticklabels(ABB, rotation=90, fontsize=7); ax.set_yticklabels([f"{a} ({int(M[i].sum())})" for i, a in enumerate(ABB)], fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_title(title, fontsize=8.5)
axs[0].set_ylabel("Core facies (samples)")
fig.tight_layout(); fig.savefig(f"{OUT}/Fig09_confusion.png", bbox_inches="tight"); plt.close(fig)
json.dump({"example_interval": [lo, hi], "acc_proposed": acc_ex[0], "acc_nearest": acc_ex[1]}, open("results/fig_example.json", "w"))
print("ok")
