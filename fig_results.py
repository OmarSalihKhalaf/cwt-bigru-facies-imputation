"""Result figures from results/*.json.  Usage: python3 -I fig_results.py <outdir>"""
import sys, json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = sys.argv[1]; R = "results"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 400})
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e3df"
P = json.load(open(f"{R}/paper_numbers.json"))
STY = {"nearest_fill": ("Nearest-facies copy", "#8a8984", "--", "s"), "rf_logs_ctx": ("RF, logs + bounding facies", "#b8860b", ":", "D"),
       "ctx_lab_bigru": ("BiGRU, facies context only", "#9a5fc7", "-.", "v"), "ctx_raw_bigru": ("BiGRU, logs + context (no CWT)", "#1baf7a", "-", "^"),
       "ctx_hyb_gru": ("CWT + logs + context, GRU", "#eb6834", "-", "o"), "ctx_hyb_bigru": ("Proposed: CWT + logs + context, BiGRU", "#2a78d6", "-", "o")}

# ---------------- Fig. 5 reproduction
rp = P["repro"]
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for i, (k, lab) in enumerate([("random_split", "Random split of\noverlapping windows\n(as submitted)"), ("depth_block_split", "Test windows share\nno depth samples\nwith training")]):
    v = np.array(rp[k]); ax.bar(i, v.mean(), 0.55, color=["#eb6834", "#2a78d6"][i], alpha=0.85)
    ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=14, color=INK, zorder=3)
    ax.text(i, v.mean() + 0.04, f"{v.mean():.1%}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels([l for _, l in [(0, "Random split of\noverlapping windows\n(as submitted)"), (1, "Test windows share\nno depth samples\nwith training")]], fontsize=7.5)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy (best epoch)"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.savefig(f"{OUT}/Fig05_reproduction.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig. 7 imputation vs gap length
cases = [c for c in ["karpur", "hugoton"] if f"imp_{c}" in P]
fig, axs = plt.subplots(1, len(cases), figsize=(7.2, 3.3), sharey=False)
axs = np.atleast_1d(axs)
for ax, c in zip(axs, cases):
    T = P[f"imp_{c}"]; Gs = sorted(float(g) for g in T)
    for m, (lab, col, ls, mk) in STY.items():
        y = [T[str(g)][m].get("acc_vote", T[str(g)][m]["acc"]) for g in Gs]
        ax.plot(Gs, y, ls=ls, marker=mk, color=col, lw=2.2 if m == "ctx_hyb_bigru" else 1.3, ms=5, label=lab, zorder=3 if m == "ctx_hyb_bigru" else 2)
    ax.set_xscale("log", base=2); ax.set_xticks(Gs); ax.set_xticklabels([f"{g:g}" for g in Gs]); ax.set_xlabel("Gap length G, ft")
    ax.set_title({"karpur": "(a) Case 1, single well", "hugoton": "(b) Case 2, blind wells STUART and CRAWFORD"}[c], fontsize=9, loc="left")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
axs[0].set_ylabel("Accuracy inside gaps")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.12))
fig.savefig(f"{OUT}/Fig07_imputation_vs_gap.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig. 10 permutation importance
pc = [c for c in ["karpur", "hugoton"] if f"perm_{c}" in P]
if pc:
    LOGN = {"karpur": ["Caliper", "Gamma ray", "Bulk density", "Neutron porosity"], "hugoton": ["Gamma ray", "log10 ILD", "ΔΦ (N–D)", "Φ (N–D average)", "PE"]}
    fig, axs = plt.subplots(1, len(pc), figsize=(7.2, 2.6)); axs = np.atleast_1d(axs)
    for ax, c in zip(axs, pc):
        imp = P[f"perm_{c}"]; L = LOGN[c]
        vals = [np.average([e["drop"] for e in imp if e["log"] == j], weights=[e["n"] for e in imp if e["log"] == j]) for j in range(len(L))]
        o = np.argsort(vals)
        ax.barh(range(len(L)), np.array(vals)[o] * 100, color="#2a78d6", height=0.6); ax.set_yticks(range(len(L))); ax.set_yticklabels(np.array(L)[o])
        ax.set_xlabel("Accuracy drop when log is shifted, % points"); ax.axvline(0, color=INK, lw=0.8)
        ax.set_title({"karpur": "(a) Case 1", "hugoton": "(b) Case 2, blind wells"}[c], fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/Fig10_permutation_importance.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig. 11 training curves
if os.path.exists(f"{R}/curves_hug8.json"):
    cv = json.load(open(f"{R}/curves_hug8.json")); hst = cv["hist"]; it = [x["iter"] for x in hst]
    fig, ax1 = plt.subplots(figsize=(7.2, 2.8))
    ax1.plot(it, [x["train_loss"] for x in hst], color=MUTED, lw=1.0, label="Training loss (synthetic gaps)")
    ax1.set_xlabel("Iteration (batch of 16 crops)"); ax1.set_ylabel("Cross-entropy loss")
    ax2 = ax1.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(it, [x["val_acc"] for x in hst], color="#2a78d6", lw=1.4, label="Validation-gap accuracy (training wells)")
    ax2.plot(it, [x["test_acc"] for x in hst], color="#eb6834", lw=1.4, label="Blind-well gap accuracy (not used for selection)")
    ax2.set_ylabel("Accuracy"); ax1.axvline(cv["selected_iter"], color=INK, ls="--", lw=1)
    ax1.annotate("selected checkpoint", xy=(cv["selected_iter"], ax1.get_ylim()[0]), xytext=(cv["selected_iter"] + 40, ax1.get_ylim()[0] + 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])), fontsize=7.5, color=INK)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10))
    fig.savefig(f"{OUT}/Fig11_training_curves.png", bbox_inches="tight"); plt.close(fig)
print("ok")
