"""Method figures: workflow (Fig. 1), CWT example (Fig. 2), network (Fig. 3), validation layouts (Fig. 4).
Usage: python3 -I fig_method.py <outdir>
"""
import sys, numpy as np, pandas as pd, pywt, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
OUT = sys.argv[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 400})
INK, MUTED, BLUE, ORANGE, AQUA, GREY = "#1a1a1a", "#6b6b6b", "#2a78d6", "#eb6834", "#1baf7a", "#e9e8e4"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=INK, lw=1))


# ---------------------------------------------------------------- Fig. 1 workflow
fig, ax = plt.subplots(figsize=(7.2, 5.4)); ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1); ax.axis("off")
H = 0.11
box(ax, 0.06, 0.86, 0.38, H, "Wireline logs\n(depth-ordered, 0.5-ft sampling)", fc="#dbe9fa", fs=8)
box(ax, 0.56, 0.86, 0.38, H, "Core facies\n(known intervals only)", fc="#dbe9fa", fs=8)
box(ax, 0.06, 0.66, 0.88, H + 0.01, "Leakage-free split by depth or by well: blocked folds  |  gaps of G ft  |  blind wells\n"
    "Normalization fitted on training samples only; no depth input", fc="#fde3d6", fs=8, bold=False)
box(ax, 0.01, 0.40, 0.33, 0.16, "CWT path\ncomplex Morlet, 16 scales (1–32 ft)\nRe and Im channels\ntwo conv blocks → 32", fs=7.3)
box(ax, 0.36, 0.40, 0.26, 0.16, "Log path\nstandardized logs\nlinear → 32", fs=7.5)
box(ax, 0.64, 0.40, 0.35, 0.16, "Facies-context path\nfacies above and below the gap,\nrelative position, gap length,\nunknown flag; linear → 32", fs=7.5)
box(ax, 0.30, 0.20, 0.40, 0.11, "BiGRU, 64 units per direction\n+ α × nearest known facies (skip)", fc="#dbe9fa", fs=8, bold=True)
box(ax, 0.01, 0.20, 0.25, 0.11, "Early stopping on\nvalidation gaps or wells\n(never test labels)", fc=GREY, fs=7.3)
box(ax, 0.74, 0.20, 0.25, 0.11, "Facies at every depth\ninside each gap", fc="#d8f1e6", fs=8)
box(ax, 0.20, 0.02, 0.60, 0.10, "Evaluation: accuracy, macro-F1, per-class F1, McNemar test,\nSCI, bed fragmentation, transition entropy, thin-bed recall", fs=7.5)
arrow(ax, 0.25, 0.86, 0.25, 0.78); arrow(ax, 0.75, 0.86, 0.75, 0.78)
arrow(ax, 0.175, 0.66, 0.175, 0.56); arrow(ax, 0.49, 0.66, 0.49, 0.56); arrow(ax, 0.815, 0.66, 0.815, 0.56)
arrow(ax, 0.175, 0.40, 0.40, 0.31); arrow(ax, 0.49, 0.40, 0.50, 0.31); arrow(ax, 0.815, 0.40, 0.60, 0.31)
arrow(ax, 0.30, 0.255, 0.26, 0.255); arrow(ax, 0.70, 0.255, 0.74, 0.255); arrow(ax, 0.865, 0.20, 0.70, 0.12)
fig.savefig(f"{OUT}/Fig01_workflow.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2 CWT example (Case 2 well SHRIMPLIN, gamma ray)
fv = pd.read_csv("data/seg2016/facies_vectors.csv"); d = fv[fv["Well Name"] == "SHRIMPLIN"].sort_values("Depth")
x = (d.GR.values - d.GR.mean()) / d.GR.std(); z = d.Depth.values
scales = np.geomspace(2, 64, 16); pad = 64
co, _ = pywt.cwt(np.pad(x, pad, mode="reflect"), scales, "cmor1.5-1.0"); co = co[:, pad:-pad] / np.sqrt(scales)[:, None]
lam = scales * 0.5
fig, axs = plt.subplots(1, 4, figsize=(7.2, 5.2), sharey=True, gridspec_kw=dict(width_ratios=[0.8, 2, 2, 0.5], wspace=0.08))
axs[0].plot(d.GR, z, color=INK, lw=0.7); axs[0].set_xlabel("GR, API"); axs[0].set_ylabel("Depth, ft"); axs[0].invert_yaxis()
for a, data, title, cm in [(axs[1], co.real, "Re W (sign of deviation kept)", "RdBu_r"), (axs[2], np.abs(co), "|W| (modulus)", "Greys")]:
    v = np.percentile(np.abs(data), 98)
    a.pcolormesh(lam, z, data.T, cmap=cm, vmin=-v if cm == "RdBu_r" else 0, vmax=v, shading="nearest", rasterized=True)
    a.set_xscale("log"); a.set_xticks([1, 2, 4, 8, 16, 32]); a.set_xticklabels(["1", "2", "4", "8", "16", "32"]); a.set_xlabel("Pseudo-wavelength λ, ft"); a.set_title(title, fontsize=9)
ABB = ["SS", "CSiS", "FSiS", "SiSh", "MS", "WS", "D", "PS", "BS"]
COL = ["#F4D03F", "#F5B041", "#DC7633", "#6E2C00", "#1B4F72", "#2E86C1", "#AED6F1", "#A569BD", "#196F3D"]   # palette of Hall (2016)
from matplotlib.patches import Patch
fac = d.Facies.values
for i in range(len(z) - 1): axs[3].fill_betweenx([z[i], z[i + 1]], 0, 1, color=COL[fac[i] - 1], lw=0)
axs[3].set_xticks([]); axs[3].set_title("Core\nfacies", fontsize=9)
fig.legend(handles=[Patch(color=COL[i], label=f"{i + 1} {ABB[i]}") for i in range(9)], loc="lower center", ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.07))
fig.savefig(f"{OUT}/Fig02_cwt_example.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 4 validation layouts
fig, axs = plt.subplots(3, 1, figsize=(7.2, 3.6), gridspec_kw=dict(hspace=0.9))
N = 160
blk = (np.arange(N) // 16 + np.random.default_rng(3).integers(0, 5, N // 16 + 1)[np.arange(N) // 16]) % 5
col = {0: ORANGE, 1: "#a9c9ef", 2: "white", 3: "white", 4: "white"}
a = axs[0]
for i in range(N):
    fc = ORANGE if blk[i] == 0 else ("#a9c9ef" if blk[i] == 1 else BLUE)
    a.add_patch(Rectangle((i, 0), 1, 1, fc=fc, ec="none"))
near = np.convolve((blk == 0).astype(float), np.ones(9), "same") > 0
for i in range(N):
    if near[i] and blk[i] != 0: a.add_patch(Rectangle((i, 0), 1, 1, fc=GREY, ec="none"))
a.set_xlim(0, N); a.set_ylim(0, 1); a.set_yticks([]); a.set_xticks([0, 40, 80, 120, 160]); a.set_xticklabels(["0", "20", "40", "60", "80 ft"])
a.set_title("(a) Case 1 classification: 8-ft blocks in five folds; test (orange), validation (light blue), training (blue), 2-ft guard (grey)", fontsize=8, loc="left")
a = axs[1]; n = 16
for i in range(N):
    fc = ORANGE if (i // n) % 5 == 0 else "#d8f1e6"
    a.add_patch(Rectangle((i, 0), 1, 1, fc=fc, ec="none"))
a.set_xlim(0, N); a.set_ylim(0, 1); a.set_yticks([]); a.set_xticks([0, 40, 80, 120, 160]); a.set_xticklabels(["0", "20", "40", "60", "80 ft"])
a.set_title("(b) Imputation: gaps of G ft (orange, here G = 8 ft) every 5G ft; facies outside the gaps are known (green)", fontsize=8, loc="left")
a = axs[2]; a.set_xlim(0, 11); a.set_ylim(0, 1); a.axis("off")
wells = ["SHRIMP-\nLIN", "SHANKLE", "LUKE\nG U", "CROSS H\nCATTLE", "NOLAN", "NEWBY", "CHURCH-\nMAN\nBIBLE", "ALEX-\nANDER D", "KIMZEY\nA", "STUART", "CRAW-\nFORD"]
for i, wname in enumerate(wells):
    a.add_patch(Rectangle((i + 0.08, 0.15), 0.84, 0.7, fc=BLUE if i < 9 else ORANGE, ec="none"))
    a.text(i + 0.5, 0.5, wname, ha="center", va="center", fontsize=5.2, color="white")
a.set_title("(c) Case 2: nine training wells (blue; one rotates as early-stopping well) and two blind wells (orange)", fontsize=8, loc="left")
fig.savefig(f"{OUT}/Fig04_validation_layouts.png", bbox_inches="tight"); plt.close(fig)
print("ok")

# ---------------------------------------------------------------- Fig. 3 network
fig, ax = plt.subplots(figsize=(7.2, 3.4)); ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.01, 0.72, 0.16, 0.2, "CWT tensor\nF × 32 × T\n(Re, Im; 16 scales)", fc="#dbe9fa", fs=7.5)
box(ax, 0.01, 0.42, 0.16, 0.2, "Standardized logs\nF × T", fc="#dbe9fa", fs=7.5)
box(ax, 0.01, 0.12, 0.16, 0.2, "Facies context\n(2C + 3) × T", fc="#dbe9fa", fs=7.5)
box(ax, 0.22, 0.72, 0.22, 0.2, "Conv 3×3, 16 filters,\nBN, ReLU, pool 2×1\n(scale); × 2 → 16 × 8 × T", fs=7)
box(ax, 0.49, 0.72, 0.13, 0.2, "Flatten,\ndropout 0.5,\nlinear → 32", fs=7.5)
box(ax, 0.22, 0.42, 0.40, 0.2, "Linear → 32", fs=7.5)
box(ax, 0.22, 0.12, 0.40, 0.2, "Linear → 32 (no dropout)", fs=7.5)
box(ax, 0.66, 0.38, 0.12, 0.28, "Concat\n96 × T,\nReLU,\ndropout 0.3\n(log, CWT)", fs=7.5)
box(ax, 0.81, 0.62, 0.18, 0.2, "BiGRU\n64 + 64 units", fc="#dbe9fa", fs=8, bold=True)
box(ax, 0.81, 0.33, 0.18, 0.2, "Linear → C logits\n+ α · e_nf (prior)", fs=7.5)
box(ax, 0.81, 0.05, 0.18, 0.18, "Softmax: facies\nat each depth", fc="#d8f1e6", fs=7.5)
arrow(ax, 0.17, 0.82, 0.22, 0.82); arrow(ax, 0.44, 0.82, 0.49, 0.82); arrow(ax, 0.62, 0.82, 0.70, 0.66)
arrow(ax, 0.17, 0.52, 0.22, 0.52); arrow(ax, 0.62, 0.52, 0.66, 0.52)
arrow(ax, 0.17, 0.22, 0.22, 0.22); arrow(ax, 0.62, 0.22, 0.70, 0.38)
arrow(ax, 0.78, 0.55, 0.81, 0.70); arrow(ax, 0.90, 0.62, 0.90, 0.53); arrow(ax, 0.90, 0.33, 0.90, 0.23)
ax.annotate("", xy=(0.81, 0.40), xytext=(0.17, 0.16), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, ls="--", connectionstyle="arc3,rad=0.25"))
ax.text(0.50, 0.035, "nearest known facies e_nf (skip connection)", fontsize=7, color=MUTED, ha="center")
fig.savefig(f"{OUT}/Fig03_network.png", bbox_inches="tight"); plt.close(fig)
print("fig3 ok")
