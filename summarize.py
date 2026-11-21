"""Collect every number used in the paper from the result files into results/paper_numbers.json, and print tables.
Usage: python3 -I summarize.py
"""
import json, os, sys, numpy as np
from scipy.stats import binomtest
sys.path.insert(0, ".")
from geo_metrics import allowed_contacts, sci, bfr, transition_entropy, thin_bed_recall
R = "results"
out = {}


def mcnemar(y, a, b):
    a, b, y = map(np.asarray, (a, b, y)); ca, cb = a == y, b == y
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return {"a_only": n01, "b_only": n10, "p": float(p)}


NAMES = {"rf_point": "RF, point logs", "hgb_point": "Gradient boosting, point logs", "rf_context": "RF, logs ±2 ft",
         "rf_cwt": "RF, logs + |W| CWT", "raw_gru": "GRU, logs", "raw_bigru": "BiGRU, logs", "cwt_cnn_bigru": "CWT–CNN–BiGRU (no logs)",
         "hyb_cnn": "Logs + CWT–CNN (no recurrence)", "hyb_gru": "Logs + CWT–CNN–GRU", "hyb_lstm": "Logs + CWT–CNN–BiLSTM",
         "hyb_bigru": "Logs + CWT–CNN–BiGRU", "hyb_bigru_complex": "Logs + Re/Im CWT–CNN–BiGRU"}

# ---------------- reproduction
rp = json.load(open(f"{R}/reproduce_original.json"))
out["repro"] = {"random": float(np.mean(rp["random_split"])), "block": float(np.mean(rp["depth_block_split"])), **rp}

# ---------------- Case 1 classification
k = json.load(open(f"{R}/karpur_b16.json")); y = np.array(k["y"]); ok = y >= 0
kc = {}
for m in k["models"]:
    a = [r["scores"][m]["acc"] for r in k["runs"]]; f = [r["scores"][m]["macroF1"] for r in k["runs"]]
    kc[m] = {"acc": float(np.mean(a)), "acc_sd": float(np.std(a)), "macroF1": float(np.mean(f)), "macroF1_sd": float(np.std(f)),
             "perclassF1": np.mean([r["scores"][m]["perclassF1"] for r in k["runs"]], 0).round(3).tolist()}
    seqs = [np.array(r["pred"][m])[ok] for r in k["runs"]]
    A = allowed_contacts([y[ok]])
    kc[m]["BFR"] = float(np.mean([bfr([s], [y[ok]]) for s in seqs]))
kc["_mcnemar_hyb_vs_rfcwt"] = [mcnemar(y[ok], np.array(r["pred"]["hyb_bigru"])[ok], np.array(r["pred"]["rf_cwt"])[ok]) for r in k["runs"]]
kc["_mcnemar_rawgru_vs_rfpoint"] = [mcnemar(y[ok], np.array(r["pred"]["raw_gru"])[ok], np.array(r["pred"]["rf_point"])[ok]) for r in k["runs"]]
out["karpur_cls"] = kc; out["karpur_classes"] = k["classes"]
d = y[ok]; out["karpur_beds"] = int(np.sum(d[1:] != d[:-1]) + 1)

# ---------------- Case 2 classification (blind wells)
h = json.load(open(f"{R}/hug_blind.json"))
import pandas as pd
REPO = "data/seg2016"
fv = pd.read_csv(f"{REPO}/facies_vectors.csv"); fv = fv[fv["Well Name"] != "Recruit F9"]
nf = pd.read_csv(f"{REPO}/nofacies_data.csv"); bl = pd.read_csv(f"{REPO}/blind_stuart_crawford_core_facies.csv")
bw = nf.merge(bl, left_on=["Well Name", "Depth"], right_on=["WellName", "Depth.ft"])
YB = {w: (lambda v: np.where(v <= 9, v - 1, 99))(bw[bw["Well Name"] == w].sort_values("Depth").LithCode.values) for w in ["STUART", "CRAWFORD"]}
ytrue = np.concatenate([YB["STUART"], YB["CRAWFORD"]]); A = allowed_contacts([g.sort_values("Depth").Facies.values - 1 for _, g in fv.groupby("Well Name")])
hc = {}
def add_cls(hh, m, key):
    sc = [r["scores"][m]["both"] for r in hh["runs"]]
    preds = [[np.array(r["pred"][m][w]) for w in ["STUART", "CRAWFORD"]] for r in hh["runs"]]
    vote = np.concatenate([np.array([np.bincount(c, minlength=9).argmax() for c in np.stack([p[j] for p in preds]).T]) for j in range(2)])
    hc[key] = {"acc": float(np.mean([s["acc"] for s in sc])), "acc_sd": float(np.std([s["acc"] for s in sc])),
               "macroF1": float(np.mean([s["macroF1"] for s in sc])), "adjacentAcc": float(np.mean([s["adjacentAcc"] for s in sc])),
               "acc_vote": float(np.mean(vote == ytrue)), "perclassF1": np.mean([s["perclassF1"] for s in sc], 0).round(3).tolist(),
               "SCI": float(np.mean([sci(p, A) for p in preds])), "BFR": float(np.mean([bfr(p, [YB["STUART"], YB["CRAWFORD"]]) for p in preds])),
               "H_T": float(np.mean([transition_entropy(p, 9) for p in preds])),
               "thin_recall": float(np.mean([thin_bed_recall(p, [YB["STUART"], YB["CRAWFORD"]])[0] for p in preds]))}
    for w in ["STUART", "CRAWFORD"]:
        hc[key][f"acc_{w}"] = float(np.mean([r["scores"][m][w]["acc"] for r in hh["runs"]]))
for m in h["models"]: add_cls(h, m, m)
if os.path.exists(f"{R}/hug_blind_complex.json"):
    hx = json.load(open(f"{R}/hug_blind_complex.json")); add_cls(hx, "hyb_bigru", "hyb_bigru_complex")
    hc["_mcnemar_hybcplx_vs_rfctx"] = [mcnemar(ytrue, np.concatenate([np.array(rx["pred"]["hyb_bigru"][w]) for w in ["STUART", "CRAWFORD"]]),
                                               np.concatenate([np.array(r["pred"]["rf_context"][w]) for w in ["STUART", "CRAWFORD"]])) for rx, r in zip(hx["runs"], h["runs"])]
true_seqs = [YB["STUART"], YB["CRAWFORD"]]
hc["_true"] = {"SCI": sci(true_seqs, A), "H_T": transition_entropy(true_seqs, 9), "beds": int(sum(np.sum(s[1:] != s[:-1]) + 1 for s in true_seqs)),
               "thin_beds": thin_bed_recall(true_seqs, true_seqs)[1]}
hc["_mcnemar_hyb_vs_rfctx"] = [mcnemar(ytrue, np.concatenate([np.array(r["pred"]["hyb_bigru"][w]) for w in ["STUART", "CRAWFORD"]]),
                                       np.concatenate([np.array(r["pred"]["rf_context"][w]) for w in ["STUART", "CRAWFORD"]])) for r in h["runs"]]
out["hug_cls"] = hc
if os.path.exists(f"{R}/tune_hug.json"): out["tune_lowo"] = {k_: v["pooled"]["acc"] for k_, v in json.load(open(f"{R}/tune_hug.json")).items()}
if os.path.exists(f"{R}/sens_cwt.json"): out["sens"] = json.load(open(f"{R}/sens_cwt.json"))

# ---------------- imputation
IM = ["nearest_fill", "rf_logs_ctx", "ctx_lab_bigru", "ctx_raw_bigru", "ctx_hyb_gru", "ctx_hyb_bigru"]
for case in ["karpur", "hugoton"]:
    files = [f"{R}/imp_final_karpur.json", f"{R}/imp_final_karpur_G4.json"] if case == "karpur" else [f"{R}/imp_full_hugoton_G8.json", f"{R}/imp_full_hugoton_rest.json"]
    files = [f for f in files if os.path.exists(f)]
    if not files: continue
    parts = [json.load(open(f)) for f in files]
    res = {"rows": [r for p in parts for r in p["rows"]], "perm_importance": [e for p in parts for e in p.get("perm_importance", [])]}
    rows = res["rows"]; tab = {}
    for G in sorted(set(r["G"] for r in rows)):
        rr = [r for r in rows if r["G"] == G]; e = {"n_test": int(sum(r["n_test"] for r in rr))}
        for m in IM:
            e[m] = {"acc": float(np.mean([r[m]["acc"] for r in rr])), "macroF1": float(np.mean([r[m]["macroF1"] for r in rr]))}
            if "acc_sd" in rr[0][m]: e[m]["acc_sd"] = float(np.mean([r[m]["acc_sd"] for r in rr])); e[m]["acc_vote"] = float(np.mean([r[m]["acc_vote"] for r in rr]))
        yt = np.concatenate([r["y_true"] for r in rr])
        hv = np.concatenate([np.array([np.bincount(c, minlength=9).argmax() for c in np.stack(r["ctx_hyb_bigru"]["pred_seeds"]).T]) for r in rr])
        e["mcnemar_vs_nearest"] = mcnemar(yt, hv, np.concatenate([r["nearest_fill"]["pred"] for r in rr]))
        e["mcnemar_vs_rfctx"] = mcnemar(yt, hv, np.concatenate([r["rf_logs_ctx"]["pred"] for r in rr]))
        e["mcnemar_vs_raw"] = mcnemar(yt, hv, np.concatenate([np.array([np.bincount(c, minlength=9).argmax() for c in np.stack(r["ctx_raw_bigru"]["pred_seeds"]).T]) for r in rr]))
        from sklearn.metrics import f1_score as _f1
        e["perclassF1_proposed"] = _f1(yt, hv, average=None, labels=list(range(9)), zero_division=0).round(3).tolist()
        e["perclassF1_nearest"] = _f1(yt, np.concatenate([r["nearest_fill"]["pred"] for r in rr]), average=None, labels=list(range(9)), zero_division=0).round(3).tolist()
        e["support"] = np.bincount(yt, minlength=9).tolist()
        tab[str(G)] = e
    out[f"imp_{case}"] = tab
    if "perm_importance" in res: out[f"perm_{case}"] = res["perm_importance"]
json.dump(out, open(f"{R}/paper_numbers.json", "w"), indent=1)

# ---------------- print
print("Repro:", out["repro"]["random"], out["repro"]["block"])
print("\nCase 1 classification"); [print(f"  {NAMES[m]:34s} {v['acc']:.3f} ± {v['acc_sd']:.3f}  mF1 {v['macroF1']:.3f}  BFR {v['BFR']:.2f}") for m, v in kc.items() if not m.startswith("_")]
print("  McNemar hyb vs rf_cwt:", kc["_mcnemar_hyb_vs_rfcwt"])
print("\nCase 2 blind classification"); [print(f"  {NAMES[m]:34s} {v['acc']:.3f} ± {v['acc_sd']:.3f}  mF1 {v['macroF1']:.3f} adj {v['adjacentAcc']:.3f} vote {v['acc_vote']:.3f} SCI {v['SCI']:.3f} BFR {v['BFR']:.2f} thin {v['thin_recall']:.2f}") for m, v in hc.items() if not m.startswith("_")]
print("  true:", hc["_true"]); print("  McNemar hyb vs rf_ctx:", hc["_mcnemar_hyb_vs_rfctx"])
for case in ["karpur", "hugoton"]:
    if f"imp_{case}" not in out: continue
    print(f"\nImputation {case}")
    for G, e in out[f"imp_{case}"].items():
        print(f"  G={G:>4} n={e['n_test']:4d} " + " ".join(f"{m}={e[m]['acc']:.3f}" + (f"/v{e[m]['acc_vote']:.3f}" if 'acc_vote' in e[m] else "") for m in IM)
              + f"  p(nf)={e['mcnemar_vs_nearest']['p']:.3g} p(rf)={e['mcnemar_vs_rfctx']['p']:.3g} p(raw)={e['mcnemar_vs_raw']['p']:.3g}")
