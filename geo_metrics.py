"""Stratigraphic metrics computed from depth-ordered facies sequences (definitions used in the paper).

Notation: a sequence y_1..y_N of facies codes in depth order. A *contact* is k with y_k != y_{k+1};
a *bed* is a maximal run of equal codes.

SCI  (Stratigraphic Consistency Index), Eq. (S1):
     SCI = (1/|K_pred|) * sum_{k in K_pred} 1[ (yhat_k -> yhat_{k+1}) in A_ref ]
     K_pred = predicted contacts; A_ref = set of facies contacts (i->j, i != j) observed in the reference
     (training) labels. SCI in [0, 1]; 1 means every predicted contact is one that occurs in the training wells.
BFR  (bed fragmentation ratio) = number of predicted beds / number of true beds (1 = same bed count; >1 = flicker).
H_T  (transition entropy, nats), Eq. (S2):
     P_ij = n_ij / sum_j n_ij over contacts only (i != j); w_i = share of contacts leaving facies i;
     H_T = - sum_i w_i sum_j P_ij ln P_ij.  Computed for predicted and for true sequences; closer values
     mean the predicted vertical stacking is as ordered (or disordered) as the real one.
Thin-bed recall: share of true beds no thicker than `thin` samples whose samples are predicted correctly
     by majority vote.
"""
import numpy as np


def contacts(seq):
    seq = np.asarray(seq); k = np.where(seq[:-1] != seq[1:])[0]
    return list(zip(seq[k], seq[k + 1]))


def beds(seq):
    seq = np.asarray(seq); edges = np.r_[0, np.where(seq[:-1] != seq[1:])[0] + 1, len(seq)]
    return [(edges[i], edges[i + 1], seq[edges[i]]) for i in range(len(edges) - 1)]


def allowed_contacts(ref_seqs):
    A = set()
    for s in ref_seqs: A.update(contacts(s))
    return A


def sci(pred_seqs, A):
    c = [t for s in pred_seqs for t in contacts(s)]
    return float(np.mean([t in A for t in c])) if c else 1.0


def bfr(pred_seqs, true_seqs):
    return sum(len(beds(s)) for s in pred_seqs) / sum(len(beds(s)) for s in true_seqs)


def transition_entropy(seqs, n_cls):
    n = np.zeros((n_cls, n_cls))
    for s in seqs:
        for i, j in contacts(s):
            if i < n_cls and j < n_cls: n[i, j] += 1
    if n.sum() == 0: return 0.0
    w = n.sum(1) / n.sum(); H = 0.0
    for i in range(n_cls):
        if n[i].sum() == 0: continue
        p = n[i][n[i] > 0] / n[i].sum(); H -= w[i] * np.sum(p * np.log(p))
    return float(H)


def thin_bed_recall(pred_seqs, true_seqs, thin=4):
    hit = tot = 0
    for p, t in zip(pred_seqs, true_seqs):
        p = np.asarray(p)
        for a, b, f in beds(t):
            if b - a <= thin:
                tot += 1; vals, cnt = np.unique(p[a:b], return_counts=True); hit += int(vals[np.argmax(cnt)] == f)
    return (hit / tot if tot else float("nan")), tot


if __name__ == "__main__":                                   # worked example for the supplementary file
    true = [2, 2, 2, 3, 3, 4, 4, 4, 5, 5]; pred = [2, 2, 3, 2, 3, 4, 4, 6, 5, 5]
    A = {(2, 3), (3, 4), (4, 5)}
    print("contacts pred:", contacts(pred))
    print("SCI =", sci([pred], A), " BFR =", bfr([pred], [true]))
    print("H_T true =", round(transition_entropy([true], 8), 3), " H_T pred =", round(transition_entropy([pred], 8), 3))
