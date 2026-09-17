"""Verification and retrieval metrics.

Headline numbers are TPR at 1% FPR and Recall@1. Accuracy is deliberately absent:
on a catalog of N garments, a model that says "different" to everything scores
close to perfect accuracy and is worthless.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """True positive rate at the operating point where FPR first reaches target.

    Flagging an honest customer is the expensive error, so the threshold is set
    by the false positive budget, not by whatever maximises accuracy.
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def verification(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auc": float("nan"), "tpr_at_1pct_fpr": float("nan"), "n_pairs": len(labels)}
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "tpr_at_1pct_fpr": tpr_at_fpr(labels, scores, 0.01),
        "n_pairs": int(len(labels)),
    }


def recall_at_k(score_matrix: np.ndarray, correct_index: np.ndarray, ks=(1, 5)) -> dict[str, float]:
    """score_matrix is queries by gallery garments. correct_index is the true column."""
    if score_matrix.size == 0:
        return {f"recall_at_{k}": float("nan") for k in ks}
    order = np.argsort(-score_matrix, axis=1)
    ranks = np.array([np.where(order[i] == correct_index[i])[0][0] for i in range(len(correct_index))])
    out = {f"recall_at_{k}": float((ranks < k).mean()) for k in ks}
    out["median_rank"] = float(np.median(ranks) + 1)
    return out
