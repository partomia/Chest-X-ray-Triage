"""Model evaluation: threshold selection and triage KPIs. Pure numpy/sklearn, unit-testable."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score,
)


def choose_threshold(y_true, prob, target_sensitivity: float) -> float:
    """Highest threshold whose sensitivity on this set is still >= target (maximises specificity)."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    pos = np.sort(prob[y_true == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target_sensitivity) * len(pos)))  # positives we are allowed to miss
    return float(pos[k])


def compute_metrics(y_true, prob, threshold: float) -> dict:
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "threshold": float(threshold),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(2 * prec * sens / (prec + sens)) if prec + sens else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def priority_band(prob: float, threshold: float, p1: float) -> str:
    """P1 = read first, P2 = likely abnormal, P3 = routine.

    P1 never starts below the operating threshold: a film the model calls negative
    is never "read first", even when the threshold lands above p1_probability."""
    if prob >= max(p1, threshold):
        return "P1"
    if prob >= threshold:
        return "P2"
    return "P3"
