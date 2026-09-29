"""
One-off analysis for choosing training.C and evaluation.p1_probability.

Trains the classifier head for a few C values on the current feature table,
picks each operating threshold on VAL as job 02 does, and prints TEST KPIs and
how the TEST films fall into P1 / P2 / P3 for candidate P1 cut-offs.
Writes nothing.  Usage:  python scripts/band_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # CAI job kernels run the script without __file__; cwd is the project
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import load_config, load_feature_table  # noqa: E402
from evaluate.metrics import choose_threshold, compute_metrics  # noqa: E402
from train.train_validate import xy  # noqa: E402

C_VALUES = (0.5, 0.05, 0.01, 0.002)
P1_CUTS = (0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def main() -> int:
    cfg = load_config()
    tcfg, ecfg = cfg["training"], cfg["evaluation"]
    df, _ = load_feature_table(cfg)
    inputs = cfg["features"]["model_inputs"]
    X_tr, y_tr = xy(df[df.split == "train"], inputs)
    X_va, y_va = xy(df[df.split == "val"], inputs)
    X_te, y_te = xy(df[df.split == "test"], inputs)

    for c in C_VALUES:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight=tcfg["class_weight"], max_iter=tcfg["max_iter"]))
        model.fit(X_tr, y_tr)
        p_va, p_te = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
        thr = choose_threshold(y_va, p_va, ecfg["target_sensitivity"])
        m = compute_metrics(y_te, p_te, thr)
        print(f"\nC={c}: threshold {thr:.3f} | TEST AUROC {m['auroc']:.3f}  sens {m['sensitivity']:.3f}  "
              f"spec {m['specificity']:.3f}  Brier {m['brier']:.3f}  (VAL Brier "
              f"{compute_metrics(y_va, p_va, thr)['brier']:.3f})")
        print("  TEST quantiles:", {q: round(float(np.quantile(p_te, q)), 4) for q in (0.25, 0.5, 0.75, 0.9)})
        above = p_te >= thr
        for cut in (x for x in P1_CUTS if x > thr):
            p1 = p_te >= cut
            p2 = above & ~p1
            print(f"  P1 >= {cut}: {p1.sum():4d} films ({int(y_te[p1].sum()):4d} pneumonia) | "
                  f"P2: {p2.sum():4d} ({int(y_te[p2].sum()):4d}) | P3: {(~above).sum():4d} ({int(y_te[~above].sum()):4d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
