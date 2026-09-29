"""
Job 2 - train-validate

Reads ONE explicit version of the feature table, trains the classifier head,
evaluates on VAL (to pick the operating threshold) and TEST (to report KPIs),
logs everything to CAI Experiments (MLflow), and writes a candidate package:
    outputs/candidate/model.joblib
    outputs/candidate/model_meta.json   <- read by the KPI gate and by serving

An MLflow failure only prints a warning: tracking must never block a candidate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # CAI job kernels run the script without __file__; cwd is the project
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import ROOT, finish, git_sha, load_config, load_feature_table, parse_args  # noqa: E402
from evaluate.metrics import choose_threshold, compute_metrics, priority_band  # noqa: E402
from features.feature_logic import QUALITY_FEATURES, feature_hash, model_matrix  # noqa: E402


def xy(df, model_inputs):
    emb = np.stack(df["embedding"].to_numpy())
    qual = df[QUALITY_FEATURES].to_dict("records")
    return model_matrix(emb, qual, model_inputs), df["label"].to_numpy()


def log_to_mlflow(cfg: dict, meta: dict, model, meta_path: Path) -> str | None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("[train-validate] mlflow not installed - skipping experiment logging")
        return None
    t, e = cfg["training"], cfg["evaluation"]
    try:
        mlflow.set_experiment(cfg["project"]["mlflow_experiment"])
        with mlflow.start_run(run_name=f"fv{meta['feature_version']}-{meta['git_sha'][:7]}") as run:
            mlflow.log_params({
                "feature_version": meta["feature_version"], "feature_hash": meta["feature_hash"],
                "backbone": meta["backbone"], "backbone_revision": meta["backbone_revision"],
                "model_inputs": ",".join(meta["model_inputs"]), "C": t["C"], "class_weight": t["class_weight"],
                "target_sensitivity": e["target_sensitivity"], "git_sha": meta["git_sha"],
                "feature_table_limit": meta["feature_table_limit"],
            })
            for split in ("val", "test"):
                mlflow.log_metrics({f"{split}_{k}": float(v) for k, v in meta["metrics"][split].items()
                                    if isinstance(v, (int, float))})
            mlflow.sklearn.log_model(model, "model")
            meta["mlflow_run_id"] = run.info.run_id
            meta_path.write_text(json.dumps(meta, indent=2))
            mlflow.log_artifact(str(meta_path))
            return run.info.run_id
    except Exception as ex:
        print(f"[train-validate] WARNING mlflow logging failed, candidate kept: {ex}")
        return None


def main() -> int:
    parse_args(argparse.ArgumentParser())
    cfg = load_config()
    fcfg, tcfg, ecfg = cfg["features"], cfg["training"], cfg["evaluation"]

    df, manifest = load_feature_table(cfg)
    current_hash = feature_hash(fcfg)
    if manifest["feature_hash"] != current_hash:
        print(f"[train-validate] feature table hash {manifest['feature_hash']} != code hash {current_hash}. "
              "Rebuild features (bump features.version).")
        return 1

    inputs = fcfg["model_inputs"]
    X_tr, y_tr = xy(df[df.split == "train"], inputs)
    X_va, y_va = xy(df[df.split == "val"], inputs)
    X_te, y_te = xy(df[df.split == "test"], inputs)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=tcfg["C"], class_weight=tcfg["class_weight"], max_iter=tcfg["max_iter"]),
    )
    model.fit(X_tr, y_tr)

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    thr = choose_threshold(y_va, p_va, ecfg["target_sensitivity"])
    m_val = compute_metrics(y_va, p_va, thr)
    m_test = compute_metrics(y_te, p_te, thr)

    out = ROOT / cfg["serving"]["candidate_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_result.json").unlink(missing_ok=True)   # a new candidate has not been gated yet
    joblib.dump(model, out / "model.joblib")
    meta = {
        "model_name": cfg["serving"]["model_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "feature_version": manifest["feature_version"],
        "feature_hash": manifest["feature_hash"],
        "feature_table_limit": int(manifest.get("limit") or 0),
        "backbone": manifest["backbone"],
        "backbone_revision": manifest["backbone_revision"],
        "model_inputs": inputs,
        "threshold": thr,
        "p1_probability": ecfg["p1_probability"],
        "quality_reference": manifest["quality_reference"],
        "metrics": {"val": m_val, "test": m_test},
        "train_rows": int(len(y_tr)),
        "mlflow_run_id": None,
    }
    meta_path = out / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log_to_mlflow(cfg, meta, model, meta_path)

    print(json.dumps({"threshold": thr, "val": m_val, "test": m_test}, indent=2))
    bands = [priority_band(float(p), thr, ecfg["p1_probability"]) for p in p_te]
    print(f"TEST bands (P1 from p >= {max(ecfg['p1_probability'], thr):.3f}):")
    for b in ("P1", "P2", "P3"):
        idx = [i for i, x in enumerate(bands) if x == b]
        print(f"  {b}: {len(idx):4d} films, {sum(int(y_te[i]) for i in idx):4d} pneumonia")
    print("TEST pneumonia-probability quantiles:",
          {q: round(float(np.quantile(p_te, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
    return 0


if __name__ == "__main__":
    finish(main())
