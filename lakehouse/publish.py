"""
Model lineage into the lakehouse, written by the CAI pipeline jobs:

  ref.training_set   per feature version and split: films, pneumonia, patients (cxr-01)
  ref.model_event    GATE_PASSED / GATE_FAILED (cxr-03), DEPLOYED / ROLLED_BACK (cxr-04),
                     with the candidate's metrics, threshold and gate checks

Only inside a CAI job (CDSW_PROJECT_ID), so a laptop or CI run of the same scripts never writes
to the platform's tables. Best-effort: without CXR_IMPALA_USER or when CDW is unreachable it prints why
and returns; a lineage record never changes a pipeline stage's result.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime


def _store():
    if not os.environ.get("CDSW_PROJECT_ID"):
        print("[lakehouse] not a CAI job - lineage not published")
        return None
    if not os.environ.get("CXR_IMPALA_USER"):
        print("[lakehouse] CXR_IMPALA_USER not set - lineage not published")
        return None
    from lakehouse.store import ImpalaStore

    return ImpalaStore()


def _safe(what: str, fn, store=None) -> bool:
    try:
        s = store or _store()
        if s is None:
            return False
        fn(s)
        print(f"[lakehouse] published {what}")
        return True
    except Exception as e:  # lineage must never fail the stage
        print(f"[lakehouse] WARNING could not publish {what}: {e}")
        return False


def training_set_rows(manifest: dict) -> list[dict]:
    created = datetime.fromisoformat(manifest["created_at"]).replace(tzinfo=None)
    pos = manifest.get("positives_by_split", {})
    splits = sorted(manifest["rows_by_split"].items()) + [("all", manifest["rows"])]
    return [{"feature_version": manifest["feature_version"], "feature_hash": manifest["feature_hash"],
             "data_split": split, "films": n,
             "pneumonia": sum(pos.values()) if split == "all" and pos else pos.get(split),
             "patients": manifest.get("patients") if split == "all" else None,
             "backbone": manifest["backbone"], "git_sha": manifest["git_sha"][:7], "created_at": created}
            for split, n in splits]


def model_event_row(event: str, meta: dict, gate: dict | None = None, detail: str = "") -> dict:
    from lakehouse.store import model_version

    m = meta.get("metrics", {})
    return {"event_id": uuid.uuid4().hex[:16], "event": event, "recorded_at": datetime.now(),
            "git_sha": meta["git_sha"][:7], "model_version": model_version(meta),
            "feature_version": meta["feature_version"], "feature_hash": meta["feature_hash"],
            "threshold": meta.get("threshold"), "val_auroc": m.get("val", {}).get("auroc"),
            "test_auroc": m.get("test", {}).get("auroc"), "test_sensitivity": m.get("test", {}).get("sensitivity"),
            "test_specificity": m.get("test", {}).get("specificity"), "test_brier": m.get("test", {}).get("brier"),
            "train_rows": meta.get("train_rows"), "gate_passed": gate.get("passed") if gate else None,
            "gate_checks": json.dumps(gate["checks"], default=str) if gate else None,
            "mlflow_run_id": meta.get("mlflow_run_id"), "detail": detail[:2000]}


def publish_training_set(manifest: dict, store=None) -> bool:
    if manifest.get("limit"):
        print("[lakehouse] feature table built with --limit - not a training set, not published")
        return False

    def write(s):
        rows = training_set_rows(manifest)
        s.ensure("ref.training_set")
        s.execute(f"DELETE FROM {s.t('ref.training_set')} WHERE feature_version = '{manifest['feature_version']}'")
        s.append("ref.training_set", rows)

    return _safe(f"training set fv{manifest['feature_version']}", write, store)


def publish_model_event(event: str, meta: dict, gate: dict | None = None, detail: str = "", store=None) -> bool:
    return _safe(f"model event {event}",
                 lambda s: s.append("ref.model_event", [model_event_row(event, meta, gate, detail)]), store)
