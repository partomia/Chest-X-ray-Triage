"""
Model versions in the Cloudera AI Registry: one registered model per finding model
(models.<name>.registry_name), one version per promotion.

A version is created when a candidate that passed its KPI gate becomes a SILENT TRIAL
(scored on live studies, never shown on the worklist) or a CHAMPION (served and ranking the
worklist). Promoting a silent trial to champion registers the same MLflow run again, as a new
version with stage champion and the approval: the registry then lists the model's whole
history, and the highest champion version is what the endpoint serves.

Each version carries its MLflow run's model (a scikit-learn pipeline on the 768 ViT features)
and the run's parameters and metrics: finding, intended population, git commit, feature version
and hash, operating threshold, val and TEST KPIs. The audit tags (stage among them) are sent at
creation, but the federal workbench stores none (nor did it for Spend-Analytics) and cannot
update them later, so the stage, the approver and the version number are recorded with the
model event in ref.model_event (registry_version), which is what the dashboards read.

Registering never fails a job: the model is already promoted; the registry records it.
"""
from __future__ import annotations

import os

KPIS = ("auroc", "sensitivity", "specificity", "brier")


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def version_tags(meta: dict, stage: str, extra: dict | None = None) -> list[dict]:
    from lakehouse.store import model_version

    test = meta["metrics"]["test"]
    tags = {"stage": stage, "model": meta.get("model"), "finding": meta.get("finding"),
            "population": meta.get("population"), "model_version": model_version(meta),
            "git_sha": meta["git_sha"][:7], "feature_version": meta["feature_version"],
            "feature_hash": meta["feature_hash"], "backbone": meta["backbone"],
            "threshold": f"{meta['threshold']:.4f}", "train_rows": str(meta.get("train_rows")),
            **{f"test_{k}": f"{float(test[k]):.4f}" for k in KPIS if test.get(k) is not None},
            **(extra or {})}
    return [{"key": k, "value": str(v)} for k, v in tags.items() if v is not None]


def find_model(client, name: str):
    models = _get(client.list_registered_models(page_size=100), "models") or []
    return next((m for m in models if _get(m, "name") == name), None)


def versions(client, name: str) -> list:
    model = find_model(client, name)
    if model is None:
        return []
    return _get(client.get_registered_model(_get(model, "model_id")), "model_versions") or []


def register_version(client, meta: dict, name: str, stage: str, experiment_id: str,
                     project_id: str | None = None, extra: dict | None = None, description: str = "") -> dict:
    """The candidate's MLflow model as a new version of `name`; {model_id, version, number, previous}."""
    before = versions(client, name)
    created = client.create_registered_model({
        "project_id": project_id or os.environ.get("CDSW_PROJECT_ID", ""), "experiment_id": experiment_id,
        "run_id": meta["mlflow_run_id"], "model_path": "model", "model_name": name,
        "tags": version_tags(meta, stage, extra), "description": description,
        "notes": f"{stage}: git {meta['git_sha'][:7]}, gate passed", "visibility": "PRIVATE"})
    model_id = _get(created, "model_id")
    after = _get(client.get_registered_model(model_id), "model_versions") or []
    new = max(after, key=lambda v: _get(v, "number") or 0)
    prev = max((_get(v, "number") or 0 for v in before), default=None)
    out = {"model_id": model_id, "version": _get(new, "model_version_id"), "number": _get(new, "number"),
           "previous": prev}
    print(f"registry: {name} version {out['number']} ({stage}; previous {prev})", flush=True)
    return out


LAST_ERROR = ""   # why the last registrar() call registered nothing, for the model event's detail


def registrar(cfg: dict, meta: dict, stage: str, extra: dict | None = None) -> dict | None:
    """register_version from inside a CAI job (cmlapi.default_client). None when disabled or on error."""
    global LAST_ERROR
    LAST_ERROR = ""
    if not cfg.get("registry", {}).get("enabled") or not meta.get("mlflow_run_id"):
        LAST_ERROR = "registry disabled or no MLflow run"
        print(f"registry: {LAST_ERROR} - not registered")
        return None
    try:
        import cmlapi
        import mlflow

        experiment_id = mlflow.get_run(meta["mlflow_run_id"]).info.experiment_id
        return register_version(cmlapi.default_client(), meta, cfg["model"]["registry_name"], stage,
                                experiment_id, extra=extra, description=cfg["model"].get("description", ""))
    except Exception as e:  # the promotion stands; the registry is its record
        LAST_ERROR = f"{type(e).__name__}: {e}"[:500]
        print(f"registry: WARNING not registered: {LAST_ERROR}")
        return None
