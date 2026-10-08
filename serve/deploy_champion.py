"""
Job 4 - deploy (runs only if the model's kpi-gate succeeded), for CXR_MODEL

models.<name>.stage_on_pass decides where a passing candidate goes:
  champion      1. archive the current champion, promote the candidate to serving.champion_dir
                2. use the Cloudera AI API v2 (cmlapi) to build + deploy serve/predict.py, which
                   serves every model's champion (the build snapshots the project files)
                3. if the build or deployment fails, put the previous champion back, so the
                   champion directories always match what the endpoint serves
  silent_trial  the candidate replaces the model in serving.trial_dir: cxr-06 scores every live
                study with it, nobody sees it, the endpoint is not touched. It goes live only
                through cxr-07-promote-champion.
Either way the version is registered in the Cloudera AI Registry (serve/registry.py) and
recorded in ref.model_event.

Inside a CAI job, cmlapi.default_client() authenticates with the job's own
credentials - no API key is stored in the project.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # CAI job kernels run the script without __file__; cwd is the project
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

from common import ROOT, finish, load_config, parse_args  # noqa: E402


def stage_dir(cfg, stage: str) -> Path:
    return ROOT / cfg["serving"]["champion_dir" if stage == "champion" else "trial_dir"]


def install(cfg, source: Path, stage: str) -> tuple[dict, Path | None]:
    """Copy a model package into the stage's directory; the one it replaces goes to the archive."""
    target = stage_dir(cfg, stage)
    archived = None
    if target.exists():
        archived = ROOT / cfg["serving"]["archive_dir"] / datetime.now(timezone.utc).strftime(f"%Y%m%dT%H%M%SZ-{stage}")
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(archived))
        print(f"archived previous {stage} -> {archived}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return json.loads((target / "model_meta.json").read_text()), archived


def promote(cfg, stage: str = "champion") -> tuple[dict, Path | None]:
    cand = ROOT / cfg["serving"]["candidate_dir"]
    gate_path = cand / "gate_result.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {"passed": False}
    meta = json.loads((cand / "model_meta.json").read_text())
    if not gate["passed"] or gate.get("candidate_git_sha") != meta["git_sha"]:
        raise SystemExit("Candidate has no passing gate result - refusing to deploy.")
    return install(cfg, cand, stage)


def rollback(cfg, archived: Path | None, stage: str = "champion") -> None:
    target = stage_dir(cfg, stage)
    shutil.rmtree(target, ignore_errors=True)
    if archived and archived.exists():
        shutil.move(str(archived), str(target))
        print(f"rolled back: previous {stage} restored from {archived}")


def wait(fn, ok: set, what: str, timeout=1800):
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        status = str(fn().status or "").lower()
        if status != last:
            print(f"  {what}: {status}", flush=True)
            last = status
        if status in ok:
            return status
        if "fail" in status or status in {"stopped", "timedout"}:
            raise RuntimeError(f"{what} ended in status {status}")
        time.sleep(20)
    raise RuntimeError(f"{what} timed out after {timeout} s")


def find_deployment(client, pid, model_id, build_id, timeout=300, poll=15):
    t0 = time.time()
    while True:
        deps = client.list_model_deployments(pid, model_id, build_id).model_deployments or []
        if deps:
            print(f"  found deployment {deps[0].id} of build {build_id}", flush=True)
            return deps[0]
        if time.time() - t0 >= timeout:
            raise RuntimeError(f"no deployment of build {build_id} appeared within {timeout} s")
        time.sleep(poll)


def deploy(cfg, meta):
    import cmlapi

    from ci.cai_jobs import resolve_runtime

    client = cmlapi.default_client()
    pid = os.environ["CDSW_PROJECT_ID"]
    name = cfg["serving"]["model_name"]
    runtime = resolve_runtime(client, cfg["cai"]["runtime_identifier"])
    found = [m for m in client.list_models(pid, search_filter=json.dumps({"name": name})).models if m.name == name]
    if found:
        model = found[0]
    else:
        model = client.create_model(cmlapi.CreateModelRequest(
            project_id=pid, name=name, description="Chest X-ray triage (decision support)",
            disable_authentication=False), pid)
    print(f"model {model.name} id={model.id}, runtime {runtime}")

    build = client.create_model_build(cmlapi.CreateModelBuildRequest(
        project_id=pid, model_id=model.id, file_path="serve/predict.py", function_name="predict",
        runtime_identifier=runtime,
        comment=f"fv{meta['feature_version']} git {meta['git_sha'][:7]} auroc {meta['metrics']['test']['auroc']:.3f}",
    ), pid, model.id)
    wait(lambda: client.get_model_build(pid, model.id, build.id), {"built", "succeeded"}, "build")
    try:
        dep = client.create_model_deployment(cmlapi.CreateModelDeploymentRequest(
            project_id=pid, model_id=model.id, build_id=build.id,
            cpu=cfg["cai"]["model_cpu"], memory=cfg["cai"]["model_memory_gb"]), pid, model.id, build.id)
    except Exception as e:
        # The API gateway gives up after 30 s and returns 500 while the workbench goes on
        # creating the deployment: look for it before calling the deploy a failure.
        if (getattr(e, "status", None) or 500) < 500:
            raise
        print(f"  create_model_deployment: {str(e).splitlines()[0]} - looking for the deployment", flush=True)
        dep = find_deployment(client, pid, model.id, build.id)
    wait(lambda: client.get_model_deployment(pid, model.id, build.id, dep.id), {"deployed"}, "deployment")
    print(f"champion deployed: model {model.id} build {build.id} deployment {dep.id}")


def go_live(cfg, meta, archived: Path | None, stage: str, extra_tags: dict | None = None,
            event: str | None = None) -> int:
    """Rebuild the endpoint for a new champion (rolling back on failure), register, record."""
    from lakehouse.publish import publish_model_event
    from serve import registry
    from serve.registry import registrar

    if stage == "champion":
        try:
            deploy(cfg, meta)
        except Exception as e:
            print(f"deploy failed: {e}")
            rollback(cfg, archived, stage)
            publish_model_event("ROLLED_BACK", meta, detail=str(e), stage=stage)
            return 1
    reg = registrar(cfg, meta, stage, extra_tags)
    detail = (f"registry {cfg['model']['registry_name']} v{reg['number']}" if reg else
              f"not registered: {registry.LAST_ERROR}")
    publish_model_event(event or ("DEPLOYED" if stage == "champion" else "SILENT_TRIAL"), meta, detail=detail,
                        stage=stage, registry_version=reg["number"] if reg else None)
    return 0


def main() -> int:
    parse_args(argparse.ArgumentParser())
    cfg = load_config()
    stage = cfg["model"]["stage_on_pass"]
    print(f"model {cfg['model']['name']}: passing candidate -> {stage}")
    meta, archived = promote(cfg, stage)
    return go_live(cfg, meta, archived, stage)


if __name__ == "__main__":
    finish(main())
