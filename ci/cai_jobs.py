"""
The CAI jobs of this project: one definition read by ci/create_cai_jobs.py
(creates them) and ci/trigger_cai_pipeline.py (runs and follows them).
Pure Python, so the GitHub runner can import it without cmlapi.

Each model has a CHAIN of build-features, train-validate, kpi-gate, deploy, run in that
order through CAI job dependencies; a dependent job starts only when its parent succeeds,
which makes the KPI gate's exit code a hard stop. The jobs are the same scripts, and the
job's environment CXR_MODEL picks the model (config/pipeline.yaml, models):
  cxr-*  pneumonia, paediatric, champion: the served model; cxr-00 syncs the code first
  qc-*   film_qc, every film, champion: a film unfit for AI triage gets no AI priority
  ptx-*  pneumothorax, adult, silent trial: scored in the lakehouse, not on the worklist,
         until cxr-07-promote-champion (manual, named approver) makes it champion
ci/trigger_cai_pipeline.py runs the chains one after the other (the federal quota fits one
2 vCPU workload beside the endpoint); a rejected candidate stops its own chain only.
cxr-setup-data and cxr-setup-nih run once on a new project; cxr-06-score-studies is started
by the lakehouse DAG (cde/dags/cxr_dag.py) for one business date at a time.
"""
from __future__ import annotations

import json
import os

# The federal workbench's Python 3.11 runtime, for jobs created from a laptop (ci/setup_cai.py)
RUNTIME = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-jupyterlab-python3.11-standard:2026.08.1-b5"

def _chain(prefix: str, model: str, parent: str | None) -> list[dict]:
    """build-features -> train-validate -> kpi-gate -> deploy for one model (env CXR_MODEL)."""
    env = {"CXR_MODEL": model}
    steps = [("01-build-features", "features/build_feature_table.py", 2, 8, 7200),   # 2 vCPU jobs only
             ("02-train-validate", "train/train_validate.py", 2, 8, 3600),
             ("03-kpi-gate", "gate/kpi_gate.py", 1, 2, 600),
             ("04-deploy", "serve/deploy_champion.py", 1, 2, 5400)]   # model build + rollout, 30 min each
    out = []
    for step, script, cpu, mem, timeout in steps:
        name = f"{prefix}-{step}" + ("-champion" if prefix == "cxr" and step == "04-deploy" else "")
        out.append({"name": name, "script": script, "parent": parent, "cpu": cpu, "memory": mem,
                    "timeout": timeout, "schedule": None, "env": env})
        parent = name
    return out


JOBS = [
    # name, script, parent, cpu, memory GB, timeout seconds, cron schedule, job environment
    {"name": "cxr-setup-data", "script": "scripts/fetch_dataset.py", "parent": None,
     "cpu": 2, "memory": 8, "timeout": 7200, "schedule": None},
    {"name": "cxr-setup-nih", "script": "scripts/fetch_nih.py", "parent": None,
     "cpu": 2, "memory": 8, "timeout": 14400, "schedule": None},   # ~13 GB of row groups read; resumes
    {"name": "cxr-00-sync-code", "script": "ci/sync_code.py", "parent": None,
     "cpu": 2, "memory": 8, "timeout": 3600, "schedule": None},          # pip installs torch on a change
    *_chain("cxr", "pneumonia", "cxr-00-sync-code"),
    *_chain("qc", "film_qc", None),
    *_chain("ptx", "pneumothorax", None),
    {"name": "cxr-05-nightly-worklist", "script": "monitor/batch_score.py", "parent": None,
     "cpu": 2, "memory": 8, "timeout": 3600, "schedule": "0 2 * * *"},
    {"name": "cxr-06-score-studies", "script": "lakehouse/score_studies.py", "parent": None,
     "cpu": 2, "memory": 8, "timeout": 3600, "schedule": None},           # started by the lakehouse DAG
    {"name": "cxr-07-promote-champion", "script": "serve/promote_champion.py", "parent": None,
     "cpu": 1, "memory": 2, "timeout": 5400, "schedule": None},   # manual: CXR_MODEL, CXR_APPROVED_BY
]

CHAINS = {
    "pneumonia": ["cxr-00-sync-code"] + [j["name"] for j in JOBS if j["name"].startswith("cxr-0") and j.get("env")],
    "film_qc": [j["name"] for j in JOBS if j["name"].startswith("qc-")],
    "pneumothorax": [j["name"] for j in JOBS if j["name"].startswith("ptx-")],
}
CHAIN = CHAINS["pneumonia"]
GATE_JOBS = {n for c in CHAINS.values() for n in c if "-03-kpi-gate" in n}
GATE_JOB = "cxr-03-kpi-gate"
SCORE_JOB = "cxr-06-score-studies"
PROMOTE_JOB = "cxr-07-promote-champion"


def resolve_runtime(client, configured: str = "") -> str:
    """The runtime image for jobs and model builds: the configured one, else the runtime
    of the current session/job (matched on the ML_RUNTIME_* variables CAI sets)."""
    if configured:
        return configured
    wanted = {k: os.environ.get(f"ML_RUNTIME_{k.upper()}") for k in ("kernel", "edition", "editor")}
    if not all(wanted.values()):
        raise SystemExit("Set cai.runtime_identifier in config/pipeline.yaml (Project Settings > Runtime) "
                         "or run this from a CAI session/job so the runtime can be detected.")
    found, token = [], None
    while True:   # the API pages its results: the session's runtime can be past the first page
        kw = {"page_token": token} if token else {}
        resp = client.list_runtimes(search_filter=json.dumps(wanted), page_size=100, **kw)
        found += list(resp.runtimes or [])
        token = getattr(resp, "next_page_token", None)
        if not token:
            break
    full = os.environ.get("ML_RUNTIME_FULL_VERSION")
    exact = [r for r in found if full and getattr(r, "full_version", None) == full]
    pick = (exact or sorted(found, key=lambda r: getattr(r, "full_version", "") or ""))
    if not pick:
        raise SystemExit(f"No runtime matches {wanted}; set cai.runtime_identifier in config/pipeline.yaml.")
    if not exact:
        print(f"WARNING runtime {full} of this session not found among {len(found)} runtimes; "
              f"using the newest match. Set cai.runtime_identifier to pin it.")
    return pick[-1].image_identifier
