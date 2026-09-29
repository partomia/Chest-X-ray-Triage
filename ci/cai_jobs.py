"""
The CAI jobs of this project: one definition read by ci/create_cai_jobs.py
(creates them) and ci/trigger_cai_pipeline.py (runs and follows them).
Pure Python, so the GitHub runner can import it without cmlapi.

CHAIN runs in this order through CAI job dependencies; a dependent job starts
only when its parent succeeds, which makes the KPI gate's exit code a hard stop.
"""
from __future__ import annotations

import json
import os

JOBS = [
    # name, script, parent, cpu, memory GB, timeout seconds, cron schedule
    {"name": "cxr-00-sync-code", "script": "ci/sync_code.py", "parent": None,
     "cpu": 1, "memory": 2, "timeout": 600, "schedule": None},
    {"name": "cxr-01-build-features", "script": "features/build_feature_table.py", "parent": "cxr-00-sync-code",
     "cpu": 4, "memory": 16, "timeout": 7200, "schedule": None},
    {"name": "cxr-02-train-validate", "script": "train/train_validate.py", "parent": "cxr-01-build-features",
     "cpu": 2, "memory": 8, "timeout": 3600, "schedule": None},
    {"name": "cxr-03-kpi-gate", "script": "gate/kpi_gate.py", "parent": "cxr-02-train-validate",
     "cpu": 1, "memory": 2, "timeout": 600, "schedule": None},
    {"name": "cxr-04-deploy-champion", "script": "serve/deploy_champion.py", "parent": "cxr-03-kpi-gate",
     "cpu": 1, "memory": 2, "timeout": 5400, "schedule": None},   # model build + rollout, 30 min each at most
    {"name": "cxr-05-nightly-worklist", "script": "monitor/batch_score.py", "parent": None,
     "cpu": 4, "memory": 8, "timeout": 3600, "schedule": "0 2 * * *"},
]

CHAIN = [j["name"] for j in JOBS if j["name"] != "cxr-05-nightly-worklist"]
GATE_JOB = "cxr-03-kpi-gate"


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
