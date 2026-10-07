#!/usr/bin/env python3
"""
One-off CAI setup over the API v2, from a laptop. Idempotent: each step finds by
name first and only creates what is missing.

  1. Project CAI_PROJECT_NAME from this GitHub repo, if absent; waits for the clone.
  2. Project environment variables: HF_HOME, and CXR_IMPALA_USER / CXR_IMPALA_PASSWORD
     (the workload user, for the lakehouse jobs; from the caller's environment, never printed).
  3. The jobs of ci/cai_jobs.py with their parents, schedule, timeout and size; an
     existing job is resized to match.
  4. --app: the application CXR Triage Worklist (needs a champion, i.e. a first chain run).

Prints the IDs for the GitHub secrets and the Airflow Variables.

  set -a; source .env; set +a      # CXR_CAI_HOST, CXR_CAI_API_KEY, CXR_IMPALA_*
  python ci/setup_cai.py --dry-run
  python ci/setup_cai.py
  python ci/setup_cai.py --run cxr-setup-data      # start a job and follow it
  python ci/setup_cai.py --app
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.cai_jobs import JOBS, RUNTIME, SCORE_JOB  # noqa: E402

CAI_PROJECT_NAME = "rsingh-chest-x-ray-triage"
GIT_URL = "https://github.com/partomia/Chest-X-ray-Triage"
PROJECT_ENV = {"HF_HOME": "/home/cdsw/.hf_cache"}
PROJECT_ENV_FROM_CALLER = ("CXR_IMPALA_USER", "CXR_IMPALA_PASSWORD")
APP = {"name": "CXR Triage Worklist", "subdomain": "rsingh-cxr-triage", "script": "app/launch_app.py",
       "cpu": 2, "memory": 8, "description": "Radiology worklist: triage, heatmap, radiologist read"}


class Workbench:
    def __init__(self, url: str, key: str):
        self.base = f"{url.rstrip('/')}/api/v2"
        self.h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def __call__(self, method: str, path: str, params=None, body=None) -> dict:
        r = requests.request(method, self.base + path, params=params, json=body, headers=self.h, timeout=120)
        if r.status_code >= 400:
            raise SystemExit(f"{method} {path}: HTTP {r.status_code} {r.text[:300]}")
        return r.json() if r.text else {}


def find_project(wb: Workbench, name: str = CAI_PROJECT_NAME) -> dict | None:
    found = wb("GET", "/projects", params={"search_filter": json.dumps({"name": name}), "page_size": 100})
    return next((p for p in found.get("projects", []) if p["name"] == name), None)


def ensure_project(wb: Workbench, dry_run: bool) -> dict | None:
    project = find_project(wb)
    if project:
        print(f"project {CAI_PROJECT_NAME}: exists ({project['id']})")
        return project
    if dry_run:
        print(f"project {CAI_PROJECT_NAME}: would create from {GIT_URL}")
        return None
    project = wb("POST", "/projects", body={
        "name": CAI_PROJECT_NAME, "template": "git", "git_url": GIT_URL, "visibility": "private",
        "default_project_engine_type": "ml_runtime",
        "description": "Chest X-ray triage on Cloudera AI and the lakehouse (github.com/partomia/Chest-X-ray-Triage)"})
    print(f"project {CAI_PROJECT_NAME}: created ({project['id']}), cloning", end="", flush=True)
    for _ in range(60):
        status = str(wb("GET", f"/projects/{project['id']}").get("creation_status", "")).lower()
        if status in ("success", "succeeded", ""):
            break
        if "fail" in status or "error" in status:
            raise SystemExit(f"\nproject creation {status}")
        print(".", end="", flush=True)
        time.sleep(5)
    print(" done")
    return project


def ensure_env(wb: Workbench, project: dict, dry_run: bool) -> None:
    missing = [k for k in PROJECT_ENV_FROM_CALLER if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"set {missing} in the environment (source .env) first")
    current = json.loads(wb("GET", f"/projects/{project['id']}").get("environment") or "{}")
    wanted = {**PROJECT_ENV, **{k: os.environ[k] for k in PROJECT_ENV_FROM_CALLER}}
    changed = sorted(k for k, v in wanted.items() if current.get(k) != v)
    if not changed:
        print("project environment: up to date")
    elif dry_run:
        print(f"project environment: would set {changed}")
    else:
        wb("PATCH", f"/projects/{project['id']}", body={"environment": json.dumps({**current, **wanted})})
        print(f"project environment: set {changed}")


def job_ids(wb: Workbench, pid: str) -> dict:
    return {j["name"]: j for j in wb("GET", f"/projects/{pid}/jobs", params={"page_size": 200}).get("jobs", [])}


def ensure_jobs(wb: Workbench, project: dict, dry_run: bool) -> dict:
    pid = project["id"]
    existing = job_ids(wb, pid)
    ids = {n: j["id"] for n, j in existing.items()}
    for job in JOBS:     # parents come first in JOBS
        size = {"cpu": job["cpu"], "memory": job["memory"]}
        if job["name"] in existing:
            have = existing[job["name"]]
            if {k: have.get(k) for k in size} == size:
                print(f"job {job['name']}: exists ({have['id']})")
            elif dry_run:
                print(f"job {job['name']}: would resize to {job['cpu']} vCPU / {job['memory']} GB")
            else:
                wb("PATCH", f"/projects/{pid}/jobs/{have['id']}", body=size)
                print(f"job {job['name']}: resized to {job['cpu']} vCPU / {job['memory']} GB")
            continue
        if dry_run:
            print(f"job {job['name']}: would create ({job['script']}, parent {job['parent']}, "
                  f"schedule {job['schedule']}, {job['cpu']} vCPU / {job['memory']} GB)")
            continue
        body = {"name": job["name"], "script": job["script"], **size, "runtime_identifier": RUNTIME,
                "timeout": job["timeout"], "kill_on_timeout": True, "arguments": ""}
        if job["parent"]:
            body["parent_job_id"] = ids[job["parent"]]
        if job["schedule"]:
            body["schedule"] = job["schedule"]
        created = wb("POST", f"/projects/{pid}/jobs", body=body)
        ids[job["name"]] = created["id"]
        print(f"job {job['name']}: created ({created['id']})")
    return ids


def ensure_app(wb: Workbench, project: dict, dry_run: bool) -> None:
    pid = project["id"]
    app = next((a for a in wb("GET", f"/projects/{pid}/applications", params={"page_size": 100})
                .get("applications", []) if a["name"] == APP["name"]), None)
    if app:
        print(f"application {APP['name']}: exists ({app['id']}, {app.get('status')})")
    elif dry_run:
        print(f"application {APP['name']}: would create ({APP['script']}, {APP['cpu']} vCPU / {APP['memory']} GB)")
    else:
        app = wb("POST", f"/projects/{pid}/applications", body={
            "project_id": pid, "name": APP["name"], "subdomain": APP["subdomain"], "script": APP["script"],
            "cpu": APP["cpu"], "memory": APP["memory"], "kernel": "python3", "runtime_identifier": RUNTIME,
            "description": APP["description"]})
        print(f"application {APP['name']}: created ({app['id']}), subdomain {APP['subdomain']}")


def run_job(wb: Workbench, pid: str, job_id: str, name: str, env: dict | None = None, poll: int = 30) -> str:
    run = wb("POST", f"/projects/{pid}/jobs/{job_id}/runs", body={"environment": env or {}})
    print(f"{name}: run {run['id']} started", flush=True)
    t0, last = time.time(), None
    while True:
        time.sleep(poll)
        st = str(wb("GET", f"/projects/{pid}/jobs/{job_id}/runs/{run['id']}").get("status", "")).lower()
        st = st.replace("engine_", "")
        if st != last:
            print(f"{name}: {st} after {time.time() - t0:.0f} s", flush=True)
            last = st
        if st in ("succeeded", "failed", "stopped", "timedout"):
            return st


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--app", action="store_true", help="also create the application (needs a champion)")
    p.add_argument("--run", default="", help="start this job by name and follow it to the end")
    p.add_argument("--env", default="", help="with --run: KEY=VALUE,... for the job run's environment")
    args, _ = p.parse_known_args()
    wb = Workbench(os.environ["CXR_CAI_HOST"], os.environ["CXR_CAI_API_KEY"])
    project = ensure_project(wb, args.dry_run)
    if project is None:
        return 0
    if args.run:
        job = job_ids(wb, project["id"]).get(args.run)
        if not job:
            raise SystemExit(f"no job {args.run}")
        env = dict(kv.split("=", 1) for kv in args.env.split(",") if kv)
        return 0 if run_job(wb, project["id"], job["id"], args.run, env) == "succeeded" else 1
    ensure_env(wb, project, args.dry_run)
    ids = ensure_jobs(wb, project, args.dry_run)
    if args.app:
        ensure_app(wb, project, args.dry_run)
    print(f"\nGitHub secrets: CAI_URL = {os.environ['CXR_CAI_HOST'].rstrip('/')}, CAI_PROJECT_ID = {project['id']}")
    print(f"Airflow Variables: CXR_CAI_PROJECT_ID = {project['id']}, "
          f"CXR_CAI_SCORE_JOB_ID = {ids.get(SCORE_JOB, '(not created)')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
