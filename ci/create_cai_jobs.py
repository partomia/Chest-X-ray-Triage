"""
One-time setup, run from a CAI session terminal: creates the six CAI jobs of
ci/cai_jobs.py with their dependencies, resource profiles, timeouts and the
nightly schedule, then prints the IDs. Jobs that already exist (by name) are
left as they are, so it is safe to re-run.

  python ci/create_cai_jobs.py            # create what is missing
  python ci/create_cai_jobs.py --dry-run  # show what would be created
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

from ci.cai_jobs import JOBS, resolve_runtime  # noqa: E402
from common import finish, load_config, parse_args  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = parse_args(ap)

    import cmlapi

    client = cmlapi.default_client()
    pid = os.environ["CDSW_PROJECT_ID"]
    runtime = resolve_runtime(client, load_config()["cai"]["runtime_identifier"])
    print(f"project {pid}, runtime {runtime}")

    existing = {j.name: j.id for j in client.list_jobs(pid, page_size=200).jobs}
    ids = dict(existing)
    for j in JOBS:
        if j["name"] in existing:
            print(f"  exists   {j['name']:28} {existing[j['name']]}")
            continue
        body = cmlapi.CreateJobRequest(
            project_id=pid, name=j["name"], script=j["script"], runtime_identifier=runtime,
            cpu=j["cpu"], memory=j["memory"], timeout=j["timeout"],
        )
        if j["parent"]:
            body.parent_job_id = ids[j["parent"]] if not args.dry_run else "<parent>"
        if j["schedule"]:
            body.schedule = j["schedule"]
        if args.dry_run:
            print(f"  would create {j['name']:24} {j['script']} parent={j['parent']} schedule={j['schedule']}")
            ids[j["name"]] = "<new>"
            continue
        job = client.create_job(body, pid)
        ids[j["name"]] = job.id
        print(f"  created  {j['name']:28} {job.id}")

    print("\nGitHub secrets for .github/workflows/cai-mlops.yml:")
    print(f"  CAI_URL        = https://{os.environ.get('CDSW_DOMAIN', '<workbench domain>')}")
    print(f"  CAI_PROJECT_ID = {pid}")
    print("  CAI_API_KEY    = <User Settings > API Keys > API v2 key>")
    return 0


if __name__ == "__main__":
    finish(main())
