"""
Job cxr-07-promote-champion (manual): a model in silent trial goes live.

Run it with the job run's environment
    CXR_MODEL=pneumothorax  CXR_APPROVED_BY=<clinical lead>  [CXR_APPROVAL_NOTE=<text>]
(python ci/setup_cai.py --run cxr-07-promote-champion --env CXR_MODEL=...,CXR_APPROVED_BY=...)

  1. the silent trial's evidence, from the lakehouse: gold.daily_model_summary (built by CDE
     from every live study the trial model scored and its signed report) summed over the
     business dates scored by THIS model version
  2. go-live criteria (models.<name>.go_live): enough days, enough reported positives, and
     sensitivity / specificity on them; not met -> PROMOTION_REFUSED, exit 1, nothing changes
  3. the trial model becomes the champion (the previous champion is archived), the endpoint
     is rebuilt to serve it (rolled back on failure), and the registry gets a new version
     with stage champion, the approver and the evidence; ref.model_event records PROMOTED

A human approves; the criteria only decide whether there is enough evidence to ask.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # CAI job kernels run the script without __file__; cwd is the project
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

from common import ROOT, finish, load_config, parse_args  # noqa: E402


def trial_evidence(store, model: str, version: str) -> dict:
    t = store.t("gold.daily_model_summary")
    rows = store.query(f"SELECT business_date, tp, fn, fp, tn FROM {t} WHERE model = '{model}' "
                       f"AND model_version = '{version}' AND stage = 'silent_trial'")
    tp, fn, fp, tn = (sum(int(r[k] or 0) for r in rows) for k in ("tp", "fn", "fp", "tn"))
    return {"days": len({str(r["business_date"]) for r in rows}), "positives": tp + fn, "negatives": tn + fp,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
            "specificity": round(tn / (tn + fp), 4) if tn + fp else None}


def criteria(ev: dict, g: dict) -> list[dict]:
    def check(name, value, rule, ok):
        return {"check": name, "value": value, "rule": rule, "passed": bool(ok)}

    return [check("days in silent trial", ev["days"], f">= {g['min_days']}", ev["days"] >= g["min_days"]),
            check("reported positives", ev["positives"], f">= {g['min_positives']}",
                  ev["positives"] >= g["min_positives"]),
            check("sensitivity (live studies)", ev["sensitivity"], f">= {g['min_sensitivity']}",
                  ev["sensitivity"] is not None and ev["sensitivity"] >= g["min_sensitivity"]),
            check("specificity (live studies)", ev["specificity"], f">= {g['min_specificity']}",
                  ev["specificity"] is not None and ev["specificity"] >= g["min_specificity"])]


def main() -> int:
    parse_args(argparse.ArgumentParser())
    cfg = load_config()
    name = cfg["model"]["name"]
    approver = os.environ.get("CXR_APPROVED_BY", "").strip()
    if not approver:
        print("set CXR_APPROVED_BY (who signs the model off) in the job run's environment")
        return 1
    if "go_live" not in cfg["model"]:
        print(f"model {name} has no go_live criteria: it is not promoted from a silent trial")
        return 1
    trial = ROOT / cfg["serving"]["trial_dir"]
    if not (trial / "model_meta.json").exists():
        print(f"model {name}: no silent-trial model in {trial}")
        return 1
    meta = json.loads((trial / "model_meta.json").read_text())

    from lakehouse.publish import publish_model_event
    from lakehouse.store import ImpalaStore, model_version
    from serve.deploy_champion import go_live, install

    version = model_version(meta)
    ev = trial_evidence(ImpalaStore(), name, version)
    checks = criteria(ev, cfg["model"]["go_live"])
    print(f"model {name} {version}: silent-trial evidence {json.dumps(ev)}")
    for c in checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'} {c['check']}: {c['value']} (rule {c['rule']})")
    gate = {"passed": all(c["passed"] for c in checks), "checks": checks}
    if not gate["passed"]:
        publish_model_event("PROMOTION_REFUSED", meta, gate, detail=f"requested by {approver}", stage="silent_trial")
        print("GO-LIVE CRITERIA NOT MET: the model stays in silent trial")
        return 1

    meta_champ, archived = install(cfg, trial, "champion")
    note = os.environ.get("CXR_APPROVAL_NOTE", "")
    tags = {"approved_by": approver, "approval_note": note[:200] or None,
            "trial_days": ev["days"], "trial_positives": ev["positives"],
            "trial_sensitivity": ev["sensitivity"], "trial_specificity": ev["specificity"]}
    rc = go_live(cfg, meta_champ, archived, "champion", {k: v for k, v in tags.items() if v is not None},
                 event="PROMOTED")
    if rc == 0:
        shutil.rmtree(trial)   # it is the champion now (archived copies keep every earlier one)
        print(f"PROMOTED: {name} {version} is the champion, approved by {approver}")
    return rc


if __name__ == "__main__":
    finish(main())
