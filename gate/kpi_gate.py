"""
Job 3 - kpi-gate

Promotes nothing by itself. It decides. PASS = the script ends normally and the
dependent deploy job runs; FAIL = exit code 1, the CAI job chain stops here, the
current champion keeps serving and the GitHub workflow reports the failure.

Rules = absolute KPIs on the TEST split + non-regression against the champion
+ the candidate's features match the serving code and the full feature table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # CAI job kernels run the script without __file__; cwd is the project
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

from common import ROOT, finish, load_config, parse_args  # noqa: E402
from features.feature_logic import feature_hash  # noqa: E402


def evaluate_gate(candidate: dict, champion: dict | None, gcfg: dict, code_hash: str) -> list[dict]:
    t = candidate["metrics"]["test"]
    checks = [
        {"check": "AUROC (test)", "value": t["auroc"], "rule": f">= {gcfg['min_auroc']}",
         "passed": t["auroc"] >= gcfg["min_auroc"]},
        {"check": "Sensitivity @ operating threshold (test)", "value": t["sensitivity"],
         "rule": f">= {gcfg['min_sensitivity']}", "passed": t["sensitivity"] >= gcfg["min_sensitivity"]},
        {"check": "Specificity @ operating threshold (test)", "value": t["specificity"],
         "rule": f">= {gcfg['min_specificity']}", "passed": t["specificity"] >= gcfg["min_specificity"]},
        {"check": "Brier score (calibration, test)", "value": t["brier"],
         "rule": f"<= {gcfg['max_brier']}", "passed": t["brier"] <= gcfg["max_brier"]},
        {"check": "Trained on the full feature table", "value": candidate.get("feature_table_limit", 0),
         "rule": "no --limit", "passed": not candidate.get("feature_table_limit")},
    ]
    if gcfg.get("require_feature_hash_match", True):
        checks.append({"check": "Feature hash matches serving code", "value": candidate["feature_hash"],
                       "rule": f"== {code_hash}", "passed": candidate["feature_hash"] == code_hash})
    if champion:
        floor = champion["metrics"]["test"]["auroc"] - gcfg["max_auroc_regression"]
        checks.append({"check": "Non-regression vs champion AUROC", "value": t["auroc"],
                       "rule": f">= {floor:.4f} (champion {champion['metrics']['test']['auroc']:.4f})",
                       "passed": t["auroc"] >= floor})
    else:
        checks.append({"check": "Non-regression vs champion AUROC", "value": None,
                       "rule": "no champion yet - first model", "passed": True})
    return checks


def main() -> int:
    parse_args(argparse.ArgumentParser())
    cfg = load_config()
    cand_dir = ROOT / cfg["serving"]["candidate_dir"]
    champ_meta = ROOT / cfg["serving"]["champion_dir"] / "model_meta.json"
    candidate = json.loads((cand_dir / "model_meta.json").read_text())
    champion = json.loads(champ_meta.read_text()) if champ_meta.exists() else None

    checks = evaluate_gate(candidate, champion, cfg["gate"], feature_hash(cfg["features"]))
    passed = all(c["passed"] for c in checks)

    print(f"{'CHECK':50} {'VALUE':>10}  RULE")
    for c in checks:
        v = f"{c['value']:.4f}" if isinstance(c["value"], float) else str(c["value"])[:10]
        print(f"{('PASS ' if c['passed'] else 'FAIL ') + c['check']:50} {v:>10}  {c['rule']}")
    print(f"\nKPI GATE: {'PASSED' if passed else 'FAILED'}")

    result = {"passed": passed, "checks": checks, "mlflow_run_id": candidate.get("mlflow_run_id"),
              "candidate_git_sha": candidate["git_sha"]}
    (cand_dir / "gate_result.json").write_text(json.dumps(result, indent=2))

    if candidate.get("mlflow_run_id"):
        try:
            import mlflow
            with mlflow.start_run(run_id=candidate["mlflow_run_id"]):
                mlflow.log_metric("kpi_gate_passed", 1.0 if passed else 0.0)
                mlflow.set_tag("kpi_gate", "PASSED" if passed else "FAILED")
        except Exception as e:  # logging must never change the gate decision
            print(f"(mlflow tag skipped: {e})")
    return 0 if passed else 1


if __name__ == "__main__":
    finish(main())
