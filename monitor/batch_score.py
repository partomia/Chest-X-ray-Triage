"""
Job 5 - nightly-worklist  (scheduled, independent of the CI chain)

Scores every film in data/incoming/, writes a prioritised worklist, and checks
feature drift (PSI of the quality features vs the TRAIN split of the champion's
feature table). Uses the same feature logic and the same champion as the endpoint.
Drift is reported, not enforced: the job fails only if scoring fails.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import ROOT, finish, load_config, load_feature_table, parse_args  # noqa: E402
from features.feature_logic import QUALITY_FEATURES, load_image, quality_features  # noqa: E402


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / max(len(actual), 1)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_level(drift: dict, m: dict, n_films: int) -> str:
    if n_films < m.get("min_films_for_drift", 0):
        return "TOO_FEW"
    worst = max(drift.values()) if drift else 0.0
    return "ALERT" if worst >= m["psi_alert"] else "WARN" if worst >= m["psi_warn"] else "OK"


def main() -> int:
    parse_args(argparse.ArgumentParser())
    from serve.predict import _META, score_images  # loads champion once

    cfg = load_config()
    files = sorted(p for p in (ROOT / cfg["data"]["incoming_dir"]).rglob("*")
                   if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    if not files:
        print("no incoming films")
        return 0

    results, qual = [], []
    for i in range(0, len(files), 64):
        batch = files[i:i + 64]
        imgs = [load_image(p) for p in batch]
        qual.extend(quality_features(im) for im in imgs)
        for p, r in zip(batch, score_images(imgs)):
            results.append({"film": str(p.relative_to(ROOT)), **r})
    wl = pd.DataFrame(results).sort_values("probability_pneumonia", ascending=False)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "outputs" / "worklist"
    out.mkdir(parents=True, exist_ok=True)
    wl.to_csv(out / f"worklist_{ts}.csv", index=False)

    cur = pd.DataFrame(qual)
    ref_df, _ = load_feature_table(cfg, _META["feature_version"])
    ref = ref_df[ref_df.split == "train"]
    drift = {c: psi(ref[c].to_numpy(), cur[c].to_numpy()) for c in QUALITY_FEATURES}
    report = {"scored": len(wl), "priority_counts": wl["priority"].value_counts().to_dict(),
              "psi": drift, "drift_level": drift_level(drift, cfg["monitoring"], len(files)),
              "champion_git_sha": _META["git_sha"][:7], "run_ts_utc": ts}
    (out / f"drift_{ts}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    finish(main())
