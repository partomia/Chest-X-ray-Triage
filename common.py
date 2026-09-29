"""Small shared helpers: config loading, paths, feature-table I/O, CAI job plumbing."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("CXR_ROOT", Path(__file__).resolve().parent))


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path = "config/pipeline.yaml") -> dict:
    """config/pipeline.yaml, then each file in CXR_CONFIG_OVERLAY (comma-separated) merged on top."""
    with open(ROOT / path) as f:
        cfg = yaml.safe_load(f)
    for overlay in filter(None, (s.strip() for s in os.environ.get("CXR_CONFIG_OVERLAY", "").split(","))):
        with open(ROOT / overlay) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def feature_table_dir(cfg: dict, version: str | None = None) -> Path:
    v = version or cfg["features"]["version"]
    return ROOT / cfg["features"]["store_dir"] / f"v{v}"


def load_feature_table(cfg: dict, version: str | None = None):
    import pandas as pd

    d = feature_table_dir(cfg, version)
    manifest = json.loads((d / "manifest.json").read_text())
    df = pd.read_parquet(d / "features.parquet")
    return df, manifest


def git_sha() -> str:
    """HEAD of the project; outputs/git_sha.txt (written by sync-code) where git is unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        p = ROOT / "outputs" / "git_sha.txt"
        return p.read_text().strip() if p.exists() else "unknown"


def parse_args(ap: argparse.ArgumentParser) -> argparse.Namespace:
    """CAI's Jupyter-kernel job runtime passes extra arguments (-f <connection file>); ignore them."""
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"(ignoring arguments: {unknown})", flush=True)
    return args


def finish(rc: int) -> None:
    """End a job script. CAI's job kernel wrapper reports ANY SystemExit, even sys.exit(0), as a
    failure, so only exit explicitly on a real failure and fall off the end on success."""
    if rc:
        sys.exit(rc)
