"""
CAI Model endpoint (Models > New Model > file serve/predict.py, function predict;
built and deployed by job cxr-04-deploy-champion).

Request : {"image_b64": "<base64 JPEG/PNG>"}
Response: {"probability_pneumonia": 0.93, "priority": "P1", "threshold": 0.41,
           "quality_flags": [], "feature_version": "1.0.0", "model_git_sha": "..."}

Uses features/feature_logic.py - the SAME code that built the training table.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # model replicas and job kernels may exec this without __file__
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

import joblib  # noqa: E402

from common import ROOT, load_config  # noqa: E402
from evaluate.metrics import priority_band  # noqa: E402
from features.feature_logic import compute_features, feature_hash, load_image, make_embedder, model_matrix  # noqa: E402

try:
    import cml.models_v1 as models

    cml_model = models.cml_model
except ImportError:  # outside CAI (tests, app, batch job)
    def cml_model(fn):
        return fn

_CFG = load_config()
_CHAMP = ROOT / _CFG["serving"]["champion_dir"]
if not (_CHAMP / "model_meta.json").exists():
    raise RuntimeError(f"No champion in {_CHAMP}: run the pipeline (jobs cxr-00 .. cxr-04) first.")
_META = json.loads((_CHAMP / "model_meta.json").read_text())
_MODEL = joblib.load(_CHAMP / "model.joblib")

# Fail fast if the serving code would compute different features than training did.
if _META["feature_hash"] != feature_hash(_CFG["features"]):
    raise RuntimeError("Champion was trained on a different feature definition than this code. Redeploy via the pipeline.")

_F = _CFG["features"]
_EMB = make_embedder(_F)


def quality_flags(q: dict, ref: dict) -> list[str]:
    """Advisory flags when a film sits outside the training range (p01..p99)."""
    flags = []
    for k, r in ref.items():
        if q[k] < r["p01"]:
            flags.append(f"{k}_LOW")
        elif q[k] > r["p99"]:
            flags.append(f"{k}_HIGH")
    return flags


def score_images(images) -> list[dict]:
    emb, qual = compute_features(images, _EMB, _F["image_size"])
    probs = _MODEL.predict_proba(model_matrix(emb, qual, _META["model_inputs"]))[:, 1]
    return [{
        "probability_pneumonia": round(float(p), 4),
        "priority": priority_band(float(p), _META["threshold"], _META["p1_probability"]),
        "threshold": round(_META["threshold"], 4),
        "quality_flags": quality_flags(q, _META["quality_reference"]),
        "feature_version": _META["feature_version"],
        "model_git_sha": _META["git_sha"][:7],
    } for p, q in zip(probs, qual)]


@cml_model
def predict(args: dict) -> dict:
    """CAI model function."""
    if not isinstance(args, dict) or "image_b64" not in args:
        return {"error": "send {'image_b64': '<base64 image>'}"}
    try:
        img = load_image(args["image_b64"])
    except Exception as e:
        return {"error": f"could not decode image_b64: {e}"}
    return score_images([img])[0]
