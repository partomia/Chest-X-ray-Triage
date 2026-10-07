"""
CAI Model endpoint (Models > New Model > file serve/predict.py, function predict;
built and deployed by the deploy jobs).

Request : {"image_b64": "<base64 JPEG/PNG>", "age_years": 4}        (age optional)
Response: {"priority": "P1", "triage_model": "pneumonia",
           "probability_pneumonia": 0.93, "threshold": 0.41,         (as before there were other models)
           "film_qc": {"unsuitable": false, "probability": 0.01, ...},
           "findings": {"pneumonia": {"probability": 0.93, "priority": "P1", "in_scope": true, ...}, ...},
           "quality_flags": [], "feature_version": "1.0.0", "model_git_sha": "..."}

One ViT pass per film; every head (a model's champion, and in batch scoring its silent trial)
scores the same embedding, so all heads must share the serving code's feature hash.

The worklist priority comes from champions only:
  film_qc champion says the film is unfit  -> NA (no AI triage: read in arrival order)
  else the most urgent band of the champion finding models whose intended population
  (models.<name>.population) includes the patient's age; none in scope -> NA
A silent-trial head is scored and returned with stage silent_trial, never used for priority.

Uses features/feature_logic.py - the SAME code that built the training tables.
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

from common import DEFAULT_MODEL, ROOT, in_scope, load_config, model_names  # noqa: E402
from evaluate.metrics import BAND_RANK, priority_band  # noqa: E402
from features.feature_logic import compute_features, feature_hash, load_image, make_embedder, model_matrix  # noqa: E402
from lakehouse.store import model_version  # noqa: E402

try:
    import cml.models_v1 as models

    cml_model = models.cml_model
except ImportError:  # outside CAI (tests, app, batch job)
    def cml_model(fn):
        return fn

QC_MODEL = "film_qc"
_CFG = load_config(model=DEFAULT_MODEL)
_F = _CFG["features"]
_HASH = feature_hash(_F)


def _load_heads() -> list[dict]:
    heads = []
    for name in model_names(_CFG):
        cfg = load_config(model=name)
        for stage, key in (("champion", "champion_dir"), ("silent_trial", "trial_dir")):
            d = ROOT / cfg["serving"][key]
            if not (d / "model_meta.json").exists():
                continue
            meta = json.loads((d / "model_meta.json").read_text())
            # Fail fast if the serving code would compute different features than training did.
            if meta["feature_hash"] != _HASH:
                raise RuntimeError(f"{name} {stage} was trained on a different feature definition than this "
                                   "code. Redeploy via the pipeline.")
            heads.append({"name": name, "stage": stage, "meta": meta, "model": joblib.load(d / "model.joblib"),
                          "population": cfg["model"]["population"], "version": model_version(meta)})
    return heads


_HEADS = _load_heads()
_CHAMPIONS = {h["name"]: h for h in _HEADS if h["stage"] == "champion"}
if DEFAULT_MODEL not in _CHAMPIONS:
    raise RuntimeError(f"No champion in {ROOT / _CFG['serving']['champion_dir']}: run the pipeline "
                       "(jobs cxr-00 .. cxr-04) first.")
_META = _CHAMPIONS[DEFAULT_MODEL]["meta"]
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


def head_scores(head: dict, emb, qual, ages) -> list[dict]:
    m = head["meta"]
    probs = head["model"].predict_proba(model_matrix(emb, qual, m["model_inputs"]))[:, 1]
    return [{"probability": round(float(p), 4), "threshold": round(m["threshold"], 4),
             "positive": bool(p >= m["threshold"]),
             "priority": priority_band(float(p), m["threshold"], m["p1_probability"]),
             "in_scope": in_scope(head["population"], age), "stage": head["stage"],
             "model_version": head["version"]} for p, age in zip(probs, ages)]


def triage(findings: dict, qc: dict | None) -> tuple[str, str | None]:
    """(worklist band, the model that set it) from the champions' results."""
    if qc and qc["unsuitable"]:
        return "NA", None
    live = [(BAND_RANK[r["priority"]], -r["probability"], name) for name, r in findings.items()
            if r["stage"] == "champion" and r["in_scope"] and name != QC_MODEL]
    if not live:
        return "NA", None
    rank, _, name = min(live)
    return findings[name]["priority"], name


def score_images(images, ages=None, include_trial: bool = False) -> list[dict]:
    ages = list(ages) if ages is not None else [None] * len(images)
    emb, qual = compute_features(images, _EMB, _F["image_size"])
    per_head = {(h["name"], h["stage"]): head_scores(h, emb, qual, ages) for h in _HEADS
                if h["stage"] == "champion" or include_trial}
    out = []
    for i, q in enumerate(qual):
        findings, trials = {}, {}
        for (name, stage), rows in per_head.items():
            (findings if stage == "champion" else trials)[name] = rows[i]
        qc_row = findings.pop(QC_MODEL, None)
        qc = {"unsuitable": qc_row["positive"], "probability": qc_row["probability"],
              "model_version": qc_row["model_version"]} if qc_row else None
        band, by = triage(findings, qc)
        pneu = findings[DEFAULT_MODEL]
        out.append({
            "priority": band, "triage_model": by,
            "probability_pneumonia": pneu["probability"], "threshold": pneu["threshold"],
            "film_qc": qc, "findings": findings, **({"silent_trial": trials} if include_trial else {}),
            "quality_flags": quality_flags(q, _META["quality_reference"]),
            "feature_version": _META["feature_version"], "model_git_sha": _META["git_sha"][:7],
        })
    return out


@cml_model
def predict(args: dict) -> dict:
    """CAI model function."""
    if not isinstance(args, dict) or "image_b64" not in args:
        return {"error": "send {'image_b64': '<base64 image>', 'age_years': <optional int>}"}
    try:
        img = load_image(args["image_b64"])
    except Exception as e:
        return {"error": f"could not decode image_b64: {e}"}
    age = args.get("age_years")
    return score_images([img], [int(age) if age is not None else None])[0]
