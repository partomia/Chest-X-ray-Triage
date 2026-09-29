"""Unit tests that run on the GitHub runner (no torch, no data)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate.metrics import choose_threshold, compute_metrics, priority_band  # noqa: E402
from features.feature_logic import (  # noqa: E402
    QUALITY_FEATURES, feature_hash, model_matrix, preprocess, quality_features,
)
from gate.kpi_gate import evaluate_gate  # noqa: E402

FCFG = {"backbone": "google/vit-base-patch16-224", "backbone_revision": "main",
        "image_size": 224, "embedding_pooling": "cls"}


def _img(seed=0, size=(300, 260)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size, dtype=np.uint8), "L")


def test_preprocess_is_square_and_deterministic():
    a, b = preprocess(_img(), 224), preprocess(_img(), 224)
    assert a.size == (224, 224)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_quality_features_complete_and_stable():
    q1, q2 = quality_features(_img(1)), quality_features(_img(1))
    assert list(q1) == QUALITY_FEATURES
    assert q1 == q2
    assert 0 <= q1["q_mean"] <= 1


def test_feature_hash_changes_with_definition():
    h = feature_hash(FCFG)
    assert h == feature_hash(dict(FCFG))
    assert h != feature_hash({**FCFG, "image_size": 256})


def test_model_matrix_respects_inputs():
    emb = np.ones((2, 4), dtype=np.float32)
    q = [quality_features(_img(i)) for i in range(2)]
    assert model_matrix(emb, q, ["embedding"]).shape == (2, 4)
    assert model_matrix(emb, q, ["embedding", "quality"]).shape == (2, 4 + len(QUALITY_FEATURES))


def test_threshold_meets_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.4 + rng.random(2000) * 0.6, 0, 1)
    thr = choose_threshold(y, p, 0.95)
    assert compute_metrics(y, p, thr)["sensitivity"] >= 0.95


def test_priority_bands():
    assert priority_band(0.9, 0.4, 0.85) == "P1"
    assert priority_band(0.5, 0.4, 0.85) == "P2"
    assert priority_band(0.88, 0.92, 0.85) == "P3"   # threshold above p1: below it is never P1
    assert priority_band(0.93, 0.92, 0.85) == "P1"
    assert priority_band(0.1, 0.4, 0.85) == "P3"


GATE = {"min_auroc": 0.93, "min_sensitivity": 0.92, "min_specificity": 0.6,
        "max_brier": 0.12, "max_auroc_regression": 0.01, "require_feature_hash_match": True}


def _meta(auroc, sens=0.95, spec=0.7, brier=0.08, h="abc"):
    return {"feature_hash": h, "metrics": {"test": {"auroc": auroc, "sensitivity": sens,
                                                    "specificity": spec, "brier": brier}}}


def test_gate_passes_good_first_model():
    assert all(c["passed"] for c in evaluate_gate(_meta(0.96), None, GATE, "abc"))


def test_gate_blocks_regression_and_low_sensitivity():
    assert not all(c["passed"] for c in evaluate_gate(_meta(0.95), _meta(0.97), GATE, "abc"))
    assert not all(c["passed"] for c in evaluate_gate(_meta(0.96, sens=0.85), None, GATE, "abc"))


def test_gate_blocks_feature_mismatch():
    assert not all(c["passed"] for c in evaluate_gate(_meta(0.96, h="old"), None, GATE, "new"))


def test_gate_blocks_model_trained_on_limited_table():
    assert not all(c["passed"] for c in evaluate_gate({**_meta(0.96), "feature_table_limit": 40}, None, GATE, "abc"))


def test_stub_embedder_is_deterministic_and_hashes_apart():
    from features.feature_logic import StubEmbedder, make_embedder

    stub_cfg = {**FCFG, "backbone": "stub:pooled-pixels", "batch_size": 8}
    emb = make_embedder(stub_cfg)
    assert isinstance(emb, StubEmbedder)
    imgs = [preprocess(_img(i), 224) for i in range(3)]
    a, b = emb.embed(imgs), emb.embed(imgs)
    assert a.shape == (3, StubEmbedder.grid ** 2) and np.array_equal(a, b)
    assert feature_hash(stub_cfg) != feature_hash(FCFG)


def test_psi_levels():
    from monitor.batch_score import drift_level, psi

    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    m = {"psi_warn": 0.10, "psi_alert": 0.25, "min_films_for_drift": 30}
    same, shifted = psi(ref, rng.normal(0, 1, 500)), psi(ref, rng.normal(1.0, 1, 500))
    assert same < 0.10 < 0.25 < shifted
    assert drift_level({"q": same}, m, 500) == "OK"
    assert drift_level({"q": shifted}, m, 500) == "ALERT"
    assert drift_level({"q": shifted}, m, 10) == "TOO_FEW"


def test_config_overlay_merges(monkeypatch):
    from common import load_config

    base = load_config()
    monkeypatch.setenv("CXR_CONFIG_OVERLAY", "config/ci.yaml, config/ci-gate-fail.yaml")
    cfg = load_config()
    assert cfg["features"]["backbone"].startswith("stub:")
    assert cfg["features"]["image_size"] == base["features"]["image_size"]   # untouched keys kept
    assert cfg["gate"]["min_auroc"] > 1 and cfg["gate"]["max_brier"] == base["gate"]["max_brier"]
    assert base["features"]["backbone"] == "google/vit-base-patch16-224"
