"""End to end on synthetic films with the stub embedder, in a throwaway project root.

build-features -> (idempotent re-run) -> train-validate -> kpi-gate (pass, then
fail on cue) -> promote / rollback -> nightly worklist -> endpoint -> app.
Each stage runs as a subprocess, the way a CAI job runs it.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("cxr")
    shutil.copytree(REPO / "config", r / "config")
    subprocess.run([sys.executable, str(REPO / "scripts/make_synthetic_cxr.py"), "--out", str(r / "data/synthetic"),
                    "--scale", "0.5", "--incoming", "6"], check=True, capture_output=True)
    return r


def sh(root, script, *args, overlay="config/ci.yaml", code=None):
    env = {**os.environ, "CXR_ROOT": str(root), "CXR_CONFIG_OVERLAY": overlay}
    cmd = [sys.executable, str(REPO / script), *args] if script.endswith(".py") else [sys.executable, "-c", script]
    p = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True, timeout=600)
    return p if code is None else (p.returncode, p.stdout + p.stderr)


def test_chain(root):
    p = sh(root, "features/build_feature_table.py", "-f", "/tmp/kernel.json")   # the CAI kernel's extra argument
    assert p.returncode == 0, p.stdout + p.stderr
    manifest = json.loads(next((root / "feature_store/ci_features").glob("v*/manifest.json")).read_text())
    assert all(c["passed"] for c in manifest["data_checks"])
    assert set(manifest["rows_by_split"]) == {"train", "val", "test"}

    p = sh(root, "features/build_feature_table.py")
    assert p.returncode == 0 and "skipping" in p.stdout

    p = sh(root, "train/train_validate.py")
    assert p.returncode == 0, p.stdout + p.stderr
    meta = json.loads((root / "outputs/ci/candidate/model_meta.json").read_text())
    assert meta["metrics"]["test"]["sensitivity"] > 0 and meta["feature_table_limit"] == 0

    p = sh(root, "gate/kpi_gate.py")
    assert p.returncode == 0 and "KPI GATE: PASSED" in p.stdout, p.stdout + p.stderr
    p = sh(root, "gate/kpi_gate.py", overlay="config/ci.yaml,config/ci-gate-fail.yaml")
    assert p.returncode == 1 and "KPI GATE: FAILED" in p.stdout
    assert not json.loads((root / "outputs/ci/candidate/gate_result.json").read_text())["passed"]


def test_promote_refuses_failed_gate_then_promotes_and_rolls_back(root):
    promote = ("from serve.deploy_champion import promote, rollback; from common import load_config; "
               "cfg = load_config(); meta, arch = promote(cfg); print('PROMOTED', arch)")
    rc, out = sh(root, promote, code=True)
    assert rc != 0 and "refusing to deploy" in out   # the last gate run above failed

    assert sh(root, "gate/kpi_gate.py").returncode == 0
    rc, out = sh(root, promote, code=True)
    assert rc == 0 and "PROMOTED None" in out
    champ = root / "outputs/ci/champion"
    assert (champ / "model_meta.json").exists()

    # the next promotion archives this champion; a failed deploy puts it back
    (champ / "serving.marker").write_text("previous champion")
    rc, out = sh(root, promote + "; assert not (arch / '..' / '..' / 'champion' / 'serving.marker').exists()"
                                 "; rollback(cfg, arch)", code=True)
    assert rc == 0 and "rolled back" in out, out
    assert (champ / "serving.marker").exists() and (champ / "model_meta.json").exists()


def test_nightly_worklist_endpoint_and_app(root):
    p = sh(root, "monitor/batch_score.py")
    assert p.returncode == 0, p.stdout + p.stderr
    drift = json.loads(sorted((root / "outputs/worklist").glob("drift_*.json"))[-1].read_text())
    assert drift["scored"] == 6 and drift["drift_level"] == "TOO_FEW"

    film = next((root / "data/synthetic/incoming").glob("*.jpeg"))
    b64 = base64.b64encode(film.read_bytes()).decode()
    rc, out = sh(root, f"import json; from serve.predict import predict; print(json.dumps(predict({{'image_b64': '{b64}'}})))"
                       "; print(json.dumps(predict({})))", code=True)
    assert rc == 0, out
    ok, bad = [json.loads(line) for line in out.strip().splitlines()[-2:]]
    assert ok["priority"] in {"P1", "P2", "P3"} and 0 <= ok["probability_pneumonia"] <= 1
    assert "error" in bad

    app = f"""
import os
os.environ['CXR_ROOT'] = {str(root)!r}
from pathlib import Path
from streamlit.testing.v1 import AppTest
os.chdir({str(REPO)!r})
at = AppTest.from_file('app/app.py', default_timeout=180)
at.run()
assert not at.exception, at.exception
assert any('Arrival order' in s.value for s in at.subheader)
at.button[0].click().run()
assert not at.exception, at.exception
assert len(at.metric) == 3
at.checkbox[0].check().run()
assert not at.exception, at.exception
at.button[-1].click().run()
assert not at.exception, at.exception
assert Path({str(root)!r}, 'outputs/feedback/feedback.csv').exists()
print('APP OK')
"""
    rc, out = sh(root, app, code=True)
    assert rc == 0 and "APP OK" in out, out
