"""ci/trigger_cai_pipeline.py against a fake CAI API v2 (no network)."""
from datetime import datetime, timedelta, timezone

import pytest

import ci.trigger_cai_pipeline as trig
from ci.cai_jobs import CHAIN, GATE_JOB

T0 = datetime(2026, 9, 29, 8, 0, tzinfo=timezone.utc)


class FakeApi:
    """Job runs appear in chain order; `outcome` maps job name -> final CAI status."""

    def __init__(self, outcome: dict):
        self.outcome = outcome
        self.posted = None

    def __call__(self, method, path, **kw):
        if method == "POST":
            self.posted = kw["json"]
            return {"id": "run-0", "status": "ENGINE_SCHEDULING", "created_at": T0.isoformat().replace("+00:00", "Z")}
        job = path.split("/")[2]
        return {"id": f"{job}-run", "status": self.outcome.get(job, "ENGINE_SUCCEEDED"), "created_at": T0.isoformat()}

    def job_ids(self):
        return {n: n for n in CHAIN}

    def latest_run(self, job_id):
        if self.outcome.get(job_id) == "NEVER":
            return None
        return {"id": f"{job_id}-run", "status": self.outcome.get(job_id, "ENGINE_SUCCEEDED"),
                "created_at": (T0 + timedelta(minutes=1)).isoformat()}


@pytest.fixture
def run(monkeypatch, capsys):
    monkeypatch.setattr(trig, "POLL_S", 0)
    monkeypatch.setenv("CAI_URL", "https://ml.example")
    monkeypatch.setenv("CAI_API_KEY", "k")
    monkeypatch.setenv("CAI_PROJECT_ID", "p")
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef")

    def _run(outcome):
        fake = FakeApi(outcome)
        monkeypatch.setattr(trig, "Api", lambda *a, **k: fake)
        rc = trig.main()
        return rc, capsys.readouterr().out, fake
    return _run


def test_status_normalisation():
    assert trig.status_of({"status": "ENGINE_SUCCEEDED"}) == "succeeded"
    assert trig.status_of({"status": "engine_timedout"}) == "timedout"
    assert trig.status_of({}) == ""


def test_whole_chain_succeeds_and_passes_the_commit(run):
    rc, out, fake = run({})
    assert rc == 0 and "new champion is serving" in out
    assert fake.posted == {"environment": {"EXPECTED_GIT_SHA": "0123456789ab"}}


def test_gate_failure_is_named(run):
    rc, out, _ = run({GATE_JOB: "ENGINE_FAILED"})
    assert rc == 1 and "KPI GATE REJECTED the candidate" in out


def test_other_failures_name_the_job(run):
    rc, out, _ = run({"cxr-01-build-features": "ENGINE_TIMEDOUT"})
    assert rc == 1 and "cxr-01-build-features timedout" in out


def test_runs_older_than_the_trigger_are_ignored(run, monkeypatch):
    monkeypatch.setattr(trig, "TIMEOUTS", {n: 0.05 for n in CHAIN})
    rc, out, _ = run({"cxr-02-train-validate": "NEVER"})
    assert rc == 1 and "no result" in out


def test_skips_cleanly_without_secrets(monkeypatch, capsys):
    monkeypatch.delenv("CAI_URL", raising=False)
    assert trig.main() == 0
    assert "skipping the CAI pipeline" in capsys.readouterr().out
