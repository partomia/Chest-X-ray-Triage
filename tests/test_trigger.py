"""ci/trigger_cai_pipeline.py against a fake CAI API v2 (no network)."""
from datetime import datetime, timedelta, timezone

import pytest

import ci.trigger_cai_pipeline as trig
from ci.cai_jobs import CHAINS, GATE_JOB

T0 = datetime(2026, 9, 29, 8, 0, tzinfo=timezone.utc)
ALL = [n for c in CHAINS.values() for n in c]


class FakeApi:
    """Job runs appear in chain order; `outcome` maps job name -> final CAI status."""

    def __init__(self, outcome: dict):
        self.outcome = outcome
        self.posted = []

    def __call__(self, method, path, **kw):
        job = path.split("/")[2]
        if method == "POST":
            self.posted.append((job, kw["json"]))
            return {"id": "run-0", "status": "ENGINE_SCHEDULING", "created_at": T0.isoformat().replace("+00:00", "Z")}
        return {"id": f"{job}-run", "status": self.outcome.get(job, "ENGINE_SUCCEEDED"), "created_at": T0.isoformat()}

    def job_ids(self):
        return {n: n for n in ALL}

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


def test_every_chain_runs_in_turn_and_sync_gets_the_commit(run):
    rc, out, fake = run({})
    assert rc == 0 and "pipeline succeeded" in out
    assert fake.posted == [("cxr-00-sync-code", {"environment": {"EXPECTED_GIT_SHA": "0123456789ab"}}),
                           ("qc-01-build-features", {"environment": {}}),
                           ("ptx-01-build-features", {"environment": {}})]


def test_gate_failure_is_named_and_the_other_chains_still_run(run):
    rc, out, fake = run({GATE_JOB: "ENGINE_FAILED"})
    assert rc == 1 and "pneumonia: KPI GATE REJECTED the candidate" in out
    assert [j for j, _ in fake.posted] == [c[0] for c in CHAINS.values()]
    assert "chains not completed: ['pneumonia']" in out


def test_a_failed_code_sync_stops_everything(run):
    rc, out, fake = run({"cxr-00-sync-code": "ENGINE_FAILED"})
    assert rc == 1 and [j for j, _ in fake.posted] == ["cxr-00-sync-code"]


def test_other_failures_name_the_job(run):
    rc, out, _ = run({"ptx-01-build-features": "ENGINE_TIMEDOUT"})
    assert rc == 1 and "ptx-01-build-features timedout" in out


def test_runs_older_than_the_trigger_are_ignored(run, monkeypatch):
    monkeypatch.setattr(trig, "TIMEOUTS", {n: 0.05 for n in ALL})
    rc, out, _ = run({"cxr-02-train-validate": "NEVER"})
    assert rc == 1 and "no result" in out


def test_skips_cleanly_without_secrets(monkeypatch, capsys):
    monkeypatch.delenv("CAI_URL", raising=False)
    assert trig.main() == 0
    assert "skipping the CAI pipeline" in capsys.readouterr().out


def test_unreachable_workbench_names_the_self_hosted_runner(monkeypatch, capsys):
    def refuse(*a, **k):
        raise trig.requests.ConnectTimeout("connect timeout=60")

    monkeypatch.setenv("CAI_URL", "https://private-bench.example")
    monkeypatch.setenv("CAI_API_KEY", "k")
    monkeypatch.setenv("CAI_PROJECT_ID", "p")
    monkeypatch.setattr(trig.requests, "request", refuse)
    assert trig.main() == 1
    assert "self-hosted runner" in capsys.readouterr().out
