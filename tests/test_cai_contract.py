"""What the CAI job runtime and the GitHub workflow need from this repo, checked statically.

Lessons carried over from the sibling projects' live runs:
  - the Jupyter-kernel job runtime runs a script without __file__ and with an extra -f argument
  - it reports ANY SystemExit, even sys.exit(0), as a failed run, which would stop the chain
"""
import ast
import re
from pathlib import Path

import yaml

from ci.cai_jobs import CHAIN, GATE_JOB, JOBS

REPO = Path(__file__).resolve().parents[1]
JOB_SCRIPTS = [j["script"] for j in JOBS] + ["ci/create_cai_jobs.py"]


def test_job_scripts_exist_and_names_are_unique():
    assert len({j["name"] for j in JOBS}) == len(JOBS)
    for s in JOB_SCRIPTS:
        assert (REPO / s).is_file(), s


def test_chain_is_linked_by_parents_in_order():
    by_name = {j["name"]: j for j in JOBS}
    assert CHAIN[0] == "cxr-00-sync-code" and by_name[CHAIN[0]]["parent"] is None
    for parent, child in zip(CHAIN, CHAIN[1:]):
        assert by_name[child]["parent"] == parent
        assert by_name[child]["schedule"] is None   # CAI: parent_job_id and schedule are exclusive
    assert GATE_JOB in CHAIN and CHAIN.index(GATE_JOB) == len(CHAIN) - 2   # deploy is the only job after it


def test_nightly_job_is_scheduled_and_outside_the_chain():
    nightly = [j for j in JOBS if j["name"] not in CHAIN]
    assert len(nightly) == 1 and nightly[0]["schedule"] and nightly[0]["parent"] is None


def _main_block(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            return node
    return None


def test_job_scripts_survive_the_cai_kernel_runtime():
    for s in JOB_SCRIPTS:
        src = (REPO / s).read_text()
        tree = ast.parse(src)
        # __file__ only inside a try/except NameError (the _repo_root helper)
        assert "except NameError" in src, f"{s}: no fallback for a missing __file__"
        top_level_file_use = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))
                              and "__file__" in ast.unparse(n)]
        assert not top_level_file_use, f"{s}: uses __file__ at module level"
        # unknown arguments tolerated
        assert "parse_args(ap)" in src or "parse_args(argparse.ArgumentParser())" in src or "argparse" not in src, s
        assert ".parse_args()" not in src, f"{s}: argparse would fail on the kernel's -f argument"
        # no unconditional SystemExit on success
        main = _main_block(tree)
        assert main is not None, s
        body = ast.unparse(main)
        assert "sys.exit(main())" not in body, f"{s}: sys.exit(0) reads as a failed CAI job run"
        assert "finish(main())" in body or re.search(r"if rc:\s*.*sys\.exit\(rc\)", body, re.S), s


def test_workflow_watches_every_chain_script_and_config():
    wf = yaml.safe_load((REPO / ".github/workflows/cai-mlops.yml").read_text())
    on = wf.get("on") or wf.get(True)   # PyYAML reads the bare key `on` as True
    paths = on["push"]["paths"]
    for s in [j["script"] for j in JOBS if j["name"] in CHAIN] + ["config/pipeline.yaml", "common.py"]:
        top = s.split("/")[0]
        assert s in paths or f"{top}/**" in paths, f"{s} not in the workflow's path filter"


def test_resolve_runtime_pages_to_the_session_runtime(monkeypatch):
    from types import SimpleNamespace as NS

    from ci.cai_jobs import resolve_runtime

    for k, v in {"KERNEL": "Python 3.11", "EDITION": "Standard", "EDITOR": "JupyterLab",
                 "FULL_VERSION": "2026.08.1-b5"}.items():
        monkeypatch.setenv(f"ML_RUNTIME_{k}", v)
    pages = {None: NS(runtimes=[NS(full_version="2025.09.1-b5", image_identifier="img:2025.09")],
                      next_page_token="p2"),
             "p2": NS(runtimes=[NS(full_version="2026.08.1-b5", image_identifier="img:2026.08")],
                      next_page_token="")}

    class Client:
        def list_runtimes(self, search_filter, page_size, page_token=None):
            return pages[page_token]

    assert resolve_runtime(Client()) == "img:2026.08"
    assert resolve_runtime(Client(), "img:pinned") == "img:pinned"


def test_deploy_survives_the_gateway_timeout_on_create_deployment(monkeypatch):
    """Live on a workbench: create_model_deployment returned 500 after the gateway's
    30 s deadline, yet the deployment was created and reached Deployed."""
    import sys
    import types
    from types import SimpleNamespace as NS

    import serve.deploy_champion as dc

    class ApiException(Exception):
        def __init__(self, status):
            super().__init__(f"({status})\nReason: Internal Server Error")
            self.status = status

    class Client:
        def __init__(self):
            self.listed = 0

        def list_models(self, pid, search_filter):
            return NS(models=[NS(name="cxr-triage", id="m1")])

        def create_model_build(self, req, pid, mid):
            return NS(id="b1")

        def get_model_build(self, pid, mid, bid):
            return NS(status="built")

        def create_model_deployment(self, req, pid, mid, bid):
            raise ApiException(500)

        def list_model_deployments(self, pid, mid, bid):
            self.listed += 1
            return NS(model_deployments=[NS(id="d1")] if self.listed > 1 else [])

        def get_model_deployment(self, pid, mid, bid, did):
            assert did == "d1"
            return NS(status="deployed")

    client = Client()
    fake = types.ModuleType("cmlapi")
    fake.default_client = lambda: client
    fake.CreateModelBuildRequest = fake.CreateModelDeploymentRequest = fake.CreateModelRequest = dict
    monkeypatch.setitem(sys.modules, "cmlapi", fake)
    monkeypatch.setenv("CDSW_PROJECT_ID", "p1")
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    cfg = {"serving": {"model_name": "cxr-triage"},
           "cai": {"runtime_identifier": "img", "model_cpu": 2, "model_memory_gb": 4}}
    meta = {"feature_version": "1.0.0", "git_sha": "abcdef0", "metrics": {"test": {"auroc": 0.97}}}
    dc.deploy(cfg, meta)
    assert client.listed == 2

    client.create_model_deployment = lambda *a: (_ for _ in ()).throw(ApiException(400))
    try:
        dc.deploy(cfg, meta)
        raise AssertionError("a 4xx must not be treated as a gateway timeout")
    except ApiException as e:
        assert e.status == 400


def test_requirements_leave_the_runtime_mlflow_alone():
    reqs = [ln.split("#")[0].strip().lower() for ln in (REPO / "requirements.txt").read_text().splitlines()]
    assert not any(r.startswith("mlflow") for r in reqs), "mlflow-cml-plugin needs the runtime's own mlflow"


def test_gitignore_keeps_data_out_but_champion_in():
    ignored = [ln.strip() for ln in (REPO / ".gitignore").read_text().splitlines()
               if ln.strip() and not ln.startswith("#")]
    for d in ("data/", "feature_store/", "outputs/", "models/archive/", ".hf_cache/"):
        assert d in ignored
    assert "models/" not in ignored and "models/champion/" not in ignored   # the model build needs it
