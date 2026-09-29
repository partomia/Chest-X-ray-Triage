# Project Log

Purpose: recover full context in a new session without replaying any prior
conversation. Chronological, most-recent-day-last. Times are IST (UTC+5:30)
unless marked UTC. Cross-refs: `PLAN.md` (names, decisions, phases),
`README.md` (architecture, how to run), `docs/DEMO_RUNBOOK.md` (setup and demo).

## 2026-09-29: Day 1

### Guide review

Reviewed build guide v1.0 and its zip against the four sibling repos
(Mule, Churn, Collections, ALM CASA) and their logs of live CAI runs. Would
have broken on the live workbench:

- `gate/kpi_gate.py` ended with `sys.exit(main())`: a PASS raises
  `SystemExit(0)`, which the CAI job kernel reports as a failed run, so the
  deploy job would never start (Mule, commit `c064cce`).
- Every job script used `Path(__file__)` at module level; the job kernel runs
  scripts without `__file__` (Mule / Churn `_repo_root()`), and passes `-f`,
  which `argparse.parse_args()` rejects (Collections `9bf580a`).
- `ci/trigger_cai_pipeline.py` compared statuses case-sensitively against
  `{"ENGINE_SUCCEEDED", "SUCCEEDED"}`; the sibling DAGs normalise
  (`lower().replace("engine_", "")`) and also treat `timedout` as terminal.
- `app/launch_app.py` used `os.system`; kept the siblings' launcher (child
  process, CORS / XSRF off for uploads behind the CAI proxy).

Also changed: job creation scripted from `ci/cai_jobs.py` (timeouts in seconds,
`kernel` not set with a runtime, per the API v2 docs); model deployment request
fields checked against Cloudera's API v2 notebook (`cpu`, `memory`,
`nvidia_gpus`, `environment`; no replicas field).

### Build (local)

- venv Python 3.11: torch 2.14, transformers 5.17, scikit-learn 1.9.1,
  pandas 3.0.6, streamlit 1.64.
- Synthetic films at scale 0.5 (633 labelled, Kermany names, 1-3 films per
  patient), stub embedder: build + train + gate pass; the gate fails on cue with
  `config/ci-gate-fail.yaml`. At scale 0.25 the gate failed on sensitivity (0.79):
  ~20 VAL positives make the 95%-sensitivity threshold too high, the same
  small-VAL effect the guide describes for the real data.
- Real backbone `google/vit-base-patch16-224` at revision `3f49326` on the
  synthetic films: first run failed with "AutoImageProcessor requires the
  Torchvision library" (transformers 5.x); `torchvision` added to
  `requirements.txt`. Then: 768-dim embeddings, 633 films in 60 s on the laptop
  CPU; the `--limit 6` model failed the gate (limit check), the next full build
  replaced the smoke table; full-table gate PASS; endpoint start-up 8.4 s,
  0.5 s per film.
- Nightly job on 5 films reported PSI ALERT: noise at that size. Added
  `monitoring.min_films_for_drift: 30` (`TOO_FEW`).
- `pytest -q`: 28 passed (~20 s, includes the Streamlit app via AppTest).

### Status

Phases 0-7 done locally. Phase 8 (live CAI project) in progress.

## 2026-09-29: Phase 8, live CAI project

- CAI project `rsingh-chest-x-ray-triage` created from Git by Ravi; runtime
  JupyterLab / Python 3.11 / Standard 2026.08.1-b5 (`ML_RUNTIME_*` set, so job
  creation can detect it). `HF_HOME` and `HF_TOKEN` as project variables.
- `pip3 install -r requirements.txt`: torch 2.14.0+cpu, transformers 5.17.0,
  mlflow 3.16.1. Training then warned `mlflow logging failed ... unexpected
  keyword argument 'run_uuid'`: the newer mlflow breaks `mlflow-cml-plugin`
  (the sibling projects keep mlflow out of requirements for this reason).
  mlflow removed from `requirements.txt` (contract test added); after
  `pip3 uninstall mlflow mlflow-skinny mlflow-tracing` the runtime's mlflow
  2.19.0 (`/opt/cmladdons`) logs runs.
- Kaggle now issues `KGAT_` API tokens, no `kaggle.json`: Kaggle CLI 2.2.4 with
  `KAGGLE_API_TOKEN`. The zip is 2.29 GB (nested copy); 1.2 GB after clean-up;
  5,216 / 16 / 624 films. Kaggle lists the licence as "other".
- Smoke test `--limit 20`: 120 rows, data checks pass, gate FAILS on the limit
  check as designed.
- Full build: 5,856 films in ~10 min (4 vCPU), train 4,453 / val 779 / test 624,
  data checks pass. Baseline: threshold 0.916; VAL AUROC 0.992, sens 0.950,
  spec 0.963, Brier 0.026; TEST AUROC 0.954, sens 0.992, spec 0.744, Brier
  0.147 (FN 3, FP 60). Gate with placeholders failed on Brier (0.12) only.
- Gate set from the baseline: AUROC 0.94, sensitivity 0.95, specificity 0.70,
  Brier 0.16. The CI overlay keeps its own sensitivity floor (0.90): the stub on
  synthetic films reaches ~0.93 and says nothing about model quality.
- The operating threshold (0.916) landed above `p1_probability` (0.85): films
  at 0.85-0.916 would have been P1 while counted negative. `priority_band` now
  starts P1 at max(p1_probability, threshold); training prints the TEST bands
  and probability quantiles to set `p1_probability` from.
- Gate MLflow tag failed (`Missing the required parameter experiment_id`): the
  plugin needs `set_experiment` before reopening a run; fixed.
