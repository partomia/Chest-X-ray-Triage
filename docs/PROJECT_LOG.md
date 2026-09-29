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

Phases 0-7 done locally. Phase 8 (live CAI project) not started: needs the CAI
project, Kaggle credentials in the session and the GitHub secrets.
