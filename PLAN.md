# Build plan: Chest X-ray Triage on Cloudera AI Workbench

Score every incoming pediatric chest X-ray for pneumonia, re-order the reading
worklist (P1 read first, P2 likely abnormal, P3 routine), explain with an
occlusion heatmap and capture the radiologist's read for the next retrain.
Decision support only, never diagnosis.

Source: build guide v1.0 (`CXR_Triage_CAI_Build_Guide.docx`, 29 Sep 2026) and its
companion zip. Where the guide and the sibling projects' live CAI learnings
differ, the live learnings win (decision 1).

## Platform mapping

| Layer | Service | What runs there |
|---|---|---|
| Feature engineering | CAI Job 01 | frozen ViT embeddings + 9 quality stats -> versioned Parquet feature table, data checks |
| Training + evaluation | CAI Job 02 + Experiments | logistic head, threshold on VAL, KPIs on TEST, MLflow run |
| KPI gate | CAI Job 03 | absolute KPIs, non-regression, feature hash, full table |
| Deploy | CAI Job 04 + API v2 | promote, model build + deployment, rollback on failure |
| Online scoring | CAI Model Deployment | `serve/predict.py::predict` |
| Worklist UI | CAI Application | Streamlit worklist, heatmap, feedback |
| Monitoring | CAI Job 05 (cron) | nightly worklist, PSI drift on quality features |
| CI | GitHub Actions | `ci.yml` offline chain on every push / PR; `cai-mlops.yml` drives the CAI chain |

Names: CAI project `chest-x-ray-triage`, jobs `cxr-00-sync-code` .. `cxr-05-nightly-worklist`
(`ci/cai_jobs.py`), model `cxr-triage`, application `CXR Triage Worklist`
(subdomain `cxr-triage`), MLflow experiment `cxr-triage`, env var prefix `CXR_`.

## Decisions

1. CAI-runtime rules from the siblings are enforced by a test, not a checklist:
   no `__file__` at module level, unknown arguments ignored, no `SystemExit` on success.
2. One job definition file (`ci/cai_jobs.py`) feeds both job creation and the
   GitHub trigger, so the names cannot drift apart.
3. CI runs the real chain on synthetic films with a stub embedder whose feature
   hash differs from any real backbone; CI proves the gate fails on cue.
4. Backbone revision pinned (`3f49326`); changing it means a new feature version.
5. A `--limit` smoke table is rebuilt automatically by the next full build and a
   model trained on one never passes the gate.
6. Patient leakage between train and val is a critical data check (the build
   stops); duplicate films across splits and unreadable files are warnings.
7. Deploy failure restores the previous champion, so `models/champion/` always
   matches what the endpoint serves and the non-regression rule stays meaningful.
8. Drift below 30 incoming films is reported as `TOO_FEW`, not as a PSI level.
9. The app scores in-process (heatmap needs ~65 scores per film); the endpoint
   is for integrations and the Test tab.

## Phases

- [x] **0. Scaffold** - repo from the guide's zip, config + overlays, requirements split, venv.
- [x] **1. Feature engineering** - feature logic, stub embedder, data checks, limit guard.
- [x] **2. Training + evaluation** - MLflow failures only warn; stale gate result removed with a new candidate.
- [x] **3. KPI gate** - limit check; exit only on failure.
- [x] **4. GitHub -> CAI API** - job definitions, `create_cai_jobs.py`, trigger (status normalisation, run-id follow, step summary, skip without secrets), `sync_code` as a function.
- [x] **5. Serving, app, monitoring** - `@cml_model`, endpoint test script, launcher as a child process, FIFO view, rollback, drift minimum.
- [x] **6. Tests + CI** - 28 tests (logic, CAI contract, trigger against a fake API, offline end-to-end incl. the app); both workflows.
- [x] **7. Docs** - README, this plan, `docs/PROJECT_LOG.md`, `docs/DEMO_RUNBOOK.md`.
- [ ] **8. Live on CAI** - project, data, baseline, gate thresholds from the baseline, jobs, first chain run, model, app, GitHub secrets, first push-triggered run, failure-path rehearsal, screenshots.

## Next steps (Phase 8)

1. Push to `origin/main` (the offline `ci.yml` runs; `cai-mlops.yml` skips until secrets exist).
2. Create the CAI project, load the Kaggle data, run the baseline (runbook 1.2-1.4).
3. Set the gate from the measured baseline TEST metrics; commit.
4. `ci/create_cai_jobs.py`, first chain run from the UI; confirm the VERIFY items.
5. App, GitHub secrets, a push-triggered run and the rejected-gate rehearsal.
6. Log timings and IDs in `docs/PROJECT_LOG.md`; screenshots into `docs/images/`.
