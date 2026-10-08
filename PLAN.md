# Build plan: Chest X-ray Triage on Cloudera AI Workbench

Score every incoming chest X-ray, re-order the reading worklist (P1 read first, P2
likely abnormal, P3 routine, NA no AI triage), explain with an occlusion heatmap and
capture the radiologist's read for the next retrain. Decision support only, never
diagnosis. Three models share one embedding: pneumonia (children), film_qc (every
film) and pneumothorax (adults, promoted from a silent trial).

Source: build guide v1.0 (`CXR_Triage_CAI_Build_Guide.docx`, 29 Sep 2026) and its
companion zip. Where the guide and the sibling projects' live CAI learnings
differ, the live learnings win (decision 1).

## Platform mapping

| Layer | Service | What runs there |
|---|---|---|
| Feature engineering | CAI Jobs `*-01` | frozen ViT embeddings + 9 quality stats -> versioned Parquet feature table per model, data checks |
| Training + evaluation | CAI Jobs `*-02` + Experiments | logistic head, threshold on VAL, KPIs on TEST, MLflow run |
| KPI gate | CAI Jobs `*-03` | absolute KPIs, non-regression, feature hash, full table |
| Deploy | CAI Jobs `*-04` + API v2 | champion (model build + deployment, rollback on failure) or silent trial; AI Registry version |
| Promotion | CAI Job `cxr-07` | silent trial -> champion on lakehouse evidence and a named approver |
| Online scoring | CAI Model Deployment | `serve/predict.py::predict`: every head, worklist band |
| Worklist UI | CAI Application | Streamlit worklist, heatmap, feedback |
| Monitoring | CAI Job 05 (cron) | nightly worklist, PSI drift on quality features |
| Lakehouse | CDE Spark + Airflow, CDW Impala, Data Visualization | hospital day: land -> bronze -> silver -> gold, CAI scoring (`cxr-06`), outcomes, 12 views, 3 dashboards |
| CI | GitHub Actions | `ci.yml` offline chain on every push / PR; `cai-mlops.yml` drives the three CAI chains |

Names: CAI project `rsingh-chest-x-ray-triage` on federal, jobs of `ci/cai_jobs.py`
(`cxr-*` pneumonia, `qc-*` film_qc, `ptx-*` pneumothorax), model `cxr-triage`,
registered models `cxr-pneumonia`, `cxr-film-qc`, `cxr-pneumothorax`, application
`CXR Triage Worklist` (subdomain `rsingh-cxr-triage`), MLflow experiment `cxr-triage`,
lakehouse databases `rsingh_cxr_*`, env var prefix `CXR_`.

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
10. More models are heads on the same features (one feature hash), each with its own
    data, gate, chain (`CXR_MODEL`) and registry entry; the worklist band comes from the
    live champions whose intended population the film is in, else NA.
11. A new model that would change who is read first starts as a silent trial; promotion
    needs go-live criteria met on live films and a named approver, never the gate alone.
12. The registry records every version; what the federal registry cannot store (tags)
    is kept in `ref.model_event` beside the version number.

## Phases

- [x] **0. Scaffold** - repo from the guide's zip, config + overlays, requirements split, venv.
- [x] **1. Feature engineering** - feature logic, stub embedder, data checks, limit guard.
- [x] **2. Training + evaluation** - MLflow failures only warn; stale gate result removed with a new candidate.
- [x] **3. KPI gate** - limit check; exit only on failure.
- [x] **4. GitHub -> CAI API** - job definitions, `create_cai_jobs.py`, trigger (status normalisation, run-id follow, step summary, skip without secrets), `sync_code` as a function.
- [x] **5. Serving, app, monitoring** - `@cml_model`, endpoint test script, launcher as a child process, FIFO view, rollback, drift minimum.
- [x] **6. Tests + CI** - logic, CAI contract, trigger against a fake API, offline end-to-end incl. the app; both workflows (62 tests now).
- [x] **7. Docs** - README, this plan, `docs/PROJECT_LOG.md`, `docs/DEMO_RUNBOOK.md`.
- [x] **8. Live on CAI** - project, data, baseline, gate from the baseline, jobs, chain, model, app, GitHub secrets, push-triggered runs (federal).
- [x] **9. Lakehouse** - CDE stages with reconciliation, Airflow DAG with CAI scoring, outcomes FIFO vs triage, semantic views, dashboards.
- [x] **10. Three models and governance** - film_qc and pneumothorax chains, band NA, silent trial with a shadow arm, go-live criteria, promotion with an approver, AI Registry versions; proven on federal 2026-10-08.

## Next steps

1. Name a real clinical approver and re-run the promotion record if the demo needs it.
2. Rotate the workload password (it is returned in job-run API responses).
3. A second, held-out pneumonia test set (the `C` choice looked at TEST).
4. Screenshots of the three-model app and the *CXR Models & Silent Trial* dashboard into `docs/images/`.
