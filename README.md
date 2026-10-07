# Chest X-ray Triage on Cloudera AI Workbench

Decision-support demo: scores every incoming chest X-ray for pneumonia and moves
likely-abnormal films to the top of the radiologist's worklist, with a heatmap of
the region that drove the score and a one-click override that feeds the next
retrain. **Workflow prioritisation, not diagnosis**: the radiologist still reads
every film; any clinical use needs regulatory clearance and local validation.

The model lives in one Cloudera AI (CAI) Workbench project: Sessions, Jobs,
Experiments (MLflow), Models, Applications and API v2. GitHub holds only code and
configuration; it triggers and follows the pipeline through the CAI API, so patient
images never leave the workbench.

Around it, a **lakehouse** on the same CDP environment ("federal") runs the hospital
day: CDE lands the RIS, PACS, report and EMR extracts and builds bronze / silver /
gold Iceberg tables with reconciliation, a CDE Airflow DAG has CAI score each day's
studies with the deployed champion through CDW Impala, CDE replays the reading list
in triage order against first-in-first-out, and two Data Visualization dashboards
show the worklist, the waits, the model and the data quality.

```
Feature Engineering -> Model Training -> Model Evaluation -> KPI Gate      (what the code does)
GitHub Push -> Train & Validate -> KPI Gate -> Cloudera AI API (deploy)    (who runs it, and when)

land -> bronze -> silver -> gold (CDE) -> score studies (CAI, Impala) -> outcomes (CDE) -> views (CDW) -> dashboards (CDV)
```

See [Lakehouse](#lakehouse-cde-cdw-data-visualization) below.

## Architecture

| Path | What it is | CAI object |
|---|---|---|
| `config/pipeline.yaml` | every setting, versioned in Git | - |
| `features/feature_logic.py` | the ONLY feature definition (train = batch = online) | - |
| `features/build_feature_table.py` | versioned feature table `feature_store/cxr_features/v<ver>/` + data checks | Job `cxr-01-build-features` |
| `train/train_validate.py` | trains the head, picks the threshold on VAL, KPIs on TEST, MLflow | Job `cxr-02-train-validate` |
| `evaluate/metrics.py` | threshold selection, KPIs, priority bands | - |
| `gate/kpi_gate.py` | absolute KPIs + non-regression vs champion; exit 1 = stop | Job `cxr-03-kpi-gate` |
| `serve/deploy_champion.py` | promote + build + deploy via API v2, rollback on failure | Job `cxr-04-deploy-champion` |
| `serve/predict.py` | model endpoint function `predict` | Model `cxr-triage` |
| `serve/explain.py` | occlusion heatmap | - |
| `monitor/batch_score.py` | nightly worklist + PSI drift | Job `cxr-05-nightly-worklist` |
| `app/launch_app.py`, `app/app.py` | Streamlit worklist, heatmap, radiologist feedback | Application |
| `ci/sync_code.py` | pulls the pushed commit into the project | Job `cxr-00-sync-code` |
| `ci/cai_jobs.py` | the six job definitions (names, scripts, parents, profiles, timeouts) | - |
| `ci/create_cai_jobs.py` | creates the six jobs from a session, prints the GitHub secrets | - |
| `ci/trigger_cai_pipeline.py` | GitHub runner -> CAI API v2: start job 00, follow the chain | GitHub Actions |
| `.github/workflows/ci.yml` | every push / PR: tests + the whole chain offline on synthetic films | GitHub Actions |
| `.github/workflows/cai-mlops.yml` | push to `main`: unit tests, then the CAI chain | GitHub Actions |
| `scripts/make_synthetic_cxr.py` | synthetic films in the Kermany layout (CI only) | - |
| `scripts/fetch_dataset.py` | the Kermany films from the pinned Hugging Face mirror, original names | Job `cxr-setup-data` |
| `ci/setup_cai.py` | project, environment, jobs, app from a laptop over API v2 (private repo: deploy key) | - |
| `lakehouse/score_studies.py` | scores one business date of `gold.fact_study`: worklist band, shadow band, every head | Job `cxr-06-score-studies` |
| `serve/promote_champion.py` | silent trial -> champion on lakehouse evidence and a named approver | Job `cxr-07-promote-champion` |
| `serve/registry.py` | a Registry version per silent trial / champion / promotion | - |
| `features/degrade.py` | films unfit for AI triage, made from real ones | - |
| `scripts/select_nih_subset.py`, `scripts/fetch_nih.py` | chooses (laptop) and fetches (CAI) the NIH films | Job `cxr-setup-nih` |
| `lakehouse/publish.py`, `lakehouse/store.py` | lineage into `ref.*`; one store over Impala and Spark | - |

Data (git-ignored, lives only in the CAI project):

```
data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg   # Kermany et al. 2018, Kaggle
data/incoming/                     # unlabelled 'new' films for the worklist
feature_store/cxr_features/v1.0.0/ # features.parquet + manifest.json (job 01)
models/champion/                   # promoted by job 04 (small, not ignored: the model build needs it)
models/archive/<ts>/               # every previous champion, for rollback
outputs/                           # candidate, gate result, worklists, drift, feedback
```

## How the pieces keep each other honest

- **One feature definition, one hash.** `feature_hash()` fingerprints the backbone,
  its pinned revision, image size, pooling and the logic version. Job 01 skips a
  version that already exists with the same hash and fails if the hash changed
  (bump `features.version`); job 02 refuses a stale table; the gate checks the
  hash; `serve/predict.py` refuses to start on a mismatch.
- **Threshold on VAL, KPIs on TEST.** The operating threshold is the highest one
  that keeps VAL sensitivity at 0.95; TEST is never used to tune anything.
- **The gate decides, the chain obeys.** Dependent CAI jobs start only when the
  parent succeeds, so `kpi_gate.py` exiting 1 stops the deploy and the GitHub
  check goes red naming the gate. The current champion keeps serving.
- **Deploy is reversible.** Job 04 archives the champion before promoting; if the
  model build or rollout fails it puts the previous champion back.

## Three models, one worklist

A reading room gets adults as well as children, and films a radiographer would send
back. Three heads share the one ViT embedding (one feature hash), each with its own
data, KPI gate, CAI job chain (`CXR_MODEL` on the job) and AI Registry entry:

| Model | Finding | Intended use | Data | After a passed gate |
|---|---|---|---|---|
| `pneumonia` | PNEUMONIA | children 0-17 | Kermany (5,856 films) | **champion**: ranks the worklist |
| `film_qc` | film unfit for AI triage | every film | real films of both datasets, each with one degraded copy (`features/degrade.py`: blur, under/over-exposure, noise, crop, rotation, inversion) | **champion**: an unfit film gets band NA |
| `pneumothorax` | PNEUMOTHORAX | adults 18+ | 6,500 NIH ChestX-ray14 films (`data_refs/nih_cxr14_subset.csv`, patient-level splits) | **silent trial**: scored on every live adult film, never shown |

- **Band NA, no AI triage.** A film outside every live champion's intended use, or one
  the film check rejects, is read in arrival order with the P3 films, as without AI.
- **Silent trial.** The lakehouse scorer runs the trial head on every live study and
  writes `gold.model_score`; CDE builds `gold.daily_model_summary` (per date, model,
  stage and version: confusion counts at the head's own threshold against the signed
  report) and a *shadow* reading arm: the worklist as if the trial model were live.
  Dashboard *CXR Models & Silent Trial* shows adult pneumothorax waits in FIFO order,
  on today's worklist (worse: flagged children jump ahead of adults) and if live.
- **Promotion needs evidence and a person.** Job `cxr-07-promote-champion` (manual,
  `CXR_MODEL`, `CXR_APPROVED_BY`) sums the trial's evidence for that model version,
  checks `models.<name>.go_live` (days, reported positives, sensitivity, specificity
  on live films), and only then installs it as champion, rebuilds the endpoint and
  records `PROMOTED`; otherwise `PROMOTION_REFUSED` and nothing changes.
- **AI Registry.** Every silent trial, champion and promotion is a new version of
  `cxr-pneumonia` / `cxr-film-qc` / `cxr-pneumothorax` (`serve/registry.py`, cmlapi
  `create_registered_model` on the MLflow run), tagged at creation with stage, finding,
  population, git commit, feature hash, threshold, TEST KPIs and, on promotion, the
  approver and the trial evidence.
- **Honest limits.** Frozen ImageNet features at 224 px see a large pneumothorax, not a
  thin apical rim: a laptop trial gave AUROC ~0.72 (sensitivity 0.85 at specificity
  0.42); the gate asks "worth a silent trial", the trial asks "fit to rank adults". The
  film check learns synthetic failures only; a real one would learn from reject analysis.

## Run it locally (no CAI, no dataset, no GPU)

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-ci.txt
.venv/bin/pytest -q                                    # 62 tests, ~35 s, includes the app and the lakehouse logic

export CXR_CONFIG_OVERLAY=config/ci.yaml               # synthetic films + stub embedder
.venv/bin/python scripts/make_synthetic_cxr.py --scale 0.5
.venv/bin/python features/build_feature_table.py
.venv/bin/python train/train_validate.py
.venv/bin/python gate/kpi_gate.py
```

The stub embedder (`backbone: stub:...`) has its own feature hash, so nothing it
builds can be served against real features. With `pip install -r requirements.txt`
and without the overlay, the same scripts use the real ViT backbone.

## Set up on Cloudera AI

The copy-paste setup and the seven-minute demo script are in
[`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md). In short:

1. Create the CAI project from this repo (Python 3.11 runtime, JupyterLab),
   `pip3 install -r requirements.txt`, set `HF_HOME=/home/cdsw/.hf_cache`.
2. Download the Kaggle dataset into `data/raw/`, copy ~40 test films to `data/incoming/`.
3. Baseline run in a session (build features, train, gate), then set the gate
   thresholds a little below the baseline in `config/pipeline.yaml`.
4. `python ci/create_cai_jobs.py` creates jobs 00-05; start `cxr-00-sync-code` once.
5. Application `app/launch_app.py`; GitHub secrets `CAI_URL`, `CAI_API_KEY`, `CAI_PROJECT_ID`.

On the federal environment everything above is scripted from a laptop with the
gitignored `.env` (`CXR_CAI_HOST`, `CXR_CAI_API_KEY`, `CXR_IMPALA_USER`,
`CXR_IMPALA_PASSWORD`, `CXR_VIZ_API_KEY`):

```bash
set -a; source .env; set +a
python ci/setup_cai.py --deploy-key ~/.ssh/cai_federal_cxr_deploy   # first time: project + git bootstrap
python ci/setup_cai.py --run cxr-setup-data                         # films + requirements, ~16 min
python ci/setup_cai.py --run cxr-setup-nih                          # 6,500 NIH films (~13 GB of row groups read)
python ci/setup_cai.py --run cxr-00-sync-code                       # starts the chain 01 -> 04
python ci/setup_cai.py --run qc-01-build-features                   # film check chain, then:
python ci/setup_cai.py --run ptx-01-build-features                  # pneumothorax chain -> silent trial
python ci/setup_cai.py --app                                        # the worklist application
# after enough trial days in the lakehouse:
python ci/setup_cai.py --run cxr-07-promote-champion --env CXR_MODEL=pneumothorax,CXR_APPROVED_BY="Dr A Rao"
```

## Lakehouse (CDE, CDW, Data Visualization)

The hospital is simulated, deterministically per business date (`config/lakehouse.json`):
48 chest films a day from the 624 Kermany **test** films (never used for training or the
threshold), arriving 07:00-21:00 from ED / OPD / ward / ICU, read by one radiologist
first in, first out at 20 minutes a film, so a queue builds and some reports are signed
after midnight (late-arriving truth). 2026-10-01 carries planted faults: two re-sent
PACS headers, a header without an accession number, an order with an impossible time.

| Stage | Code | Platform | Writes (database `rsingh_cxr_<layer>`) |
|---|---|---|---|
| land | `cde/jobs/land_sources.py` | CDE Spark | `s3a://.../rsingh_cxr/landing/<source>/<date>/` + `_manifest.json` |
| bronze | `cde/jobs/ingest_bronze.py` | CDE Spark | `bronze.ris_order`, `pacs_study`, `radiology_report`, `emr_patient`, `quarantine` |
| silver | `cde/jobs/build_silver.py` | CDE Spark | `silver.imaging_order`, `study`, `report`, `patient` (typed, de-duplicated) |
| gold | `cde/jobs/build_gold.py` | CDE Spark | `gold.dim_patient` (de-identified), `fact_study`, `fact_report` |
| score | `lakehouse/score_studies.py` | CAI job via Impala | `gold.triage_score`, `ref.triage_run` (snapshot read, bands, PSI drift) |
| outcomes | `cde/jobs/build_outcomes.py` | CDE Spark | `gold.fact_triage_outcome`, `gold.daily_triage_summary` |
| views | `sql/semantic/*.sql`, `scripts/run_semantic.py` | CDW Impala | `semantic.v_daily_kpi`, `v_triage_outcome`, ... (10 views) |
| dashboards | `dataviz/build_dashboard.py` | Data Visualization | *CXR Triage Operations*, *CXR Model, Drift & Data Quality* |

Every CDE stage reconciles its counts into `ref.recon_results` (MATCHED / EXPLAINED /
MISMATCH) and `ref.load_audit`, and fails on a mismatch, which stops the DAG. The model
pipeline publishes `ref.training_set` (job 01) and `ref.model_event` (gate and deploy),
best-effort. The outcomes compare the two orderings on the same reading model, so the
difference in waits is the triage alone; on the stub scorer pneumonia films wait a
median ~3 h under FIFO and ~40 min in triage order, and normal films wait longer.

```bash
python scripts/run_local.py all                       # every demo date on local Spark + Iceberg (stub scorer)
./cde/scripts/deploy_jobs.sh                          # CDE files resource from `git archive` + 5 Spark jobs
./cde/scripts/deploy_dag.sh                           # DAG rsingh-cxr-orchestration (paused)
python cde/scripts/set_airflow_variables.py           # CXR_CAI_HOST / _PROJECT_ID / _API_KEY / _SCORE_JOB_ID
cde job run --name rsingh-cxr-orchestration --config-json '{"business_date": "2026-10-06"}'
python scripts/run_semantic.py --engine impala        # views + KPI consistency check
python dataviz/build_dashboard.py --import --connection federal-impala-1
```

The repository is private, so CDE gets the code as a files resource uploaded from a pushed
commit (`cde/DEPLOYED_SHA`) rather than a git repository holding a GitHub token.

## CAI runtime lessons built in

Carried over from the live runs of the sibling projects (Mule Account
Identifier, Customer Churn Prediction, Collections Delinquency, ALM IRRBB CASA)
and enforced by `tests/test_cai_contract.py`:

| Lesson | Where it is handled |
|---|---|
| The Jupyter-kernel job runtime runs scripts without `__file__` | `_repo_root()` fallback in every job script |
| ... and passes an extra `-f <file>` argument | `common.parse_args()` ignores unknown arguments |
| Any `SystemExit`, even `sys.exit(0)`, reads as a failed job run | `common.finish()` exits only on failure, so a passing gate lets deploy run |
| Job-run status comes back as `ENGINE_SUCCEEDED` etc.; `arguments` are ignored, `environment` is applied | `ci/trigger_cai_pipeline.py` normalises status, passes the commit as `EXPECTED_GIT_SHA` |
| `exec`-ing Streamlit kills the application's kernel | `app/launch_app.py` runs it as a child process |
| MLflow can break on runtime package pins | tracking failures only warn; the candidate is kept |
| Workflows go red before the platform secrets exist | `cai-mlops.yml` prints a notice and succeeds until `CAI_URL` is set |

## Changes from build guide v1.0

- `torchvision` added to `requirements.txt`: `transformers` 5.x image processors need it (found in the local ViT run).
- `ci/create_cai_jobs.py` creates the six jobs (section 9.2) instead of the UI; job timeouts are in seconds (API v2).
- `features.backbone_revision` pinned to `3f49326` (the Hugging Face commit of `google/vit-base-patch16-224` on 29 Sep 2026).
- Job 01 runs data checks (patient leakage, empty class, unreadable files, duplicate films across splits) and rebuilds a `--limit` smoke table automatically; the gate rejects a model trained on one.
- Job 04 rolls back to the previous champion if the build or deployment fails.
- A second workflow (`ci.yml`) runs the whole chain offline on every push and PR; the gate is also shown failing on cue.
- Nightly drift reports `TOO_FEW` below 30 incoming films instead of a PSI on noise.

## Still to check on the target workbench

Run once against the live workbench (see `docs/DEMO_RUNBOOK.md`, section VERIFY):
model build / deployment status strings (`built`, `deployed`) in `deploy_champion.py`,
the `ML_RUNTIME_*` runtime detection (or set `cai.runtime_identifier`), whether
`cdsw-build.sh` runs on your runtime, and the dataset licence text (CC BY 4.0).
