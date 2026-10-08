# Chest X-ray Triage on Cloudera AI Workbench

Decision-support demo: scores every incoming chest X-ray and moves likely-abnormal
films to the top of the radiologist's worklist, with a heatmap of the region that
drove the score and a one-click override that feeds the next retrain. Three models
share one image embedding:

- **pneumonia** for children (Kermany films): ranks the pediatric worklist;
- **film_qc** for every film: a blurred, badly exposed, cropped, rotated or inverted
  film gets no AI priority (band NA) instead of a confident wrong one;
- **pneumothorax** for adults (NIH ChestX-ray14 films): entered as a **silent trial**
  (scored on live films, shown to nobody), earned its go-live on ten days of lakehouse
  evidence and was **promoted to champion with a named approver**.

Every silent trial, champion and promotion is a version in the **Cloudera AI
Registry**. **Workflow prioritisation, not diagnosis**: the radiologist still reads
every film; any clinical use needs regulatory clearance and local validation.

The models live in one Cloudera AI (CAI) Workbench project: Sessions, Jobs,
Experiments (MLflow), Models, Applications, AI Registry and API v2. GitHub holds only
code and configuration; it triggers and follows the pipeline through the CAI API, so
patient images never leave the workbench.

Around it, a **lakehouse** on the same CDP environment ("federal") runs the hospital
day: CDE lands the RIS, PACS, report and EMR extracts and builds bronze / silver /
gold Iceberg tables with reconciliation, a CDE Airflow DAG has CAI score each day's
studies with every model head through CDW Impala, CDE replays the reading list in
triage order against first-in-first-out (and, for a model in silent trial, as if it
were live), and three Data Visualization dashboards show the worklist, the waits, the
models, the silent trial and the data quality.

Status on federal (2026-10-08): all three models are champions (registry
`cxr-pneumonia` v2, `cxr-film-qc` v1, `cxr-pneumothorax` v3), ten business dates
(2026-09-28 .. 10-07, 479 films) are in the lakehouse, and the GitHub workflow runs
all three chains green. Details and every number: [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).

```
Feature Engineering -> Model Training -> Model Evaluation -> KPI Gate      (what the code does)
GitHub Push -> Train & Validate -> KPI Gate -> Cloudera AI API (deploy)    (who runs it, and when)
  per model:  cxr-* pneumonia -> champion   qc-* film_qc -> champion   ptx-* pneumothorax -> silent trial
  silent trial -> lakehouse evidence -> go-live criteria + approver (cxr-07) -> champion; Registry version at each step

land -> bronze -> silver -> gold (CDE) -> score studies (CAI, Impala) -> outcomes (CDE) -> views (CDW) -> dashboards (CDV)
```

See [Lakehouse](#lakehouse-cde-cdw-data-visualization) below.

## Architecture

| Path | What it is | CAI object |
|---|---|---|
| `config/pipeline.yaml` | every setting, versioned in Git; `models.<name>` overrides per model (`CXR_MODEL`) | - |
| `features/feature_logic.py` | the ONLY feature definition (train = batch = online) | - |
| `features/build_feature_table.py` | versioned feature table per model (`feature_store/{cxr,qc,nih}_features/v<ver>/`) + data checks | Jobs `cxr-01` / `qc-01` / `ptx-01-build-features` |
| `train/train_validate.py` | trains the head, picks the threshold on VAL, KPIs on TEST, MLflow | Jobs `cxr-02` / `qc-02` / `ptx-02-train-validate` |
| `evaluate/metrics.py` | threshold selection, KPIs, priority bands | - |
| `gate/kpi_gate.py` | absolute KPIs + non-regression vs champion (else the model in trial); exit 1 = stop | Jobs `cxr-03` / `qc-03` / `ptx-03-kpi-gate` |
| `serve/deploy_champion.py` | installs a passing candidate as champion (build + deploy via API v2, rollback on failure) or silent trial; registers the version | Jobs `cxr-04-deploy-champion`, `qc-04-deploy`, `ptx-04-deploy` |
| `serve/predict.py` | model endpoint function `predict`: one embedding, every head, worklist band | Model `cxr-triage` |
| `serve/explain.py` | occlusion heatmap | - |
| `monitor/batch_score.py` | nightly worklist + PSI drift | Job `cxr-05-nightly-worklist` |
| `app/launch_app.py`, `app/app.py` | Streamlit worklist, heatmap, radiologist feedback | Application |
| `ci/sync_code.py` | pulls the pushed commit into the project (its child jobs `cxr-01..04` follow) | Job `cxr-00-sync-code` |
| `ci/cai_jobs.py` | the 18 job definitions (names, scripts, parents, profiles, timeouts, `CXR_MODEL`) | - |
| `ci/create_cai_jobs.py` | creates the jobs from a session, prints the GitHub secrets | - |
| `ci/trigger_cai_pipeline.py` | GitHub runner -> CAI API v2: runs the three chains in turn (pneumonia from job 00), follows each | GitHub Actions |
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
data/raw/nih_cxr14/{train,val,test,lakehouse}/*.jpg              # 6,500 NIH films (cxr-setup-nih)
data/incoming/                     # unlabelled 'new' films for the worklist
feature_store/{cxr,qc,nih}_features/v1.0.0/  # features.parquet + manifest.json (jobs 01)
models/champion/                   # pneumonia champion (small, not ignored: the model build needs it)
models/{film_qc,pneumothorax}/{champion,silent_trial}/   # the other heads, by stage
models/archive/[<model>/]<ts>/     # every previous champion / trial, for rollback
outputs/[<model>/]candidate/       # candidate + gate result; worklists, drift, feedback
```

`data_refs/nih_cxr14_subset.csv` (in Git) lists the NIH films and their patient-level
split, chosen by `scripts/select_nih_subset.py` from the pinned Hugging Face mirror
`timm/nih-chest-xray-14` (revision `c1bf579`); `cde/reference/adult_films.csv` lists the
300 held-out adult films the lakehouse hospital uses (60 pneumothorax, 151 normal, 89 other).

## How the pieces keep each other honest

- **One feature definition, one hash.** `feature_hash()` fingerprints the backbone,
  its pinned revision, image size, pooling and the logic version. Job 01 skips a
  version that already exists with the same hash and fails if the hash changed
  (bump `features.version`); job 02 refuses a stale table; the gate checks the
  hash; `serve/predict.py` refuses to start on a mismatch.
- **Threshold on VAL, KPIs on TEST.** The operating threshold is the highest one
  that keeps VAL sensitivity at the model's target (pneumonia 0.95, film_qc 0.97,
  pneumothorax 0.85). Two choices were made after seeing TEST and are recorded as such:
  pneumonia's `C` (runbook 1.4) and film_qc's 0.97 (its first gate failed at 0.925
  test sensitivity with a 0.95 target set on 100 VAL positives).
- **The gate decides, the chain obeys.** Dependent CAI jobs start only when the
  parent succeeds, so `kpi_gate.py` exiting 1 stops the deploy and the GitHub
  check goes red naming the gate. The current champion keeps serving.
- **Deploy is reversible.** Job 04 archives the champion before promoting; if the
  model build or rollout fails it puts the previous champion back (`ROLLED_BACK` in
  `ref.model_event`, as happened twice on federal when a build stalled in the platform).
- **New evidence before new power.** A model that ranks no one yet enters as a silent
  trial; going live takes lakehouse evidence on live films and a person's approval.

## Three models, one worklist

A reading room gets adults as well as children, and films a radiographer would send
back. Three heads share the one ViT embedding (one feature hash), each with its own
data, KPI gate, CAI job chain (`CXR_MODEL` on the job) and AI Registry entry:

| Model | Finding | Intended use | Data | After a passed gate | TEST on federal | Now |
|---|---|---|---|---|---|---|
| `pneumonia` | PNEUMONIA | children 0-17 | Kermany (5,856 films) | **champion**: ranks the worklist | AUROC 0.969, sens 0.990, spec 0.752 | champion, `cxr-pneumonia` v2 |
| `film_qc` | film unfit for AI triage | every film | real films of both datasets, each with one degraded copy (`features/degrade.py`: blur, under/over-exposure, noise, crop, rotation, inversion) | **champion**: an unfit film gets band NA | AUROC 0.9997, sens 0.948, spec 0.998 | champion, `cxr-film-qc` v1 |
| `pneumothorax` | PNEUMOTHORAX | adults 18+ | 6,500 NIH ChestX-ray14 films (`data_refs/nih_cxr14_subset.csv`, patient-level splits: train 4,000 / val 600 / test 1,600, a quarter positive) | **silent trial**: scored on every live adult film, never shown | AUROC 0.867, sens 0.938, spec 0.556 | promoted 2026-10-08, `cxr-pneumothorax` v3 |

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
  records `PROMOTED`; otherwise `PROMOTION_REFUSED` and nothing changes. A later
  candidate that passes its gate enters as a **challenger** in silent trial, while the
  promoted champion keeps serving; it goes live the same way.
- **AI Registry.** Every silent trial, champion and promotion is a new version of
  `cxr-pneumonia` / `cxr-film-qc` / `cxr-pneumothorax` (`serve/registry.py`, cmlapi
  `create_registered_model` on the MLflow run). A version carries the MLflow model and
  the run's parameters and metrics (finding, population, git commit, feature hash,
  threshold, VAL and TEST KPIs). The audit tags (stage, approver, trial evidence) are
  sent too, but the federal workbench stores no version tags, so the stage, approver
  and version number are recorded with the model event in `ref.model_event`
  (`registry_version`), which the dashboards show. A failed registration never fails
  the job; its reason goes into the event's `detail`.
- **What the trial showed (federal, 2026-09-28 .. 10-07).** 159 reported adult films,
  37 with pneumothorax: sensitivity 0.946 (35 / 37), specificity 0.590, against go-live
  minimums of 7 days, 20 positives, 0.80 and 0.35. Adult pneumothorax films waited a
  median 324-404 min on the pediatric-only worklist (behind flagged children), 174-248
  min first-in-first-out, and 30-115 min with the trial model live (shadow arm).
  film_qc caught both planted unfit films and rejected 2 of 477 real films.
- **Honest limits.** Frozen ImageNet features at 224 px see a large pneumothorax, not a
  thin apical rim: AUROC 0.867 on 4,000 training films (a laptop trial on 465 gave
  ~0.72). Specificity 0.59 means about four in ten normal adult films are also moved
  up; the gate asks "worth a silent trial", the trial asks "fit to rank adults", and
  the approver weighs the cost. The film check learns synthetic failures only; a real
  one would learn from reject analysis.

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
4. `python ci/create_cai_jobs.py` creates the jobs of `ci/cai_jobs.py`; start `cxr-00-sync-code` once.
5. Application `app/launch_app.py`; GitHub secrets `CAI_URL`, `CAI_API_KEY`, `CAI_PROJECT_ID`.

On the federal environment everything above is scripted from a laptop with the
gitignored `.env` (`CXR_CAI_HOST`, `CXR_CAI_API_KEY`, `CXR_IMPALA_USER`,
`CXR_IMPALA_PASSWORD`, `CXR_VIZ_API_KEY`):

```bash
set -a; source .env; set +a
python ci/setup_cai.py --deploy-key ~/.ssh/cai_federal_cxr_deploy   # first time: project + git bootstrap
python ci/setup_cai.py                                              # (re)creates the 18 jobs, sets sizes and CXR_MODEL
python ci/setup_cai.py --run cxr-setup-data                         # Kermany films + requirements, ~16 min
python ci/setup_cai.py --run cxr-setup-nih                          # 6,500 NIH films, ~4 min on federal
python ci/setup_cai.py --run cxr-00-sync-code                       # sync, then its children cxr-01 -> 04 (pneumonia)
python ci/setup_cai.py --run qc-01-build-features                   # film check chain -> champion
python ci/setup_cai.py --run ptx-01-build-features                  # pneumothorax chain -> silent trial
python ci/setup_cai.py --app                                        # the worklist application
# after enough trial days in the lakehouse (README, Lakehouse), with the trial evidence in hand:
python ci/setup_cai.py --run cxr-07-promote-champion \
  --env "CXR_MODEL=pneumothorax,CXR_APPROVED_BY=<clinical lead>,CXR_APPROVAL_NOTE=<evidence; no commas>"
```

`--env` splits on commas, so keep them out of the values. `gh workflow run cxr-triage-mlops`
runs all three chains from GitHub instead (about 20 minutes when nothing but the code
changed). The per-user quota fits the model endpoint and **one** 2 vCPU workload: stop the
application while jobs run (a job otherwise waits in scheduling until its timeout), and
start it again afterwards.

## Lakehouse (CDE, CDW, Data Visualization)

The hospital is simulated, deterministically per business date (`config/lakehouse.json`):
48 chest films a day: 32 children from the 624 Kermany **test** films and 16 adults from
the 300 held-out NIH lakehouse films (`cde/reference/adult_films.csv`), none used for
training or a threshold. Age comes from the EMR and decides which model's intended use a
film falls in. Films arrive 07:00-21:00 from ED / OPD / ward / ICU and are read by one
radiologist first in, first out at 20 minutes a film, so a queue builds and some reports
are signed after midnight (late-arriving truth). 2026-10-01 carries planted faults: two
re-sent PACS headers, a header without an accession number, an order with an impossible
time, and two films made unfit for AI triage (one blurred, one rotated).

| Stage | Code | Platform | Writes (database `rsingh_cxr_<layer>`) |
|---|---|---|---|
| land | `cde/jobs/land_sources.py` | CDE Spark | `s3a://.../rsingh_cxr/landing/<source>/<date>/` + `_manifest.json` |
| bronze | `cde/jobs/ingest_bronze.py` | CDE Spark | `bronze.ris_order`, `pacs_study`, `radiology_report`, `emr_patient`, `quarantine` |
| silver | `cde/jobs/build_silver.py` | CDE Spark | `silver.imaging_order`, `study`, `report`, `patient` (typed, de-duplicated) |
| gold | `cde/jobs/build_gold.py` | CDE Spark | `gold.dim_patient` (de-identified), `fact_study`, `fact_report` |
| score | `lakehouse/score_studies.py` | CAI job via Impala | `gold.triage_score` (worklist band, deciding model, film check, shadow band), `gold.model_score` (every head: champion and trial), `ref.triage_run` (snapshot read, PSI drift) |
| outcomes | `cde/jobs/build_outcomes.py` | CDE Spark | `gold.fact_triage_outcome` (FIFO, triage and shadow waits), `gold.daily_triage_summary`, `gold.daily_model_summary` (per model, stage, version: confusion counts against the signed report) |
| views | `sql/semantic/*.sql`, `scripts/run_semantic.py` | CDW Impala | `semantic.v_daily_kpi`, `v_triage_outcome`, `v_model_daily`, `v_silent_trial`, ... (12 views) |
| dashboards | `dataviz/build_dashboard.py` | Data Visualization | *CXR Triage Operations*, *CXR Model, Drift & Data Quality*, *CXR Models & Silent Trial* (42 visuals) |

Every CDE stage reconciles its counts into `ref.recon_results` (MATCHED / EXPLAINED /
MISMATCH) and `ref.load_audit`, and fails on a mismatch, which stops the DAG. The model
pipeline publishes `ref.training_set` (job 01) and `ref.model_event` (gate and deploy),
best-effort. The outcomes compare the two orderings on the same reading model, so the
difference in waits is the triage alone. On federal with the real models (10 dates,
479 films), pediatric pneumonia films waited a median 147-229 min under FIFO and 29-38
min in triage order; normal films wait longer (the cost). For the silent-trial numbers
see *Three models, one worklist* above.

Backfill or re-run a range of dates (after a change to the simulation or the models):
for each date in order, the four CDE stages land -> gold, then `cxr-06-score-studies`
with `CXR_BUSINESS_DATE`, then outcomes (a late report lands with the next date).

```bash
python scripts/run_local.py all                       # every demo date on local Spark + Iceberg (stub scorer, all heads)
./cde/scripts/deploy_jobs.sh                          # CDE files resource from `git archive` + 5 Spark jobs
./cde/scripts/deploy_dag.sh                           # DAG rsingh-cxr-orchestration (paused)
python cde/scripts/set_airflow_variables.py           # CXR_CAI_HOST / _PROJECT_ID / _API_KEY / _SCORE_JOB_ID
cde job run --name rsingh-cxr-orchestration --config-json '{"business_date": "2026-10-06"}'
cde job run --name rsingh-cxr-land-sources --arg=--business-date --arg=2026-10-06 --wait   # one stage by hand
python ci/setup_cai.py --run cxr-06-score-studies --env CXR_BUSINESS_DATE=2026-10-06       # score one date
python scripts/run_semantic.py --engine impala        # views + KPI consistency check
python dataviz/build_dashboard.py --import --connection federal-impala-1
python dataviz/build_dashboard.py --verify            # every visual through the Data API
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
| API v2 job `environment`: an object on create, a JSON **string** on update | `ci/setup_cai.py` sends each form where it is wanted |
| A job is created only if its script already exists in the project | sync the code first (`cxr-00`), then `ci/setup_cai.py` |
| Starting `cxr-00-sync-code` by hand also starts its child jobs | the pneumonia chain re-runs after a manual sync; expect it |
| `mlflow.get_run` inside a CAI job fails ("Missing the required parameter experiment_id") | `serve/registry.py` looks the experiment up by name |
| The AI Registry stores no version tags (create or update) | stage, approver, version: `ref.model_event`; KPIs: the MLflow run in the version |
| A model build can stall in `pushing` / `building` | job 04 times out after 30 min and rolls back; re-run the deploy job |
| The quota fits the endpoint plus one 2 vCPU workload | application stopped while jobs run; the trigger runs the chains one after another |

## Changes from build guide v1.0

- `torchvision` added to `requirements.txt`: `transformers` 5.x image processors need it (found in the local ViT run).
- `ci/create_cai_jobs.py` (session) and `ci/setup_cai.py` (laptop) create the jobs (section 9.2: six; now 18) instead of the UI; job timeouts are in seconds (API v2).
- `features.backbone_revision` pinned to `3f49326` (the Hugging Face commit of `google/vit-base-patch16-224` on 29 Sep 2026).
- Job 01 runs data checks (patient leakage, empty class, unreadable files, duplicate films across splits) and rebuilds a `--limit` smoke table automatically; the gate rejects a model trained on one.
- Job 04 rolls back to the previous champion if the build or deployment fails.
- A second workflow (`ci.yml`) runs the whole chain offline on every push and PR; the gate is also shown failing on cue.
- Nightly drift reports `TOO_FEW` below 30 incoming films instead of a PSI on noise.

- Three models (`models.<name>` in `config/pipeline.yaml`, one chain each), band NA, a silent trial with a shadow reading arm, promotion with go-live criteria and an approver, and an AI Registry version at every step (beyond the guide).

## Still to check on another workbench

Confirmed on federal: model build / deployment status strings, job-run status values,
the registry calls, the job environment forms. On a new workbench check once (see
`docs/DEMO_RUNBOOK.md`, section VERIFY): the `ML_RUNTIME_*` runtime detection (or set
`cai.runtime_identifier`), whether `cdsw-build.sh` runs on your runtime, whether the AI
Registry is configured, and the dataset licences (Kermany CC BY 4.0; NIH ChestX-ray14,
NIH Clinical Center terms: cite Wang et al. 2017).
