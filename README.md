# Chest X-ray Triage on Cloudera AI Workbench

Decision-support demo: scores every incoming chest X-ray for pneumonia and moves
likely-abnormal films to the top of the radiologist's worklist, with a heatmap of
the region that drove the score and a one-click override that feeds the next
retrain. **Workflow prioritisation, not diagnosis**: the radiologist still reads
every film; any clinical use needs regulatory clearance and local validation.

Everything runs inside one Cloudera AI (CAI) Workbench project: Sessions, Jobs,
Experiments (MLflow), Models, Applications and API v2. No CDW, no CDE. GitHub
holds only code and configuration; it triggers and follows the pipeline through
the CAI API, so patient images never leave the workbench.

```
Feature Engineering -> Model Training -> Model Evaluation -> KPI Gate      (what the code does)
GitHub Push -> Train & Validate -> KPI Gate -> Cloudera AI API (deploy)    (who runs it, and when)
```

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

## Run it locally (no CAI, no dataset, no GPU)

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-ci.txt
.venv/bin/pytest -q                                    # 28 tests, ~20 s, includes the app

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
