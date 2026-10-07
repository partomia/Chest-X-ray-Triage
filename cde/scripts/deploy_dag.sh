#!/usr/bin/env bash
# Register/update the Airflow DAG as a `--type airflow` CDE job from the files resource that
# deploy_jobs.sh uploads (run that first: it also uploads cde/dags/cxr_dag.py). Re-run both
# after every DAG change. The DAG registers paused: unpause it deliberately to start the daily
# schedule, or trigger one business date by hand (below).

set -euo pipefail

RESOURCE="${RESOURCE:-rsingh-cxr-pipeline}"
DAG_JOB_NAME="${DAG_JOB_NAME:-rsingh-cxr-orchestration}"
DAG_PATH="cde/dags/cxr_dag.py"

if cde job describe --name "${DAG_JOB_NAME}" &>/dev/null; then
  echo "==> Updating ${DAG_JOB_NAME}"
  cde job update --name "${DAG_JOB_NAME}" --dag-file "${DAG_PATH}" --mount-1-resource "${RESOURCE}"
  echo "Give Airflow ~30s to re-parse before triggering."
  exit 0
fi

echo "==> Creating ${DAG_JOB_NAME}"
cde job create --name "${DAG_JOB_NAME}" --type airflow --dag-file "${DAG_PATH}" --mount-1-resource "${RESOURCE}"
echo "DAG cxr_triage_lakehouse registered (paused). Set its Variables (cde/scripts/set_airflow_variables.py),"
echo "then trigger one business date:"
echo "  cde job run --name ${DAG_JOB_NAME} --config-json '{\"business_date\": \"2026-10-06\"}'"
