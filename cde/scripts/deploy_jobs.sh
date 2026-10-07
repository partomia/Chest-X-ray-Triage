#!/usr/bin/env bash
# Upload the lakehouse code to a CDE files resource and (re)create one Spark job per stage,
# each reading its application file from the resource (mounted at /app/mount, so the jobs
# find config/lakehouse.json and cde/reference/ next to cde/jobs/). The jobs need only
# PySpark and the standard library: no python-env resource.
#
# The GitHub repository is private, so instead of a CDE git repository (which would need a
# GitHub token stored in CDE) the files come from `git archive` of a commit: by default HEAD,
# which must be pushed. cde/DEPLOYED_SHA in the resource records the commit.
#
# After a code change: push, then re-run this script (the DAG: ./cde/scripts/deploy_dag.sh).
# Resources are small (about 50 studies a day): a 1-core / 2 GB driver and 1 to 2 executors
# of 1 core / 2 GB, well inside the shared federal YuniKorn queue.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
COMMIT="${COMMIT:-HEAD}"
RESOURCE="${RESOURCE:-rsingh-cxr-pipeline}"
JOB_PREFIX="${JOB_PREFIX:-rsingh-cxr}"
DB_PREFIX="${DB_PREFIX:-rsingh_cxr}"
LANDING="${LANDING:-s3a://federal-buk-574bcea0/data/IB/rsingh_cxr/landing}"
RESOURCES=(--driver-cores "${DRIVER_CORES:-1}" --driver-memory "${DRIVER_MEMORY:-2g}"
           --executor-cores "${EXECUTOR_CORES:-1}" --executor-memory "${EXECUTOR_MEMORY:-2g}"
           --min-executors 1 --initial-executors 1 --max-executors "${MAX_EXECUTORS:-2}"
           --conf spark.sql.shuffle.partitions=4
           --conf spark.sql.adaptive.enabled=true)
FILES=(config/lakehouse.json cde/reference/test_films.csv cde/dags/cxr_dag.py
       cde/jobs/cxr_common.py cde/jobs/land_sources.py cde/jobs/ingest_bronze.py cde/jobs/build_silver.py
       cde/jobs/build_gold.py cde/jobs/build_outcomes.py)

SHA=$(git rev-parse "${COMMIT}")
if ! git branch -r --contains "${SHA}" | grep -q .; then
  echo "commit ${SHA:0:7} is not on any remote branch: push it first" >&2
  exit 1
fi
STAGE=$(mktemp -d)
trap 'rm -rf "${STAGE}"' EXIT
git archive "${SHA}" "${FILES[@]}" | tar -x -C "${STAGE}"
echo "${SHA}" > "${STAGE}/cde/DEPLOYED_SHA"

echo "==> Resource ${RESOURCE} <- ${SHA:0:7}"
if ! cde resource describe --name "${RESOURCE}" &>/dev/null; then
  cde resource create --name "${RESOURCE}" --type files
fi
for f in "${FILES[@]}" cde/DEPLOYED_SHA; do
  cde resource upload --name "${RESOURCE}" --local-path "${STAGE}/${f}" --resource-path "${f}" --hide-progress-bars
done

create_job() {
  local name=$1 file=$2
  if cde job describe --name "${name}" &>/dev/null; then
    cde job delete --name "${name}"
  fi
  echo "==> Creating job ${name} (${file})"
  cde job create --name "${name}" --type spark \
    --mount-1-resource "${RESOURCE}" \
    --application-file "${file}" \
    "${RESOURCES[@]}" \
    --arg=--db-prefix --arg="${DB_PREFIX}" --arg=--landing --arg="${LANDING}"
}

create_job "${JOB_PREFIX}-land-sources"   "cde/jobs/land_sources.py"
create_job "${JOB_PREFIX}-ingest-bronze"  "cde/jobs/ingest_bronze.py"
create_job "${JOB_PREFIX}-build-silver"   "cde/jobs/build_silver.py"
create_job "${JOB_PREFIX}-build-gold"     "cde/jobs/build_gold.py"
create_job "${JOB_PREFIX}-build-outcomes" "cde/jobs/build_outcomes.py"

echo ""
echo "Jobs deployed from ${RESOURCE} (${SHA:0:7}). One stage by hand (run-time args replace the job's):"
echo "  cde job run --name ${JOB_PREFIX}-land-sources --arg=--business-date --arg=2026-09-28 \\"
echo "    --arg=--db-prefix --arg=${DB_PREFIX} --arg=--landing --arg=${LANDING} --wait"
echo "Then register the DAG: ./cde/scripts/deploy_dag.sh"
