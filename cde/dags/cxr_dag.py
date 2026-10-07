"""
Airflow DAG (CDE): the daily chest X-ray triage lakehouse, one business date per run.

  land_sources -> ingest_bronze -> build_silver -> build_gold      (CDE Spark: rsingh-cxr-* jobs)
    -> cai_score_studies                                          (CAI job cxr-06-score-studies)
    -> build_outcomes                                             (CDE Spark)

land_sources writes the hospital's RIS, PACS, report and EMR extracts of the date to the landing
zone (a stand-in for the source systems' own drops). Every CDE stage reconciles its counts into
<db_prefix>_ref.recon_results and fails on a MISMATCH, which stops the run before the next
layer. cai_score_studies starts the CAI job with CXR_BUSINESS_DATE in the job run's environment
(a job run ignores arguments; the environment map is applied) and polls it; the job scores the
date's gold.fact_study with the deployed champion and writes gold.triage_score in Impala.
build_outcomes then replays the reading list in triage order against FIFO for the date and the
day before (whose reports may have been signed since).

Airflow Variables (cde/scripts/set_airflow_variables.py sets them):
  CXR_CAI_HOST            https://federal-cml.<env>.cloudera.site (CAI workbench URL)
  CXR_CAI_PROJECT_ID      project id of rsingh-chest-x-ray-triage
  CXR_CAI_API_KEY         CAI API v2 key
  CXR_CAI_SCORE_JOB_ID    id of cxr-06-score-studies
Without them cai_score_studies fails: unscored studies are a reconciliation mismatch in
build_outcomes, so the outcomes of a date are never published without its triage scores.

Scheduled daily at 01:30 UTC (07:00 IST), after the other daily DAGs on the cluster. Manual
trigger (Trigger DAG w/ config): {"business_date": "2026-10-01"}; empty = yesterday.

Job names must match cde/scripts/deploy_jobs.sh exactly (CDEJobRunOperator fails with 404
"job not found" otherwise).
"""

import time
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from cloudera.cdp.airflow.operators.cde_operator import CDEJobRunOperator

JOB_PREFIX = "rsingh-cxr"
DB_PREFIX = "rsingh_cxr"
# Scheduled runs: the day before the interval end; manual runs: the business_date param (empty = yesterday).
BUSINESS_DATE = ("{{ params.business_date or ((data_interval_end - macros.timedelta(days=1)).strftime('%Y-%m-%d') "
                 "if dag_run.run_type == 'scheduled' else (macros.datetime.utcnow() - macros.timedelta(days=1))"
                 ".strftime('%Y-%m-%d')) }}")
DAILY = "30 1 * * *"
TERMINAL_OK = {"succeeded"}
TERMINAL_BAD = {"failed", "stopped", "timedout"}
CAI_DEADLINE_MIN = 60


def trigger_cai_score(business_date: str, run_id: str, **_):
    host = Variable.get("CXR_CAI_HOST", default_var="").rstrip("/")
    if not host:
        raise AirflowException("Airflow Variable CXR_CAI_HOST is not set (cde/scripts/set_airflow_variables.py)")
    project = Variable.get("CXR_CAI_PROJECT_ID")
    job = Variable.get("CXR_CAI_SCORE_JOB_ID")
    headers = {"Authorization": f"Bearer {Variable.get('CXR_CAI_API_KEY')}", "Content-Type": "application/json"}
    env = {"CXR_BUSINESS_DATE": business_date, "CXR_TRIGGERED_BY": f"airflow:{run_id}"}
    url = f"{host}/api/v2/projects/{project}/jobs/{job}/runs"
    resp = requests.post(url, json={"environment": env}, headers=headers, timeout=60)
    resp.raise_for_status()
    cai_run = resp.json()["id"]
    print(f"Started CAI job run {cai_run} (cxr-06-score-studies) with environment: {env}")

    deadline = time.time() + CAI_DEADLINE_MIN * 60
    while time.time() < deadline:
        time.sleep(30)
        r = requests.get(f"{url}/{cai_run}", headers=headers, timeout=60)
        r.raise_for_status()
        status = str(r.json().get("status", "")).lower().replace("engine_", "")
        print(f"CAI run {cai_run}: {status}")
        if status in TERMINAL_OK:
            return cai_run
        if status in TERMINAL_BAD:
            raise AirflowException(f"CAI job run {cai_run} ended with status {status} "
                                   f"(see ref.triage_run and the job's history in the workbench)")
    raise AirflowException(f"CAI job run {cai_run} did not finish within {CAI_DEADLINE_MIN} minutes")


default_args = {
    "owner": "cxr-triage",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="cxr_triage_lakehouse",
    description="RIS/PACS/report/EMR extracts -> bronze/silver/gold with reconciliation (CDE) -> "
                "triage scoring (CAI) -> FIFO vs triage outcomes (CDE)",
    default_args=default_args,
    schedule_interval=DAILY,
    # With catchup=False an unpaused DAG runs the latest closed interval as soon as it is
    # registered, hence paused on creation: unpausing is a deliberate step.
    start_date=datetime(2026, 10, 6, 1, 30),
    catchup=False,
    is_paused_upon_creation=True,
    # every stage replaces its business date's partitions: two runs at once would collide
    max_active_runs=1,
    params={"business_date": ""},
    tags=["cxr", "iceberg", "radiology", "triage"],
) as dag:

    def cde_task(task_id: str, job: str, *extra: str) -> CDEJobRunOperator:
        # run-time args replace the job's own args, so repeat --db-prefix
        return CDEJobRunOperator(
            task_id=task_id,
            job_name=f"{JOB_PREFIX}-{job}",
            overrides={"spark": {"args": ["--business-date", BUSINESS_DATE, "--db-prefix", DB_PREFIX,
                                          "--pipeline-run", "{{ run_id }}", *extra]}},
            wait=True,
        )

    land = cde_task("land_sources", "land-sources")
    bronze = cde_task("ingest_bronze", "ingest-bronze")
    silver = cde_task("build_silver", "build-silver")
    gold = cde_task("build_gold", "build-gold")
    # retries=0: a scoring failure (missing films, no champion) is not transient
    score = PythonOperator(task_id="cai_score_studies", python_callable=trigger_cai_score,
                           op_kwargs={"business_date": BUSINESS_DATE, "run_id": "{{ run_id }}"}, retries=0)
    outcomes = cde_task("build_outcomes", "build-outcomes")

    land >> bronze >> silver >> gold >> score >> outcomes
