-- Model lineage and drift: written by the CAI jobs (lakehouse/publish.py, lakehouse/score_studies.py).

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_model_event;

CREATE VIEW rsingh_cxr_semantic.v_model_event
COMMENT 'Gate decisions and deployments of the triage model, with the candidate metrics'
AS
SELECT
    `event` AS model_event,
    recorded_at,
    CAST(recorded_at AS DATE)  AS event_date,
    model_version,
    git_sha,
    feature_version,
    threshold,
    val_auroc,
    test_auroc,
    test_sensitivity,
    test_specificity,
    test_brier,
    train_rows,
    gate_passed,
    mlflow_run_id,
    detail,
    CASE WHEN ROW_NUMBER() OVER (PARTITION BY `event` ORDER BY recorded_at DESC) = 1 THEN 1 ELSE 0 END AS is_latest
FROM rsingh_cxr_ref.model_event;

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_scoring_run;

CREATE VIEW rsingh_cxr_semantic.v_scoring_run
COMMENT 'Each CAI scoring run of a business date: snapshot read, bands, missing films, PSI drift'
AS
SELECT
    business_date,
    run_id,
    triggered_by,
    status,
    source_snapshot_id,
    studies,
    scored,
    missing_films,
    p1,
    p2,
    p3,
    model_version,
    threshold,
    psi_max,
    drift_level,
    psi_json,
    started_at,
    ended_at,
    message,
    CASE WHEN ROW_NUMBER() OVER (PARTITION BY business_date ORDER BY ended_at DESC) = 1 THEN 1 ELSE 0 END AS is_final,
    CASE WHEN business_date = MAX(business_date) OVER () THEN 1 ELSE 0 END AS is_latest
FROM rsingh_cxr_ref.triage_run;

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_training_set;

CREATE VIEW rsingh_cxr_semantic.v_training_set
COMMENT 'Films, pneumonia and patients per feature version and split of the training table'
AS
SELECT feature_version, data_split, films, pneumonia,
       CAST(pneumonia AS DOUBLE) / NULLIF(films, 0) AS prevalence,
       patients, backbone, git_sha, created_at
FROM rsingh_cxr_ref.training_set;
