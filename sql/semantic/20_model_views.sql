-- Model lineage and drift: written by the CAI jobs (lakehouse/publish.py, lakehouse/score_studies.py).

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_model_event;

CREATE VIEW rsingh_cxr_semantic.v_model_event
COMMENT 'Gate decisions, silent trials, deployments and promotions of every model, with the candidate metrics'
AS
SELECT
    `event` AS model_event,
    COALESCE(model, 'pneumonia') AS model,
    stage,
    registry_version,
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
    CASE WHEN ROW_NUMBER() OVER (PARTITION BY COALESCE(model, 'pneumonia'), `event` ORDER BY recorded_at DESC) = 1
         THEN 1 ELSE 0 END AS is_latest
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
    not_triaged,
    film_unsuitable,
    heads_json,
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
COMMENT 'Films, positives and patients per model, feature version and split of its training table'
AS
SELECT COALESCE(model, 'pneumonia') AS model, feature_version, data_split, films,
       COALESCE(positives, pneumonia) AS positives,
       CAST(COALESCE(positives, pneumonia) AS DOUBLE) / NULLIF(films, 0) AS prevalence,
       patients, backbone, git_sha, created_at
FROM rsingh_cxr_ref.training_set;

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_model_daily;

CREATE VIEW rsingh_cxr_semantic.v_model_daily
COMMENT 'Per business date and model head (champion or silent trial): in-scope films, reported positives, accuracy at its own threshold'
AS
SELECT
    business_date,
    model,
    stage,
    model_version,
    scored,
    in_scope,
    reported,
    positives,
    tp,
    fn,
    fp,
    tn,
    sensitivity,
    specificity,
    ppv,
    refreshed_at,
    CASE WHEN business_date = MAX(business_date) OVER () THEN 1 ELSE 0 END AS is_latest
FROM rsingh_cxr_gold.daily_model_summary;

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_silent_trial;

CREATE VIEW rsingh_cxr_semantic.v_silent_trial
COMMENT 'Evidence so far for each model version in silent trial: what cxr-07-promote-champion weighs against go-live criteria'
AS
SELECT
    model,
    model_version,
    COUNT(DISTINCT business_date)                         AS trial_days,
    MIN(business_date)                                    AS first_day,
    MAX(business_date)                                    AS last_day,
    SUM(in_scope)                                         AS in_scope,
    SUM(reported)                                         AS reported,
    SUM(tp + fn)                                          AS positives,
    SUM(tp)                                               AS tp,
    SUM(fn)                                               AS fn,
    SUM(fp)                                               AS fp,
    SUM(tn)                                               AS tn,
    CAST(SUM(tp) AS DOUBLE) / NULLIF(SUM(tp + fn), 0)     AS sensitivity,
    CAST(SUM(tn) AS DOUBLE) / NULLIF(SUM(tn + fp), 0)     AS specificity
FROM rsingh_cxr_gold.daily_model_summary
WHERE stage = 'silent_trial'
GROUP BY model, model_version;
