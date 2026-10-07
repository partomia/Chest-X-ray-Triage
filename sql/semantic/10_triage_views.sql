-- Triage operations: what the reading room and the clinical lead look at.
-- Written with the default database names (rsingh_cxr_*) so it pastes into Hue as is;
-- scripts/run_semantic.py --db-prefix rewrites them. Portable Impala / Spark SQL.

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_triage_outcome;

CREATE VIEW rsingh_cxr_semantic.v_triage_outcome
COMMENT 'One row per chest film: model band, truth when reported, simulated FIFO vs triage report times'
AS
SELECT
    o.business_date,
    o.accession_no,
    o.study_ts,
    o.ordering_unit,
    o.clinical_priority,
    f.modality,
    f.view_position,
    f.station,
    f.age_years,
    CASE WHEN f.age_years < 1 THEN '<1' WHEN f.age_years < 3 THEN '1-2' ELSE '3-5' END AS age_band,
    p.sex,
    o.fifo_seq,
    o.triage_seq,
    o.probability,
    COALESCE(o.priority, 'UNSCORED')                     AS priority,
    o.model_version,
    o.truth_known,
    o.finding,
    o.outcome,
    o.fifo_wait_min,
    o.triage_wait_min,
    o.wait_saved_min,
    CASE WHEN o.truth = 1 THEN 1 ELSE 0 END              AS is_pneumonia,
    CASE WHEN o.priority = 'P1' THEN 1 ELSE 0 END        AS is_p1,
    CASE WHEN o.business_date = MAX(o.business_date) OVER () THEN 1 ELSE 0 END AS is_latest
FROM rsingh_cxr_gold.fact_triage_outcome o
JOIN rsingh_cxr_gold.fact_study f
  ON f.business_date = o.business_date AND f.accession_no = o.accession_no
LEFT JOIN rsingh_cxr_gold.dim_patient p
  ON p.patient_key = f.patient_key;

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_daily_kpi;

CREATE VIEW rsingh_cxr_semantic.v_daily_kpi
COMMENT 'Certified daily triage KPIs: volumes, bands, sensitivity/specificity on reported films, waits'
AS
SELECT
    s.business_date,
    s.studies,
    s.scored,
    s.reported,
    s.truth_completeness,
    s.pneumonia,
    s.p1,
    s.p2,
    s.p3,
    s.unscored,
    s.tp,
    s.fn,
    s.fp,
    s.tn,
    s.sensitivity,
    s.specificity,
    s.ppv,
    s.pneumonia_fifo_median_min,
    s.pneumonia_triage_median_min,
    s.pneumonia_fifo_median_min - s.pneumonia_triage_median_min AS pneumonia_median_minutes_saved,
    CASE WHEN s.pneumonia_fifo_median_min > 0
         THEN 1 - s.pneumonia_triage_median_min / s.pneumonia_fifo_median_min END AS pneumonia_wait_reduction,
    s.pneumonia_fifo_p90_min,
    s.pneumonia_triage_p90_min,
    s.normal_fifo_median_min,
    s.normal_triage_median_min,
    s.normal_triage_median_min - s.normal_fifo_median_min AS normal_median_minutes_added,
    s.pneumonia_minutes_saved,
    s.model_version,
    s.refreshed_at,
    CASE WHEN s.business_date = MAX(s.business_date) OVER () THEN 1 ELSE 0 END AS is_latest
FROM rsingh_cxr_gold.daily_triage_summary s;

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_band_daily;

CREATE VIEW rsingh_cxr_semantic.v_band_daily
COMMENT 'Per business date and priority band: films, reported, pneumonia and its rate among reported'
AS
SELECT
    business_date,
    COALESCE(priority, 'UNSCORED')                                     AS priority,
    COUNT(*)                                                            AS films,
    SUM(CASE WHEN truth_known THEN 1 ELSE 0 END)                        AS reported,
    SUM(CASE WHEN truth = 1 THEN 1 ELSE 0 END)                          AS pneumonia,
    CAST(SUM(CASE WHEN truth = 1 THEN 1 ELSE 0 END) AS DOUBLE)
        / NULLIF(SUM(CASE WHEN truth_known THEN 1 ELSE 0 END), 0)       AS pneumonia_rate,
    AVG(probability)                                                    AS avg_probability,
    AVG(triage_wait_min)                                                AS avg_triage_wait_min,
    AVG(fifo_wait_min)                                                  AS avg_fifo_wait_min,
    CASE WHEN business_date = MAX(business_date) OVER () THEN 1 ELSE 0 END AS is_latest
FROM rsingh_cxr_gold.fact_triage_outcome
GROUP BY business_date, COALESCE(priority, 'UNSCORED');

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_unit_daily;

CREATE VIEW rsingh_cxr_semantic.v_unit_daily
COMMENT 'Per business date and ordering unit: films, P1 films, pneumonia, average waits in both arms'
AS
SELECT
    business_date,
    ordering_unit,
    COUNT(*)                                                AS films,
    SUM(CASE WHEN priority = 'P1' THEN 1 ELSE 0 END)        AS p1_films,
    SUM(CASE WHEN truth = 1 THEN 1 ELSE 0 END)              AS pneumonia,
    AVG(fifo_wait_min)                                      AS avg_fifo_wait_min,
    AVG(triage_wait_min)                                    AS avg_triage_wait_min,
    AVG(CASE WHEN truth = 1 THEN fifo_wait_min END)         AS pneumonia_avg_fifo_wait_min,
    AVG(CASE WHEN truth = 1 THEN triage_wait_min END)       AS pneumonia_avg_triage_wait_min,
    CASE WHEN business_date = MAX(business_date) OVER () THEN 1 ELSE 0 END AS is_latest
FROM rsingh_cxr_gold.fact_triage_outcome
GROUP BY business_date, ordering_unit;
