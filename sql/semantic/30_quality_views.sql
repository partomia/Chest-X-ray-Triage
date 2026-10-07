-- Data quality: reconciliation checks, stage status and quarantined source records.

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_data_quality;

CREATE VIEW rsingh_cxr_semantic.v_data_quality
COMMENT 'Every reconciliation check per business date, stage and entity: MATCHED, EXPLAINED or MISMATCH'
AS
SELECT
    business_date,
    stage,
    entity,
    check_name,
    expected,
    actual,
    actual - expected                                    AS difference,
    status,
    CASE WHEN status = 'MISMATCH' THEN 1 ELSE 0 END      AS is_mismatch,
    CASE WHEN status = 'EXPLAINED' THEN 1 ELSE 0 END     AS is_explained,
    CASE WHEN status = 'MATCHED' THEN 1 ELSE 0 END      AS is_matched,
    detail,
    logged_at,
    CASE WHEN business_date = MAX(business_date) OVER () THEN 1 ELSE 0 END AS is_latest
FROM rsingh_cxr_ref.recon_results;

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_stage_status;

CREATE VIEW rsingh_cxr_semantic.v_stage_status
COMMENT 'Latest stage-level status per business date (the * rows of ref.load_audit)'
AS
SELECT business_date, stage, status, rows_in, rows_out, rows_rejected, ended_at, pipeline_run
FROM (
    SELECT a.*, ROW_NUMBER() OVER (PARTITION BY business_date, stage ORDER BY ended_at DESC) AS rn
    FROM rsingh_cxr_ref.load_audit a
    WHERE entity = '*'
) x
WHERE rn = 1;

DROP VIEW IF EXISTS rsingh_cxr_semantic.v_quarantine;

CREATE VIEW rsingh_cxr_semantic.v_quarantine
COMMENT 'Source records that failed parsing or their contract, with every reason'
AS
SELECT business_date, `source` AS source_system, entity, `_source_file` AS file_name, `_line` AS line_no, reasons,
       `record` AS source_record, `_ingested_at` AS ingested_at
FROM rsingh_cxr_bronze.quarantine;
