#!/usr/bin/env python3
"""
The three chest X-ray triage dashboards in Cloudera Data Visualization, as code:

  CXR Triage Operations          the reading room: today's worklist in triage order, films per
                                 band and unit, pneumonia waits under FIFO vs triage, the trend
  CXR Model, Drift & Data Quality  the models behind the worklist (gate decisions, deployments,
                                 scoring runs, PSI drift) and the pipeline's reconciliation checks
                                 and quarantined source records
  CXR Models & Silent Trial      the pneumothorax model in silent trial: its evidence against the
                                 go-live criteria, adult pneumothorax waits today vs if it were
                                 live; films unfit for AI triage; every model head per day

Datasets, visuals and sheets are declared below; this script turns them into one Data
Visualization export file (dataviz/cxr_dashboards.json) with fixed UUIDs and primary keys
(12000+, apart from the other projects' dashboards on the same instance), so an import
updates the dashboards in place. Every dataset is a view in rsingh_cxr_semantic (sql/semantic/).

  python dataviz/build_dashboard.py                          # write the file (column types from Impala)
  python dataviz/build_dashboard.py --import --connection X  # and import it into the CDW instance
  python dataviz/build_dashboard.py --check                  # every visual's query directly on Impala
  python dataviz/build_dashboard.py --verify                 # every visual through the Data API
  python dataviz/build_dashboard.py --list-connections

Column types come from Impala (CXR_IMPALA_USER / CXR_IMPALA_PASSWORD). The CDW Data
Visualization instance (config/lakehouse.json dataviz_url) signs users in with SAML, so the API
calls need a Data Visualization API key in CXR_VIZ_API_KEY. Without one, import the file in the
UI: Data -> Import Visual Artifacts, and pick the connection to the federal Impala warehouse.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "dataviz" / "cxr_dashboards.json"
DB = "rsingh_cxr_semantic"
NS = uuid.UUID("8c1f4e2b-7d93-4a6e-b215-3f0a9c6d4e71")
DASHBOARD_PK0 = 12000
DATASET_PK0 = 12100
VISUAL_PK0 = 12200
# The export format version, used when the instance's own cannot be read (no API key).
DEFAULT_VERSION = {"Arcviz Version": "8.1.4.1000", "Description": "8.1.4.1000-4"}

DATASETS = {                          # key: (name, view, integer columns that are dimensions)
    "kpi": ("CXR - Daily triage KPIs", "v_daily_kpi", {"is_latest"}),
    "film": ("CXR - Triage outcome per film", "v_triage_outcome",
             {"is_latest", "fifo_seq", "triage_seq", "age_years", "is_pneumonia", "is_p1"}),
    "band": ("CXR - Bands per day", "v_band_daily", {"is_latest"}),
    "unit": ("CXR - Ordering units per day", "v_unit_daily", {"is_latest"}),
    "model": ("CXR - Model events", "v_model_event", {"is_latest", "train_rows"}),
    "scoring": ("CXR - Scoring runs", "v_scoring_run", {"is_latest", "is_final", "source_snapshot_id"}),
    "dq": ("CXR - Data quality checks", "v_data_quality", {"is_latest"}),
    "quarantine": ("CXR - Quarantined source records", "v_quarantine", {"line_no"}),
    "mdaily": ("CXR - Model heads per day", "v_model_daily", {"is_latest"}),
    "trial": ("CXR - Silent trial evidence", "v_silent_trial", set()),
}

LATEST = "[is_latest] = 1"
PNEUMONIA_DEPLOYED = ["[model] = 'pneumonia'", "[model_event] IN ('DEPLOYED', 'PROMOTED')"]
PTX = "[model] = 'pneumothorax'"
pct = lambda e: f"round(100 * {e}, 1)"  # noqa: E731

OPS_SHEETS = [
    ("Latest day", [
        dict(type="kpi", ds="kpi", title="Films", measures=[("sum([studies])", "Films")], filters=[LATEST],
             pos=(1, 1, 13, 10)),
        dict(type="kpi", ds="kpi", title="P1 - read first", measures=[("sum([p1])", "P1")], filters=[LATEST],
             pos=(14, 1, 13, 10)),
        dict(type="kpi", ds="kpi", title="Pneumonia: median wait, FIFO (min)",
             measures=[("max([pneumonia_fifo_median_min])", "FIFO min")], filters=[LATEST], pos=(27, 1, 13, 10)),
        dict(type="kpi", ds="kpi", title="Pneumonia: median wait, triage (min)",
             measures=[("max([pneumonia_triage_median_min])", "Triage min")], filters=[LATEST], pos=(40, 1, 13, 10)),
        dict(type="kpi", ds="kpi", title="Sensitivity % (reported films)",
             measures=[(pct("max([sensitivity])"), "Sensitivity %")], filters=[LATEST], pos=(53, 1, 12, 10)),
        dict(type="trellis-bars", ds="band", title="Films and pneumonia per band (latest day)",
             x=[("priority", "Band")], measures=[("sum([films])", "Films"), ("sum([pneumonia])", "Pneumonia")],
             filters=[LATEST], sort_dim="priority", pos=(1, 11, 32, 22)),
        dict(type="trellis-bars", ds="unit", title="Pneumonia: average wait (min) per ordering unit, FIFO vs triage",
             x=[("ordering_unit", "Unit")],
             measures=[("round(max([pneumonia_avg_fifo_wait_min]), 1)", "FIFO"),
                       ("round(max([pneumonia_avg_triage_wait_min]), 1)", "Triage")],
             filters=[LATEST], sort_dim="ordering_unit", pos=(33, 11, 32, 22)),
        dict(type="table", ds="film", title="Worklist in triage order (latest day)",
             dims=[("triage_seq", "Triage #"), ("fifo_seq", "FIFO #"), ("accession_no", "Accession"),
                   ("ordering_unit", "Unit"), ("clinical_priority", "Clinical"), ("priority", "Band"),
                   ("outcome", "Outcome"), ("finding", "Report")],
             measures=[("max([probability])", "P(pneumonia)"), ("max([triage_wait_min])", "Triage wait"),
                       ("max([fifo_wait_min])", "FIFO wait")],
             filters=[LATEST], sort_dim="triage_seq", pos=(1, 33, 64, 30)),
    ]),
    ("Trend", [
        dict(type="trellis-lines", ds="kpi", title="Pneumonia median wait (min): FIFO vs triage",
             x=[("business_date", "Business date")],
             measures=[("max([pneumonia_fifo_median_min])", "FIFO"), ("max([pneumonia_triage_median_min])", "Triage")],
             pos=(1, 1, 32, 22)),
        dict(type="trellis-lines", ds="kpi", title="Normal films: median wait (min), the cost of triage",
             x=[("business_date", "Business date")],
             measures=[("max([normal_fifo_median_min])", "FIFO"), ("max([normal_triage_median_min])", "Triage")],
             pos=(33, 1, 32, 22)),
        dict(type="trellis-bars", ds="band", title="Films per band per day",
             x=[("business_date", "Business date")], measures=[("sum([films])", "Films")],
             color=[("priority", "Band")], pos=(1, 23, 32, 22)),
        dict(type="trellis-lines", ds="kpi", title="Sensitivity and specificity % on reported films",
             x=[("business_date", "Business date")],
             measures=[(pct("max([sensitivity])"), "Sensitivity %"), (pct("max([specificity])"), "Specificity %")],
             pos=(33, 23, 32, 22)),
        dict(type="table", ds="kpi", title="Daily triage KPIs",
             dims=[("business_date", "Business date")],
             measures=[("sum([studies])", "Films"), ("sum([reported])", "Reported"), ("sum([pneumonia])", "Pneumonia"),
                       ("sum([p1])", "P1"), ("sum([p2])", "P2"), ("sum([p3])", "P3"), ("sum([fn])", "Missed (FN)"),
                       (pct("max([sensitivity])"), "Sens %"), (pct("max([specificity])"), "Spec %"),
                       ("max([pneumonia_median_minutes_saved])", "Median min saved"),
                       (pct("max([truth_completeness])"), "Truth %")],
             sort_dim="business_date", sort_asc=False, pos=(1, 45, 64, 22)),
    ]),
]

MODEL_SHEETS = [
    ("Model and drift", [
        dict(type="kpi", ds="model", title="Pneumonia champion: test AUROC",
             measures=[("max([test_auroc])", "AUROC")], filters=[*PNEUMONIA_DEPLOYED, LATEST],
             may_be_empty=True, pos=(1, 1, 16, 10)),
        dict(type="kpi", ds="model", title="Pneumonia champion: test sensitivity",
             measures=[("max([test_sensitivity])", "Sensitivity")], filters=[*PNEUMONIA_DEPLOYED, LATEST],
             may_be_empty=True, pos=(17, 1, 16, 10)),
        dict(type="kpi", ds="scoring", title="Films scored (latest day)", measures=[("sum([scored])", "Scored")],
             filters=[LATEST, "[is_final] = 1"], pos=(33, 1, 16, 10)),
        dict(type="kpi", ds="scoring", title="Worst feature PSI (latest day)",
             measures=[("max([psi_max])", "PSI")], filters=[LATEST, "[is_final] = 1"], pos=(49, 1, 16, 10)),
        dict(type="trellis-lines", ds="scoring", title="Worst quality-feature PSI per scored day (warn 0.10, alert 0.25)",
             x=[("business_date", "Business date")], measures=[("max([psi_max])", "PSI")],
             filters=["[is_final] = 1"], pos=(1, 11, 32, 22)),
        dict(type="trellis-bars", ds="scoring", title="Bands assigned per scored day",
             x=[("business_date", "Business date")],
             measures=[("sum([p1])", "P1"), ("sum([p2])", "P2"), ("sum([p3])", "P3")],
             filters=["[is_final] = 1"], pos=(33, 11, 32, 22)),
        dict(type="table", ds="model", title="Gate decisions, silent trials, deployments and promotions",
             dims=[("recorded_at", "When"), ("model", "Model"), ("stage", "Stage"), ("model_event", "Event"),
                   ("model_version", "Version")],
             measures=[("max([test_auroc])", "Test AUROC"), ("max([test_sensitivity])", "Test sens"),
                       ("max([test_specificity])", "Test spec"), ("max([threshold])", "Threshold"),
                       ("max([train_rows])", "Train films")],
             may_be_empty=True, sort_dim="recorded_at", sort_asc=False, pos=(1, 33, 64, 18)),
        dict(type="table", ds="scoring", title="Scoring runs (CAI job cxr-06 per business date)",
             dims=[("business_date", "Business date"), ("run_id", "Run"), ("status", "Status"),
                   ("drift_level", "Drift"), ("model_version", "Model"), ("triggered_by", "Triggered by")],
             measures=[("sum([studies])", "Studies"), ("sum([scored])", "Scored"),
                       ("sum([missing_films])", "Missing films"), ("max([psi_max])", "PSI")],
             sort_dim="business_date", sort_asc=False, pos=(1, 51, 64, 22)),
    ]),
    ("Data quality", [
        dict(type="kpi", ds="dq", title="Checks (latest day)", measures=[("sum(1)", "Checks")], filters=[LATEST],
             pos=(1, 1, 16, 10)),
        dict(type="kpi", ds="dq", title="Matched", measures=[("sum([is_matched])", "Matched")], filters=[LATEST],
             pos=(17, 1, 16, 10)),
        dict(type="kpi", ds="dq", title="Explained (a known cause)", measures=[("sum([is_explained])", "Explained")],
             filters=[LATEST], pos=(33, 1, 16, 10)),
        dict(type="kpi", ds="dq", title="Mismatches", measures=[("sum([is_mismatch])", "Mismatches")],
             filters=[LATEST], pos=(49, 1, 16, 10)),
        dict(type="trellis-bars", ds="dq", title="Checks per business date",
             x=[("business_date", "Business date")],
             measures=[("sum([is_matched])", "Matched"), ("sum([is_explained])", "Explained"),
                       ("sum([is_mismatch])", "Mismatch")], pos=(1, 11, 64, 20)),
        dict(type="table", ds="dq", title="Explained and mismatched checks, with the reason",
             dims=[("business_date", "Business date"), ("stage", "Stage"), ("entity", "Entity"),
                   ("check_name", "Check"), ("status", "Status"), ("detail", "Detail")],
             measures=[("sum([expected])", "Expected"), ("sum([actual])", "Actual")],
             filters=["[status] <> 'MATCHED'"], may_be_empty=True, sort_dim="business_date", sort_asc=False,
             pos=(1, 31, 64, 22)),
        dict(type="table", ds="quarantine", title="Quarantined source records",
             dims=[("business_date", "Business date"), ("source_system", "Source"), ("file_name", "File"),
                   ("reasons", "Reasons"), ("source_record", "Record")],
             measures=[("max([line_no])", "Line")], may_be_empty=True, sort_dim="business_date", sort_asc=False,
             pos=(1, 53, 64, 18)),
    ]),
]

TRIAL_SHEETS = [
    ("Silent trial", [
        dict(type="kpi", ds="trial", title="Pneumothorax: days in silent trial", measures=[("max([trial_days])", "Days")],
             filters=[PTX], may_be_empty=True, pos=(1, 1, 13, 10)),
        dict(type="kpi", ds="trial", title="Reported pneumothorax (live films)", measures=[("sum([positives])", "Positives")],
             filters=[PTX], may_be_empty=True, pos=(14, 1, 13, 10)),
        dict(type="kpi", ds="trial", title="Trial sensitivity %", measures=[(pct("max([sensitivity])"), "Sensitivity %")],
             filters=[PTX], may_be_empty=True, pos=(27, 1, 13, 10)),
        dict(type="kpi", ds="trial", title="Trial specificity %", measures=[(pct("max([specificity])"), "Specificity %")],
             filters=[PTX], may_be_empty=True, pos=(40, 1, 13, 10)),
        dict(type="kpi", ds="kpi", title="Pneumothorax: median wait if live (min)",
             measures=[("max([pneumothorax_shadow_median_min])", "If live")], filters=[LATEST], pos=(53, 1, 12, 10)),
        dict(type="trellis-lines", ds="kpi", title="Pneumothorax median wait (min): FIFO, today's worklist, if the trial were live",
             x=[("business_date", "Business date")],
             measures=[("max([pneumothorax_fifo_median_min])", "FIFO"),
                       ("max([pneumothorax_triage_median_min])", "Worklist today"),
                       ("max([pneumothorax_shadow_median_min])", "If live")], pos=(1, 11, 32, 22)),
        dict(type="trellis-lines", ds="mdaily", title="Pneumothorax trial: sensitivity and specificity % per day",
             x=[("business_date", "Business date")],
             measures=[(pct("max([sensitivity])"), "Sensitivity %"), (pct("max([specificity])"), "Specificity %")],
             filters=[PTX, "[stage] = 'silent_trial'"], may_be_empty=True, pos=(33, 11, 32, 22)),
        dict(type="table", ds="trial", title="Silent-trial evidence per model version (go-live: cxr-07-promote-champion)",
             dims=[("model", "Model"), ("model_version", "Version"), ("first_day", "From"), ("last_day", "To")],
             measures=[("max([trial_days])", "Days"), ("sum([in_scope])", "In scope"), ("sum([positives])", "Positives"),
                       ("sum([tp])", "TP"), ("sum([fn])", "FN"), ("sum([fp])", "FP"), ("sum([tn])", "TN"),
                       (pct("max([sensitivity])"), "Sens %"), (pct("max([specificity])"), "Spec %")],
             may_be_empty=True, sort_dim="model", pos=(1, 33, 64, 16)),
    ]),
    ("Every model", [
        dict(type="kpi", ds="kpi", title="Adult films (latest day)", measures=[("sum([adults])", "Adults")],
             filters=[LATEST], pos=(1, 1, 16, 10)),
        dict(type="kpi", ds="kpi", title="No AI triage - NA (latest day)", measures=[("sum([not_triaged])", "NA")],
             filters=[LATEST], pos=(17, 1, 16, 10)),
        dict(type="kpi", ds="kpi", title="Films unfit for AI triage (latest day)",
             measures=[("sum([film_unsuitable])", "Unfit")], filters=[LATEST], pos=(33, 1, 16, 10)),
        dict(type="kpi", ds="kpi", title="Pneumothorax films (latest day)", measures=[("sum([pneumothorax])", "Pneumothorax")],
             filters=[LATEST], pos=(49, 1, 16, 10)),
        dict(type="trellis-bars", ds="mdaily", title="In-scope films per model head per day",
             x=[("business_date", "Business date")], measures=[("sum([in_scope])", "In scope")],
             color=[("model", "Model")], pos=(1, 11, 64, 20)),
        dict(type="table", ds="mdaily", title="Every model head on the latest day, at its own threshold",
             dims=[("model", "Model"), ("stage", "Stage"), ("model_version", "Version")],
             measures=[("sum([scored])", "Scored"), ("sum([in_scope])", "In scope"), ("sum([reported])", "Reported"),
                       ("sum([positives])", "Positives"), ("sum([tp])", "TP"), ("sum([fn])", "FN"),
                       (pct("max([sensitivity])"), "Sens %"), (pct("max([specificity])"), "Spec %")],
             filters=[LATEST], sort_dim="model", pos=(1, 31, 64, 16)),
    ]),
]

DASHBOARDS = [
    dict(title="CXR Triage Operations", pk=DASHBOARD_PK0, key="operations", sheets=OPS_SHEETS, main_ds="kpi",
         subtitle="Chest X-ray worklist in triage order, and how long pneumonia films wait under FIFO vs triage"),
    dict(title="CXR Model, Drift & Data Quality", pk=DASHBOARD_PK0 + 1, key="model-quality", sheets=MODEL_SHEETS,
         main_ds="scoring", subtitle="Gate decisions, deployments, scoring runs and PSI drift; pipeline reconciliation"),
    dict(title="CXR Models & Silent Trial", pk=DASHBOARD_PK0 + 2, key="models-trial", sheets=TRIAL_SHEETS,
         main_ds="trial", subtitle="Pneumothorax in silent trial: its evidence, and what going live would change; "
                                   "the film check and every model head"),
]

SHELVES = {
    "kpi": [("dimensions_shelf", 1, 1), ("aggregates_shelf", 1, 2), ("compare_shelf", 1, 2), ("label_shelf", 1, 2),
            ("tooltip_shelf", 1, 2), ("x_shelf", 1, 1), ("y_shelf", 1, 1), ("filters_shelf", 2, 3)],
    "table": [("dimensions_shelf", 1, 1), ("aggregates_shelf", 1, 2), ("filters_shelf", 2, 3)],
    "trellis-bars": [("x_shelf", 1, 3), ("y_shelf", 1, 3), ("color_shelf", 1, 3), ("tooltip_shelf", 1, 2),
                     ("drill_shelf", 1, 1), ("label_shelf", 1, 2), ("filters_shelf", 2, 3)],
    "trellis-lines": [("x_shelf", 1, 3), ("y_shelf", 1, 3), ("color_shelf", 1, 3), ("tooltip_shelf", 1, 2),
                      ("filters_shelf", 2, 3)],
}


def visuals_of(d: dict):
    for sheet, items in d["sheets"]:
        for v in items:
            yield sheet, v


def uid(*parts: str) -> str:
    return str(uuid.uuid5(NS, "/".join(parts)))


class _Impala:
    """query(sql) -> (columns, rows), over lakehouse.store.ImpalaStore."""

    def __init__(self):
        from lakehouse.store import ImpalaStore

        self.store = ImpalaStore()

    def query(self, sql: str) -> tuple[list[str], list[tuple]]:
        cur = self.store._connect().cursor()
        try:
            cur.execute(sql)
            cols = [d[0].split(".")[-1] for d in cur.description or []]
            return cols, [tuple(r) for r in cur.fetchall()] if cur.description else []
        finally:
            cur.close()


def impala():
    return _Impala()


def column_types(engine, ds_key: str) -> dict[str, str]:
    cols, rows = engine.query(f"DESCRIBE {DB}.{DATASETS[ds_key][1]}")
    return {r[cols.index("name")]: str(r[cols.index("type")]).upper() for r in rows}


def is_dim(ds_key: str, col: str, typ: str) -> bool:
    return col in DATASETS[ds_key][2] or not any(t in typ for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL"))


def dataset_record(key: str, pk: int, types: dict[str, str], conn_id: int, dashboards: list[int]) -> dict:
    name, view, _ = DATASETS[key]
    table = f"{DB}.{view}"
    cols = [{"alias": c, "type": t, "name": c, "isdim": is_dim(key, c, t)} for c, t in types.items()]
    return {"model": "datasets.dataset", "pk": pk, "fields": {
        "dataconnection": conn_id, "dataset_name": name, "dataset_type": "singletable", "dataset_detail": table,
        "dataset_description": f"{table} (sql/semantic)",
        "dataset_info": json.dumps([{"tablename": table, "columns": cols}]),
        "dataset_tablenames": json.dumps([table]), "uuid": uid("dataset", key), "imported_uuid": None,
        "cache_sequence": 0, "dataset_settings": "{}", "search_enabled": False, "dashboards": dashboards,
        "version_id": pk, "version_group_id": pk, "is_active_version": True,
        "version_name": "chest-x-ray-triage", "is_named_version": False}}


def dim_item(col: str, alias: str, typ: str) -> dict:
    return {"dataset_colname": col, "dataset_coltype": typ, "expression_for_trigger": f"[{col}]", "col_alias": alias}


def measure_item(expr: str, alias: str) -> dict:
    return {"custom_expr": expr, "expression_for_trigger": expr, "expr_hasagg": True, "col_alias": alias,
            "dataset_colname": alias, "dataset_coltype": "DOUBLE"}


def filter_item(expr: str) -> dict:
    return {"custom_expr": expr, "expression_for_trigger": expr, "filter_input": {}, "filter_data": [],
            "dataset_colname": "", "dataset_coltype": "STRING", "filter_column": ""}


def visual_record(v: dict, pk: int, sheet: str, types: dict[str, str], dataset_pk: int, dash: dict) -> dict:
    kind = v["type"]
    shelves = {name: [] for name, _, _ in SHELVES[kind]}
    sources = {}

    def add_dims(shelf, pairs):
        for col, alias in pairs:
            shelves[shelf].append(dim_item(col, alias, types[col]))
            sources[f"[{col}] as 'sub:{alias}'"] = shelf

    def add_measures(shelf, pairs):
        for expr, alias in pairs:
            shelves[shelf].append(measure_item(expr, alias))
            sources[f"{expr} as 'sub:{alias}'"] = shelf

    if kind in ("kpi", "table"):
        add_dims("dimensions_shelf", v.get("dims", []))
        add_measures("aggregates_shelf", v["measures"])
    else:
        add_dims("x_shelf", v["x"])
        add_measures("y_shelf", v["measures"])
        add_dims("color_shelf", v.get("color", []))
    for expr in v.get("filters", []):
        shelves["filters_shelf"].append(filter_item(expr))
        sources[expr] = "filters_shelf"
    if v.get("sort_desc"):
        shelves["y_shelf"][0]["order"] = {"priority": 1, "ascending": False}
    if v.get("sort_dim"):
        shelf = "dimensions_shelf" if kind == "table" else "x_shelf"
        item = next(i for i in shelves[shelf] if i["dataset_colname"] == v["sort_dim"])
        item["order"] = {"priority": 1, "ascending": v.get("sort_asc", True)}
    report = {
        "report_title": v["title"], "report_subtitle": "", "dashboard_id": dash["pk"],
        "limit": v.get("limit", 1000), "sample_pct": "Off", "selected_segments": [], "report_derived_data": [],
        "click_behaviors": {}, "sort_orders_asc": {}, "user_settings": {}, **shelves,
        "core": {"viz_type": kind, "saved_shelf_sources": sources,
                 "shelves": [{"name": n, "shelf_type": s, "column_type": c} for n, s, c in SHELVES[kind]]},
    }
    return {"model": "reports.report", "pk": pk, "fields": {
        "report_name": "", "report_description": f"{dash['title']} / {sheet}", "dataset": dataset_pk,
        "workspace": 1, "report_type": kind, "report_mode": "", "dashboard_url_name": "",
        "report_data": json.dumps({"report_data": report, "report_type": kind}), "shared_visual_dashboards": None,
        "parent_report": None, "uuid": uid("visual", dash["key"], sheet, v["title"]), "imported_uuid": None,
        "has_css_styles": False, "report_search_text": ""}}


def dashboard_record(d: dict, pk0: int, visuals: list[dict], ds_pk: dict[str, int], types: dict) -> dict:
    sheets, pk = [], pk0
    for order, (sheet, items) in enumerate(d["sheets"], 1):
        placed = []
        for v in items:
            pk += 1
            visuals.append(visual_record(v, pk, sheet, types[v["ds"]], ds_pk[v["ds"]], d))
            placed.append((pk, v["pos"]))
        sheets.append({"sheet_id": order, "order": order, "sheet_handle_title": sheet, "behaviors": {},
                       "visual_widgets": [{"col": c, "row": r, "size_x": w, "size_y": h, "id": f"uri-{i}-widget-{p}"}
                                          for i, (p, (c, r, w, h)) in enumerate(placed, 1)],
                       "control_widgets": []})
    dash = {"report_title": d["title"], "numColumns": 64, "report_subtitle": d["subtitle"],
            "dashboard_widgets": sheets[0]["visual_widgets"], "dashboard_sheets": sheets,
            "user_settings": {"dashboard_width": "1280", "display_filters": "true",
                              "permit_csv_download_dashboard": "true"},
            "global_control_widgets": [], "control_widgets": [], "click_behavior": {}}
    return {"model": "reports.report", "pk": d["pk"], "fields": {
        "report_name": d["title"], "report_description": "dataviz/build_dashboard.py",
        "dataset": ds_pk[d["main_ds"]], "workspace": 1, "report_type": "dashboard", "report_mode": None,
        "dashboard_url_name": "", "report_data": json.dumps(dash), "shared_visual_dashboards": "[]",
        "parent_report": None, "uuid": uid("dashboard", d["key"]), "imported_uuid": None, "has_css_styles": False,
        "report_search_text": None}}


def missing_columns(types: dict[str, dict[str, str]]) -> list[str]:
    out = []
    for d in DASHBOARDS:
        for sheet, v in visuals_of(d):
            exprs = [e for e, _ in v["measures"]] + v.get("filters", [])
            cols = {c for c, _ in v.get("dims", []) + v.get("x", []) + v.get("color", [])}
            cols |= {c for e in exprs for c in re.findall(r"\[(\w+)\]", e)}
            out += [f"{d['title']} / {sheet} / {v['title']}: {c}" for c in sorted(cols - set(types[v["ds"]]))]
    return out


def build(types: dict[str, dict[str, str]], conn_id: int, version: dict) -> dict:
    missing = missing_columns(types)
    if missing:
        raise SystemExit("columns not in the dataset's view:\n  " + "\n  ".join(missing))
    ds_pk = {k: DATASET_PK0 + i for i, k in enumerate(DATASETS)}
    visuals, dashboards, pk0 = [], [], VISUAL_PK0
    for d in DASHBOARDS:
        dashboards.append(dashboard_record(d, pk0, visuals, ds_pk, types))
        pk0 += 100
    used_by = {k: [d["pk"] for d in DASHBOARDS if any(v["ds"] == k for _, v in visuals_of(d))] for k in DATASETS}
    return {"segments": [], "staticasset": [], "dashboards": dashboards, "appgroupmembership": [],
            "reportannotation": [], "events": [], "customcss": [], "reportimage": [], "dateranges": [],
            "visuals": visuals, "colorpalette": [], "appgroups": [],
            "datasets": [dataset_record(k, ds_pk[k], types[k], conn_id, used_by[k]) for k in DATASETS],
            "version": version}


def _strip(e: str) -> str:
    return re.sub(r"\[(\w+)\]", r"\1", e)


def tile_sql(v: dict) -> str:
    """A KPI tile's number as plain Impala SQL on its view ([col] -> col)."""
    where = " AND ".join(_strip(f) for f in v.get("filters", [])) or "TRUE"
    return f"SELECT {_strip(v['measures'][0][0])} FROM {DB}.{DATASETS[v['ds']][1]} WHERE {where}"


def visual_sql(v: dict) -> str:
    """The query a visual sends, as plain Impala SQL."""
    dims = [c for c, _ in v.get("dims", []) + v.get("x", []) + v.get("color", [])]
    where = " AND ".join(_strip(f) for f in v.get("filters", [])) or "TRUE"
    group = f" GROUP BY {', '.join(dims)}" if dims else ""
    return (f"SELECT {', '.join(dims + [_strip(e) for e, _ in v['measures']])} FROM {DB}.{DATASETS[v['ds']][1]} "
            f"WHERE {where}{group} LIMIT {v.get('limit', 1000)}")


def check(engine) -> int:
    """Every visual's query straight on Impala: the expressions parse and the visual has rows."""
    failed = 0
    for d in DASHBOARDS:
        for sheet, v in visuals_of(d):
            try:
                rows = engine.query(visual_sql(v))[1]
                ok, note = bool(rows) or v.get("may_be_empty", False), f"{len(rows)} rows"
                if rows and v["type"] == "kpi":
                    note = f"{rows[0][-1]}"
            except Exception as e:  # noqa: BLE001
                ok, note = False, str(e).strip().splitlines()[-1][:200]
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {d['title']} / {sheet} / {v['title']}: {note}", flush=True)
    return failed


class DataViz:
    """The CDW Data Visualization instance, authenticated with a Data Visualization API key."""

    def __init__(self):
        import requests

        key = os.environ.get("CXR_VIZ_API_KEY")
        if not key:
            raise SystemExit("set CXR_VIZ_API_KEY (Data Visualization: Site Administration -> Manage API Keys), "
                             "or import dataviz/cxr_dashboards.json in the UI")
        self.url = json.loads((ROOT / "config" / "lakehouse.json").read_text())["dataviz_url"].rstrip("/")
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"apikey {key}"

    def get(self, path: str, **params):
        r = self.s.get(self.url + path, params=params, timeout=60)
        if r.status_code != 200:
            raise SystemExit(f"GET {path}: HTTP {r.status_code} {r.text[:200]}")
        return r.json()

    def connections(self) -> list[dict]:
        return self.get("/arc/adminapi/v1/connections")

    def connection_id(self, name: str) -> int:
        found = next((c for c in self.connections() if c["name"] == name), None)
        if found is None:
            raise SystemExit(f"no connection {name!r}; --list-connections shows them")
        return found["id"]

    def version(self) -> dict:
        return self.get("/arc/migration/api/export/", dashboards="[]", filename="version", dry_run="False")["version"]

    def import_file(self, path: Path, connection: str) -> None:
        with path.open("rb") as f:
            r = self.s.post(self.url + "/arc/migration/api/import/", files={"import_file": f},
                            data={"dry_run": "False", "dataconnection_name": connection}, timeout=300)
        print(f"import: HTTP {r.status_code} {r.text[:500]}")
        if r.status_code != 200:
            raise SystemExit(1)

    def verify(self, engine) -> int:
        """Every visual's query through the Data API (Data Visualization -> its connection -> Impala);
        each KPI tile's number is also computed directly in Impala and compared."""
        ids = {d["name"]: d["id"] for d in self.get("/arc/adminapi/v1/datasets")}
        failed = 0
        for d in DASHBOARDS:
            for sheet, v in visuals_of(d):
                failed += not self.verify_visual(v, f"{d['title']} / {sheet}", ids, engine)
        return failed

    def verify_visual(self, v: dict, where: str, ids: dict, engine) -> bool:
        dims = v.get("dims", []) + v.get("x", []) + v.get("color", [])
        dsreq = {"version": 1, "type": "SQL", "limit": v.get("limit", 1000),
                 "dimensions": [{"type": "SIMPLE", "expr": f"[{c}] as '{a}'"} for c, a in dims],
                 "aggregates": [{"expr": f"{e} as '{a}'"} for e, a in v["measures"]],
                 "filters": v.get("filters", []), "dataset_id": ids[DATASETS[v["ds"]][0]]}
        r = self.s.post(self.url + "/arc/api/data", data={"version": 1, "dsreq": json.dumps(dsreq)}, timeout=300)
        rows = json.loads(r.json()["rows"]) if r.status_code == 200 else None
        ok = bool(rows) or (rows is not None and v.get("may_be_empty", False))
        note = f"{len(rows)} rows" if rows is not None else f"HTTP {r.status_code} {r.text[:200]}"
        if ok and v["type"] == "kpi":
            want = engine.query(tile_sql(v))[1][0][0]
            got = rows[0][-1] if isinstance(rows[0], list) else list(rows[0].values())[-1]
            ok = abs(float(got) - float(want)) < 1e-6
            note = f"{got} (Impala {want})"
        print(f"{'ok  ' if ok else 'FAIL'} {where} / {v['title']}: {note}")
        return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--import", dest="do_import", action="store_true", help="import the file (needs CXR_VIZ_API_KEY)")
    p.add_argument("--connection", help="Data Visualization connection to the Impala warehouse (with --import)")
    p.add_argument("--verify", action="store_true", help="run every visual's query through the Data API")
    p.add_argument("--check", action="store_true", help="run every visual's query directly on Impala")
    p.add_argument("--list-connections", action="store_true")
    args = p.parse_args()
    if args.list_connections:
        for c in DataViz().connections():
            print(c["id"], c["name"], c.get("type"))
        return 0
    engine = impala()
    if args.check:
        return 1 if check(engine) else 0
    if args.verify:
        return 1 if DataViz().verify(engine) else 0
    viz = DataViz() if args.do_import else None
    if viz and not args.connection:
        p.error("--import needs --connection (see --list-connections)")
    types = {k: column_types(engine, k) for k in DATASETS}
    conn_id = viz.connection_id(args.connection) if viz else 1
    doc = build(types, conn_id, viz.version() if viz else DEFAULT_VERSION)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(doc['dashboards'])} dashboards, {len(doc['datasets'])} datasets, "
          f"{len(doc['visuals'])} visuals, {sum(len(d['sheets']) for d in DASHBOARDS)} sheets")
    if viz:
        viz.import_file(OUT, args.connection)
        print(f"open {viz.url}/arc/apps/ -> Dashboards -> " + " / ".join(d["title"] for d in DASHBOARDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
