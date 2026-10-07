#!/usr/bin/env python3
"""
Run the lakehouse on a laptop (or in CI) against local Spark + Iceberg, with the landing zone
in a local folder. The CDE job files and the CAI scoring code are the ones the platform runs;
only the Spark session and the store differ.

  python scripts/run_local.py all                        # every demo date, every stage
  python scripts/run_local.py land bronze silver gold --dates 2026-10-01
  python scripts/run_local.py score outcomes --scorer champion   # models/champion + data/raw films
  python scripts/run_local.py semantic                   # sql/semantic/*.sql as Spark views
  python scripts/run_local.py history rsingh_cxr_gold.fact_study
  python scripts/run_local.py sql "SELECT * FROM rsingh_cxr_gold.daily_triage_summary"

Stages run per date in pipeline order (a date completes before the next starts), as the DAG
does. The default scorer is a STUB that derives noisy probabilities from the reference labels
for every head (pneumonia champion, pneumothorax silent trial, film_qc; model_version
*-stub-local): it exercises the tables and the dashboards, not the models.
--scorer champion uses serve.predict with the installed models and the films in data/raw/.
Arguments after `--` go to every CDE job.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "cde" / "jobs"
sys.path.insert(0, str(JOBS_DIR))
sys.path.insert(0, str(ROOT))
import cxr_common as C  # noqa: E402

ICEBERG_PACKAGE = os.environ.get("CXR_ICEBERG_PACKAGE", "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.0")
SQLITE_PACKAGE = "org.xerial:sqlite-jdbc:3.46.1.3"
STAGES = {"land": "land_sources.py", "bronze": "ingest_bronze.py", "silver": "build_silver.py",
          "gold": "build_gold.py", "score": None, "outcomes": "build_outcomes.py", "semantic": None}
CFG = C.config()
STUB_META = {"feature_version": "stub", "git_sha": "local00", "threshold": 0.5, "p1_probability": 0.9}


def local_spark(warehouse: Path, driver_memory: str = "4g"):
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(JOBS_DIR), os.environ.get("PYTHONPATH", "")])
    warehouse.mkdir(parents=True, exist_ok=True)
    # Iceberg JDBC catalog on SQLite: unlike the Hadoop catalog it stores the semantic views
    spark = (SparkSession.builder.appName("cxr-local").master("local[4]")
             .config("spark.driver.host", "127.0.0.1").config("spark.driver.bindAddress", "127.0.0.1")
             .config("spark.jars.packages", f"{ICEBERG_PACKAGE},{SQLITE_PACKAGE}")
             .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
             .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
             .config("spark.sql.catalog.local.warehouse", str(warehouse))
             .config("spark.sql.catalog.local.type", "jdbc")
             .config("spark.sql.catalog.local.uri", f"jdbc:sqlite:{warehouse.resolve() / 'catalog.db'}")
             .config("spark.sql.catalog.local.jdbc.schema-version", "V1")
             .config("spark.sql.defaultCatalog", "local")
             .config("spark.driver.memory", driver_memory)
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.default.parallelism", "4")
             .config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    C.configure(spark)
    return spark


def load_job(filename: str):
    """Load a job module without registering it in sys.modules, so its closures pickle by value."""
    spec = importlib.util.spec_from_file_location(filename[:-3], JOBS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def stub_scorer():
    """Deterministic per film, in serve.predict's result shape: pneumonia (champion, children),
    pneumothorax (silent trial, adults) and film_qc (champion: planted films are unfit). A
    film whose finding a head detects mostly scores high on that head, every other film low."""
    from common import in_scope
    from evaluate.metrics import priority_band

    heads = {"pneumonia": ("PNEUMONIA", "pediatric", "champion"),
             "pneumothorax": ("PNEUMOTHORAX", "adult", "silent_trial")}

    def head(name, label, age, key):
        finding, population, stage = heads[name]
        rng = random.Random(f"stub-{name}-{key}")
        prob = min(0.999, max(0.001, rng.betavariate(6, 1.2) if label == finding else rng.betavariate(1.3, 5)))
        scope = in_scope(population, age)
        return {"probability": round(prob, 4), "threshold": STUB_META["threshold"], "positive": prob >= 0.5,
                "priority": priority_band(prob, STUB_META["threshold"], STUB_META["p1_probability"]) if scope else None,
                "in_scope": scope, "stage": stage, "model_version": f"{name}-stub-local"}

    def score(paths, ages):
        out = []
        for p, age in zip(paths, ages):
            unfit = p.parent.name == "UNSUITABLE"
            findings = {"pneumonia": head("pneumonia", p.parent.name, age, p.name)}
            trial = {"pneumothorax": head("pneumothorax", p.parent.name, age, p.name)}
            live = findings["pneumonia"]
            band, by = ("NA", None) if unfit or not live["in_scope"] else (live["priority"], "pneumonia")
            out.append({"priority": band, "triage_model": by, "probability_pneumonia": live["probability"],
                        "threshold": STUB_META["threshold"], "quality_flags": [], "findings": findings,
                        "silent_trial": trial, "film_qc": {"probability": 0.98 if unfit else 0.02,
                                                           "unsuitable": unfit, "model_version": "film_qc-stub-local"}})
        return out, []

    pools = C.pools(CFG)
    films = {f["image_file"]: Path("stub") / f["label"] / f["image_file"] for pool in pools.values() for f in pool}
    for d in CFG.get("planted_films", {}):
        for s in C.studies_of_day(CFG, date.fromisoformat(d), pools):
            if s["planted"]:
                films[s["image_file"]] = Path("stub") / "UNSUITABLE" / s["image_file"]
    return (score, STUB_META), films


def run_score(spark, d: str, prefix: str, scorer: str):
    from lakehouse.score_studies import score_date
    from lakehouse.store import SparkStore

    store = SparkStore(spark, prefix)
    if scorer == "champion":
        return score_date(store, date.fromisoformat(d), "run_local")
    (fn, meta), films = stub_scorer()
    return score_date(store, date.fromisoformat(d), "run_local", scorer=(fn, meta),
                      drift_fn=lambda q: ({}, "UNAVAILABLE"), films=films)


def main() -> int:
    if "--" in sys.argv:
        i = sys.argv.index("--")
        argv, extra = sys.argv[1:i], sys.argv[i + 1:]
    else:
        argv, extra = sys.argv[1:], []
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stages", nargs="+")
    p.add_argument("--dates", default="all", help="comma-separated business dates, or all (config demo_dates)")
    p.add_argument("--landing", default=str(ROOT / "data" / "lake" / "landing"))
    p.add_argument("--warehouse", default=str(ROOT / "data" / "lake" / "warehouse"))
    p.add_argument("--db-prefix", default=CFG["db_prefix"])
    p.add_argument("--scorer", choices=["stub", "champion"], default="stub")
    args = p.parse_args(argv)
    dates = CFG["demo_dates"] if args.dates == "all" else args.dates.split(",")
    stages = args.stages

    if stages[0] in ("history", "sql"):
        spark = local_spark(Path(args.warehouse))
        if stages[0] == "history":
            spark.sql(f"SELECT committed_at, snapshot_id, operation, summary['added-records'] AS added, "
                      f"summary['deleted-records'] AS deleted, summary['total-records'] AS total "
                      f"FROM {stages[1]}.snapshots ORDER BY committed_at").show(100, truncate=False)
        else:
            spark.sql(" ".join(stages[1:])).show(200, truncate=False)
        return 0
    if stages == ["all"]:
        stages = list(STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        p.error(f"unknown stages {unknown}; choose from {list(STAGES)}")

    spark = local_spark(Path(args.warehouse))
    common = ["--landing", args.landing, "--db-prefix", args.db_prefix]
    per_date = [s for s in STAGES if s in stages and s != "semantic"]
    for d in dates:
        for stage in per_date:
            print(f"\n=== {stage} {d}", flush=True)
            if stage == "score":
                run_score(spark, d, args.db_prefix, args.scorer)
            else:
                load_job(STAGES[stage]).run(spark, ["--business-date", d, *common, *extra])
    if "semantic" in stages:
        import run_semantic

        print("\n=== semantic", flush=True)
        return run_semantic.run(run_semantic.SparkEngine(spark, args.db_prefix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
