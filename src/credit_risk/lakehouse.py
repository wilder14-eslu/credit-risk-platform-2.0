"""Utilidades de Lakehouse para los jobs serverless (Spark, Delta, task values, permisos).

pyspark se importa de forma perezosa: el resto del paquete se prueba en CI sin Spark.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from credit_risk.config import UCNames, categorical_features, numeric_features

logger = logging.getLogger("credit_risk")

_FEATURE_DDL = ",\n  ".join(
    [f"{c} DOUBLE" for c in numeric_features()] + [f"{c} STRING" for c in categorical_features()]
)

DDL: dict[str, str] = {
    "inference_log": f"""
CREATE TABLE IF NOT EXISTS {{t}} (
  request_id STRING NOT NULL,
  loan_id STRING,
  event_ts TIMESTAMP COMMENT 'Mes de originación (reloj simulado en el replay) o hora real (API)',
  source STRING COMMENT 'replay | api | dashboard',
  variant STRING COMMENT 'champion | challenger',
  model_version STRING,
  probability DOUBLE,
  decision STRING,
  risk_band STRING,
  latency_ms DOUBLE,
  {_FEATURE_DDL}
) COMMENT 'Log de inferencias (payload + predicción) para monitoreo y A/B testing'
""",
    "outcomes": """
CREATE TABLE IF NOT EXISTS {t} (
  loan_id STRING NOT NULL,
  request_id STRING,
  actual_default INT,
  observed_ts TIMESTAMP COMMENT 'Cuando se conoce el desenlace (originación + retraso)',
  source STRING
) COMMENT 'Desenlaces reales (ground truth) que llegan con retraso'
""",
    "reference_profile": """
CREATE TABLE IF NOT EXISTS {t} (
  model_version STRING, created_ts TIMESTAMP, profile_json STRING, reference_metrics_json STRING
) COMMENT 'Distribución de entrenamiento por versión de modelo'
""",
    "monitoring_metrics": """
CREATE TABLE IF NOT EXISTS {t} (
  run_id STRING, run_ts TIMESTAMP, clock_month TIMESTAMP, model_version STRING, n_rows BIGINT,
  n_labeled BIGINT, features_drifted INT, prediction_psi DOUBLE, roc_auc DOUBLE, ks DOUBLE,
  brier DOUBLE, auc_drop DOUBLE, default_rate_observed DOUBLE, default_rate_predicted DOUBLE,
  ddm_state STRING, page_hinkley_statistic DOUBLE, concept_drift BOOLEAN, severity STRING,
  retrain BOOLEAN, reasons STRING
)
""",
    "drift_by_feature": """
CREATE TABLE IF NOT EXISTS {t} (
  run_id STRING, run_ts TIMESTAMP, clock_month TIMESTAMP, model_version STRING, feature STRING,
  kind STRING, psi DOUBLE, ks_statistic DOUBLE, ks_pvalue DOUBLE, ks_drift BOOLEAN,
  null_rate_ref DOUBLE, null_rate_cur DOUBLE, mean_ref DOUBLE, mean_cur DOUBLE, status STRING
)
""",
    "ab_test_results": """
CREATE TABLE IF NOT EXISTS {t} (
  run_ts TIMESTAMP, clock_month TIMESTAMP, champion_version STRING, challenger_version STRING,
  decision STRING, reason STRING, n_champion BIGINT, n_challenger BIGINT, auc_champion DOUBLE,
  auc_challenger DOUBLE, auc_diff DOUBLE, ci_low DOUBLE, ci_high DOUBLE, p_value DOUBLE,
  bad_rate_approved_champion DOUBLE, bad_rate_approved_challenger DOUBLE, bad_rate_pvalue DOUBLE,
  months_running DOUBLE
)
""",
    "model_benchmark": """
CREATE TABLE IF NOT EXISTS {t} (
  run_ts TIMESTAMP, algorithm STRING, test_roc_auc DOUBLE, test_pr_auc DOUBLE, test_ks DOUBLE,
  test_brier DOUBLE, train_roc_auc DOUBLE, latency_ms DOUBLE, fit_diagnosis STRING,
  selected BOOLEAN, registered_version STRING
)
""",
    "retrain_events": """
CREATE TABLE IF NOT EXISTS {t} (
  event_ts TIMESTAMP, clock_month TIMESTAMP, trigger STRING, reasons STRING, model_version STRING
)
""",
    "data_quality_log": """
CREATE TABLE IF NOT EXISTS {t} (
  run_ts TIMESTAMP, layer STRING, name STRING, column_name STRING, passed BOOLEAN,
  observed DOUBLE, threshold DOUBLE, severity STRING
)
""",
}


def job_args(extra: dict[str, Any] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema", default="credit_risk")
    parser.add_argument("--project-root", default=None)
    for name, default in (extra or {}).items():
        parser.add_argument(f"--{name}", default=default)
    args, _ = parser.parse_known_args()
    return args


def names_from(args: argparse.Namespace) -> UCNames:
    return UCNames(catalog=args.catalog, schema=args.schema)


def spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def ensure_objects(names: UCNames) -> None:
    s = spark()
    s.sql(f"CREATE SCHEMA IF NOT EXISTS {names.catalog}.{names.schema}")
    for volume in ("raw", "artifacts"):
        s.sql(f"CREATE VOLUME IF NOT EXISTS {names.catalog}.{names.schema}.{volume}")
    for table, ddl in DDL.items():
        s.sql(ddl.format(t=names.fq(table)))


def table_exists(fq_name: str) -> bool:
    return bool(spark().catalog.tableExists(fq_name))


def read_pandas(query_or_table: str) -> pd.DataFrame:
    s = spark()
    df = s.sql(query_or_table) if " " in query_or_table.strip() else s.table(query_or_table)
    return df.toPandas()


def write_pandas(frame: pd.DataFrame, fq_name: str, mode: str = "append") -> None:
    if frame.empty:
        logger.info("Nada que escribir en %s", fq_name)
        return
    s = spark()
    sdf = s.createDataFrame(frame)
    if mode == "append" and table_exists(fq_name):
        target = s.table(fq_name).columns
        sdf = sdf.select(*[c for c in target if c in sdf.columns])
    sdf.write.mode(mode).option("mergeSchema", "true").saveAsTable(fq_name)
    logger.info("%s filas -> %s (%s)", len(frame), fq_name, mode)


def grant_app_access(names: UCNames, principals: list[str]) -> None:
    """Permite a los service principals de las apps leer y escribir las tablas del esquema."""
    s = spark()
    for principal in principals:
        for stmt in (
            f"GRANT USE CATALOG ON CATALOG {names.catalog} TO `{principal}`",
            f"GRANT USE SCHEMA, SELECT, MODIFY ON SCHEMA {names.catalog}.{names.schema} TO `{principal}`",
        ):
            try:
                s.sql(stmt)
            except Exception as exc:  # Free Edition o permisos insuficientes: se documenta
                logger.warning("No se pudo ejecutar '%s': %s", stmt, exc)


def set_task_value(key: str, value: Any) -> None:
    try:
        from databricks.sdk.runtime import dbutils

        dbutils.jobs.taskValues.set(key=key, value=value)
    except Exception as exc:  # ejecución local
        logger.info("task value %s=%s (sin dbutils: %s)", key, value, exc)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
