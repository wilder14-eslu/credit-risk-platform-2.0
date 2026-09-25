"""Prueba end-to-end local de los jobs de Databricks (sin Databricks).

Sustituye Spark/Delta por DuckDB y Unity Catalog por un registro MLflow en SQLite,
y ejecuta los MISMOS scripts de `jobs/` (entrenamiento, validación, replay,
monitoreo y A/B) para verificar el ciclo completo antes de desplegar.

Uso:
    python tests/e2e/run_pipeline_locally.py                  # datos sintéticos
    python tests/e2e/run_pipeline_locally.py --csv data/accepted_2007_to_2018Q4.csv --rows 300000
Requiere: pip install duckdb (además de requirements-dev.txt).
"""

from __future__ import annotations

import argparse
import os
import re
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import duckdb  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import UCNames, platform_config  # noqa: E402
from credit_risk.data.lending_club import parse_raw  # noqa: E402
from credit_risk.data.quality import assert_quality, clean, run_expectations  # noqa: E402
from credit_risk.data.synthetic import generate_raw  # noqa: E402
from credit_risk.features.engineering import DERIVED_FEATURES, build_features  # noqa: E402
from credit_risk.registry import mlflow_registry as R  # noqa: E402
from credit_risk.registry import serving_endpoint as S  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="credit-risk-e2e-"))
CON = duckdb.connect()
TABLES: set[str] = set()
TASK_VALUES: dict = {}


def _t(name: str) -> str:
    return name.replace(".", "__")


def _sql(q: str) -> str:
    return re.sub(r"\b(\w+)\.(\w+)\.(\w+)\b", lambda m: _t(m.group(0)) if m.group(1) == "workspace" else m.group(0), q)


def read_pandas(q: str) -> pd.DataFrame:
    return CON.execute(_sql(q) if " " in q.strip() else f"SELECT * FROM {_t(q)}").df()


def write_pandas(frame: pd.DataFrame, name: str, mode: str = "append") -> None:
    if frame.empty:
        return
    frame = frame.copy()
    if mode == "append" and name in TABLES:  # igual que en Databricks: solo columnas de la tabla
        target = [r[0] for r in CON.execute(f"DESCRIBE {_t(name)}").fetchall()]
        frame = frame[[c for c in frame.columns if c in target]]
    for c in frame.columns:
        if frame[c].dtype == object and frame[c].isna().all():
            frame[c] = frame[c].astype("string")
    CON.register("tmpdf", frame)
    if mode == "overwrite" or name not in TABLES:
        CON.execute(f"CREATE OR REPLACE TABLE {_t(name)} AS SELECT * FROM tmpdf")
    else:
        CON.execute(f"INSERT INTO {_t(name)} BY NAME SELECT * FROM tmpdf")
    CON.unregister("tmpdf")
    TABLES.add(name)


def patch_platform(names: UCNames) -> None:
    lh.read_pandas, lh.write_pandas = read_pandas, write_pandas
    lh.set_task_value = TASK_VALUES.__setitem__
    lh.grant_app_access = lambda *a, **k: None
    uri = f"sqlite:///{TMP / 'mlflow.db'}"

    def setup_mlflow(experiment=None):
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        mlflow.set_experiment("e2e")

    R.setup_mlflow = setup_mlflow
    S.deploy = lambda n, champ, chall=None, *a, **k: print(f"   [Model Serving] champion=v{champ} challenger={chall}")
    S.grant_query_to_apps = lambda *a, **k: []
    UCNames.volume_path = property(lambda self: str(TMP / "volume"))
    for table, ddl in lh.DDL.items():  # mismas tablas que crea ensure_objects en Databricks
        stmt = re.sub(r"COMMENT '[^']*'", "", ddl.format(t=_t(names.fq(table)))).replace("STRING", "VARCHAR")
        CON.execute(stmt)
        TABLES.add(names.fq(table))


def build_gold(raw: pd.DataFrame, names: UCNames) -> None:
    """Equivalente pandas de los jobs 01-02 (que en Databricks corren con Spark)."""
    canonical = parse_raw(raw)
    assert_quality(run_expectations(canonical))
    silver = clean(canonical)
    derived = build_features(silver)[list(DERIVED_FEATURES)]
    write_pandas(
        pd.concat([silver.reset_index(drop=True), derived.reset_index(drop=True)], axis=1),
        names.feature_table,
        "overwrite",
    )
    print(f"   gold: {len(silver):,} préstamos")


def run(script: str, *args: str) -> None:
    print(f"\n== {script} {' '.join(args)}")
    sys.argv = [script, "--project-root", str(ROOT), "--catalog", "workspace", "--schema", "credit_risk", *args]
    runpy.run_path(str(ROOT / "jobs" / script), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV de Lending Club (si no, datos sintéticos)")
    parser.add_argument("--rows", type=int, default=60000)
    parser.add_argument("--min-auc", type=float, default=None, help="relaja el gate si usas una muestra chica")
    opts = parser.parse_args()
    names = UCNames("workspace", "credit_risk")
    patch_platform(names)
    platform_config()["replay"]["max_rows_per_month"] = 800
    platform_config()["monitoring"]["min_rows"] = 150
    if opts.min_auc is not None:
        platform_config()["quality_gates"].update(min_test_roc_auc=opts.min_auc, max_overfit_gap=0.5)

    raw = pd.read_csv(opts.csv, dtype=str, nrows=opts.rows) if opts.csv else generate_raw(opts.rows, seed=1)
    print("== 01-02 ingest + calidad + feature table")
    build_gold(raw, names)
    run("03_train_register.py", "--optuna-trials", "0")
    run("04_validate_and_deploy.py")
    for _ in range(4):  # 2014-01 a 2015-12 en bloques de 6 meses
        run("05_replay_production.py", "--months", "6")
        run("06_monitor.py")
        run("07_ab_evaluate.py")
        if TASK_VALUES.get("retrain") == "true":
            print(f"   >> monitoreo pide Continuous Training (as_of={TASK_VALUES['clock']})")
            run(
                "03_train_register.py",
                "--optuna-trials",
                "0",
                "--trigger",
                "monitoring",
                "--as-of",
                TASK_VALUES["clock"],
            )
            run("04_validate_and_deploy.py")

    print("\n== Resumen del monitoreo")
    print(
        read_pandas(
            "SELECT clock_month, model_version, n_rows, n_labeled, features_drifted, round(roc_auc,3) auc, "
            "round(auc_drop,3) auc_drop, severity, retrain FROM workspace.credit_risk.monitoring_metrics ORDER BY clock_month"
        ).to_string(index=False)
    )
    if names.ab_results in TABLES:
        print(
            read_pandas(
                "SELECT clock_month, champion_version, challenger_version, decision, reason "
                "FROM workspace.credit_risk.ab_test_results"
            ).to_string(index=False)
        )
    print(f"\nOK: ciclo completo ejecutado. Artefactos en {TMP}")


if __name__ == "__main__":
    main()
