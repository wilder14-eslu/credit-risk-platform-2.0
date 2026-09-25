"""Job 1 - Setup de Unity Catalog + ingesta Bronze.

Crea esquema, volúmenes y tablas operativas (idempotente) y carga el CSV crudo
de Lending Club desde el Volume `raw` a la capa Bronze, conservando solo las
columnas conocidas al momento de la solicitud (el leakage nunca entra).
Con `--source synthetic` genera datos con el mismo formato (modo demo / CI).
"""

# Bootstrap: agrega <raíz del bundle>/src al path (el bundle pasa --project-root).
import os
import sys

_root = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--project-root"), None)
if _root is None:
    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        _root = os.getcwd()
sys.path.insert(0, os.path.join(_root, "src"))
os.environ.setdefault("CREDIT_RISK_CONFIG_DIR", os.path.join(_root, "config"))

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import categorical_features, data_schema, numeric_features  # noqa: E402


def bronze_columns() -> list[str]:
    schema = data_schema()
    cols = {schema["id_source"], schema["date_source"], schema["target_source"], "fico_range_high"}
    cols |= {spec["source"] for spec in numeric_features().values()}
    cols |= {spec["source"] for spec in categorical_features().values()}
    return sorted(cols)


def main() -> None:
    lh.configure_logging()
    args = lh.job_args({"source": "real", "synthetic-rows": "150000"})
    names = lh.names_from(args)
    lh.ensure_objects(names)
    s = lh.spark()
    from pyspark.sql import functions as F

    if args.source == "synthetic":
        from credit_risk.data.synthetic import generate_raw

        raw = s.createDataFrame(generate_raw(int(args.synthetic_rows)).astype(str))
        origin = "synthetic"
    else:
        path = f"{names.volume_path}/raw/{data_schema()['source_file']}"
        try:
            raw = s.read.option("header", "true").option("multiLine", "true").option("escape", '"').csv(path)
        except Exception as exc:
            raise FileNotFoundError(
                f"No encontré {path}. Sube accepted_2007_to_2018Q4.csv al Volume "
                "'raw' (ver docs/DATA.md) o ejecuta el job con source=synthetic."
            ) from exc
        origin = path

    cols = [c for c in bronze_columns() if c in raw.columns]
    bronze = (
        raw.select(*[F.col(c).cast("string") for c in cols])
        .withColumn("ingested_ts", F.current_timestamp())
        .withColumn("source_file", F.lit(origin))
    )
    bronze.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(names.bronze)
    lh.set_task_value("bronze_rows", int(s.table(names.bronze).count()))


if __name__ == "__main__":
    main()
