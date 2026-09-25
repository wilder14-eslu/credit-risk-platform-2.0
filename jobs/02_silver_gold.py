"""Job 2 - Bronze -> Silver (parseo + calidad) -> Gold (feature table en Unity Catalog).

- El parser pandas (`credit_risk.data.lending_club.parse_raw`) se aplica por lotes con
  `mapInPandas`: el mismo código probado en CI transforma los ~2.2 M préstamos.
- Las expectativas de calidad se registran en `data_quality_log`; un error bloquea el pipeline.
- Gold: feature table con PK `loan_id` (feature store offline) + variables derivadas.
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

import pandas as pd  # noqa: E402

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import date_column, id_column  # noqa: E402
from credit_risk.data.lending_club import parse_batches, spark_schema_ddl  # noqa: E402
from credit_risk.data.quality import assert_quality, clean, run_expectations  # noqa: E402
from credit_risk.features.engineering import DERIVED_FEATURES, build_features  # noqa: E402


def clean_batches(batches):
    for batch in batches:
        yield clean(batch)


def gold_batches(batches):
    for batch in batches:
        derived = build_features(batch)[list(DERIVED_FEATURES)]
        yield pd.concat([batch.reset_index(drop=True), derived.reset_index(drop=True)], axis=1)


def main() -> None:
    lh.configure_logging()
    args = lh.job_args({"quality-sample": "300000"})
    names = lh.names_from(args)
    s = lh.spark()
    from pyspark.sql import functions as F

    ddl = spark_schema_ddl()
    parsed = s.table(names.bronze).drop("ingested_ts", "source_file").mapInPandas(parse_batches, ddl)
    parsed = parsed.dropDuplicates([id_column()])

    total = parsed.count()
    fraction = min(1.0, int(args.quality_sample) / max(total, 1))
    sample = parsed.sample(fraction=fraction, seed=42).toPandas()
    results = run_expectations(sample, min_rows=min(1000, total))
    log = pd.DataFrame([r.to_dict() for r in results]).rename(columns={"column": "column_name"})
    log["run_ts"], log["layer"] = lh.utcnow(), "bronze->silver"
    lh.write_pandas(log, names.data_quality_log)
    assert_quality(results)

    silver = parsed.mapInPandas(clean_batches, ddl)
    silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(names.silver)

    gold_ddl = ddl + ", " + ", ".join(f"{c} DOUBLE" for c in DERIVED_FEATURES)
    gold = s.table(names.silver).mapInPandas(gold_batches, gold_ddl).withColumn("feature_ts", F.current_timestamp())
    gold.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(names.feature_table)
    s.sql(f"ALTER TABLE {names.feature_table} ALTER COLUMN {id_column()} SET NOT NULL")
    try:
        s.sql(f"ALTER TABLE {names.feature_table} ADD CONSTRAINT gold_loan_features_pk PRIMARY KEY ({id_column()})")
    except Exception as exc:  # ya existe en re-ejecuciones
        lh.logger.info("PK ya definida: %s", exc)
    s.sql(
        f"COMMENT ON TABLE {names.feature_table} IS "
        "'Feature table Lending Club (PK loan_id): variables de solicitud + derivadas + target'"
    )

    stats = (
        s.table(names.feature_table)
        .groupBy(F.year(date_column()).alias("year"))
        .agg(F.count("*").alias("loans"), F.avg("target_default").alias("default_rate"))
        .orderBy("year")
        .toPandas()
    )
    lh.logger.info("Préstamos por año:\n%s", stats.to_string(index=False))
    lh.set_task_value("gold_rows", int(s.table(names.feature_table).count()))


if __name__ == "__main__":
    main()
