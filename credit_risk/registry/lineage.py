"""Trazabilidad de una versión del modelo: código, datos y cortes temporales con los que se entrenó.

Es la contraparte de "data lineage / data versioning" del cap. 10 de *Practical Machine Learning on
Databricks*: dado un modelo en producción se debe poder reconstruir (1) el commit de código, (2) la versión
Delta exacta del feature table y (3) los cortes del split. Función pura, se prueba sin MLflow ni Spark.
"""

from __future__ import annotations

from collections.abc import Mapping

REQUIRED_LINEAGE_TAGS = (
    "git_sha",
    "gold_table",
    "gold_delta_version",
    "data_source",
    "train_end",
    "validation_end",
    "test_end",
)
UNRESOLVED = "unknown"


def missing_lineage(tags: Mapping[str, str] | None) -> list[str]:
    """Tags de trazabilidad ausentes, vacíos o sin resolver (p. ej. versión Delta `unknown`)."""
    tags = tags or {}
    problems = [f"falta el tag '{k}'" for k in REQUIRED_LINEAGE_TAGS if not str(tags.get(k, "")).strip()]
    if str(tags.get("gold_delta_version", "")).strip() == UNRESOLVED:
        problems.append("'gold_delta_version' sin resolver (DESCRIBE HISTORY falló)")
    return problems
