"""Expectativas de calidad de datos sobre el formato canónico (gate Bronze -> Silver)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd

from credit_risk.config import (
    categorical_features,
    data_schema,
    date_column,
    id_column,
    numeric_features,
    target_name,
)

logger = logging.getLogger(__name__)

# Nulos esperables en Lending Club (campos opcionales o agregados tarde al formulario)
_MAX_NULL = {
    "emp_length_years": 0.12,
    "mort_acc": 0.10,
    "pub_rec_bankruptcies": 0.05,
    "revol_util": 0.02,
    "dti": 0.02,
    "inq_last_6mths": 0.02,
}


@dataclass
class Expectation:
    name: str
    column: str
    passed: bool
    observed: float
    threshold: float
    severity: str  # "error" bloquea el pipeline; "warning" solo se registra

    def to_dict(self) -> dict:
        return asdict(self)


def run_expectations(data: pd.DataFrame, min_rows: int = 1000) -> list[Expectation]:
    results: list[Expectation] = []
    n = max(len(data), 1)
    results.append(Expectation("row_count_min", "*", len(data) >= min_rows, float(len(data)), min_rows, "error"))
    dup = float(data[id_column()].duplicated().mean())
    results.append(Expectation("unique_id", id_column(), dup == 0.0, dup, 0.0, "error"))
    bad_dates = float(data[date_column()].isna().mean())
    results.append(Expectation("valid_issue_date", date_column(), bad_dates == 0.0, bad_dates, 0.0, "error"))

    for name, spec in numeric_features().items():
        values = pd.to_numeric(data[name], errors="coerce")
        null_rate = float(values.isna().mean())
        limit = _MAX_NULL.get(name, 0.01)
        results.append(Expectation("null_rate", name, null_rate <= limit, null_rate, limit, "error"))
        for bound, op in (
            ("min", values.dropna() < spec.get("min", -1e18)),
            ("max", values.dropna() > spec.get("max", 1e18)),
        ):
            rate = float(op.sum() / n)
            results.append(Expectation(f"{bound}_value", name, rate <= 0.001, rate, 0.001, "warning"))

    for name, spec in categorical_features().items():
        null_rate = float(data[name].isna().mean())
        results.append(Expectation("null_rate", name, null_rate <= 0.01, null_rate, 0.01, "error"))
        if "values" in spec:
            unknown = float((~data[name].dropna().isin(spec["values"])).sum() / n)
            results.append(Expectation("known_categories", name, unknown <= 0.001, unknown, 0.001, "warning"))

    labeled = data[target_name()].dropna()
    rate = float(labeled.mean()) if len(labeled) else float("nan")
    results.append(Expectation("default_rate_range", target_name(), 0.03 <= rate <= 0.40, rate, 0.40, "error"))
    leak = [c for c in data_schema()["leakage_columns"] if c in data.columns]
    results.append(Expectation("no_leakage_columns", ",".join(leak) or "-", not leak, float(len(leak)), 0.0, "error"))
    return results


def assert_quality(results: list[Expectation]) -> None:
    for r in results:
        if not r.passed and r.severity == "warning":
            logger.warning("Expectativa en warning: %s(%s)=%.4f", r.name, r.column, r.observed)
    failed = [r for r in results if not r.passed and r.severity == "error"]
    if failed:
        detail = ", ".join(f"{r.name}({r.column})={r.observed:.4f}" for r in failed)
        raise ValueError(f"Gate de calidad de datos fallido: {detail}")


def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Descarta filas con valores imposibles (fila a fila: se puede aplicar por lotes en Spark)."""
    mask = pd.Series(True, index=data.index)
    for name, spec in numeric_features().items():
        values = data[name]
        if "min" in spec:
            mask &= values.isna() | (values >= spec["min"])
        if "max" in spec:
            mask &= values.isna() | (values <= spec["max"])
    cleaned = data.loc[mask].copy()
    dropped = len(data) - len(cleaned)
    if dropped:
        logger.info("Filas descartadas por rango: %d", dropped)
    return cleaned
