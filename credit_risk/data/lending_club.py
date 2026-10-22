"""Parser del formato crudo de Lending Club -> formato canónico.

Es una función pandas pura: se prueba en CI sin Spark y en Databricks se aplica
por lotes con `mapInPandas`, de modo que el mismo código transforma datos en
tests, en el pipeline y en la API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import (
    categorical_features,
    data_schema,
    date_column,
    id_column,
    numeric_features,
    platform_config,
    target_name,
)

_EMP_LENGTH = {
    "< 1 year": 0.0,
    "1 year": 1.0,
    **{f"{i} years": float(i) for i in range(2, 10)},
    "10+ years": 10.0,
}


def _to_float(series: pd.Series) -> pd.Series:
    """Convierte números que pueden venir como texto ('13.56%', ' 36 months')."""
    if series.dtype.kind in "fi":
        return series.astype("float64")
    cleaned = series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True).replace("", pd.NA)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def _to_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("string").str.strip(), format="%b-%Y", errors="coerce")


def label_from_status(status: pd.Series) -> pd.Series:
    """1 = incumplió, 0 = pagó, NaN = préstamo sin desenlace (no se usa para entrenar)."""
    mapping = data_schema()["target_mapping"]
    stripped = status.astype("string").str.strip()
    label = pd.Series(np.nan, index=status.index, dtype="float64")
    label[stripped.isin(mapping["positive"])] = 1.0
    label[stripped.isin(mapping["negative"])] = 0.0
    return label


def is_matured(issue_month: pd.Series, term_months: pd.Series, snapshot: str | None = None) -> pd.Series:
    """True si el plazo del préstamo terminó antes del corte del dataset (evita censura por la derecha)."""
    snapshot_ts = pd.Timestamp(snapshot or platform_config()["data"]["label_snapshot"])
    months_left = (snapshot_ts.year - issue_month.dt.year) * 12 + (snapshot_ts.month - issue_month.dt.month)
    return (months_left >= term_months).fillna(False)


def canonical_columns() -> list[str]:
    return [
        id_column(),
        date_column(),
        *numeric_features(),
        *categorical_features(),
        "loan_status",
        target_name(),
    ]


def spark_schema_ddl() -> str:
    """Esquema DDL de la salida de `parse_raw` (para `mapInPandas`)."""
    parts = [f"{id_column()} STRING", f"{date_column()} TIMESTAMP"]
    parts += [f"{c} DOUBLE" for c in numeric_features()]
    parts += [f"{c} STRING" for c in categorical_features()]
    parts += ["loan_status STRING", f"{target_name()} DOUBLE"]
    return ", ".join(parts)


def parse_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Formato crudo de Lending Club -> formato canónico (sin columnas de leakage)."""
    schema = data_schema()
    raw = raw.loc[pd.to_numeric(raw.get("loan_amnt"), errors="coerce").notna()]  # quita filas pie
    out = pd.DataFrame(index=raw.index)
    out[id_column()] = raw[schema["id_source"]].astype("string").str.strip()
    issue = _to_month(raw[schema["date_source"]])
    out[date_column()] = issue

    for name, spec in numeric_features().items():
        source = spec["source"]
        if name == "emp_length_years":
            out[name] = raw[source].astype("string").str.strip().map(_EMP_LENGTH).astype("float64")
        elif name == "fico_score":
            low = _to_float(raw[source])
            high = _to_float(raw["fico_range_high"]) if "fico_range_high" in raw else low
            out[name] = (low + high.fillna(low)) / 2.0
        elif name == "credit_history_months":
            first = _to_month(raw[source])
            months = (issue.dt.year - first.dt.year) * 12 + (issue.dt.month - first.dt.month)
            out[name] = months.astype("float64")
        else:
            out[name] = _to_float(raw[source])

    for name, spec in categorical_features().items():
        values = raw[spec["source"]].astype("string").str.strip()
        if name == "home_ownership":
            values = values.replace({"NONE": "OTHER", "ANY": "OTHER"})
        if name == "application_type":
            values = values.replace({"JOINT": "Joint App", "INDIVIDUAL": "Individual"})
        out[name] = values.astype(object).where(values.notna(), None)

    out["loan_status"] = raw[schema["target_source"]].astype("string").str.strip().astype(object)
    label = label_from_status(raw[schema["target_source"]])
    out[target_name()] = label.where(is_matured(issue, out["term_months"]))
    return out[canonical_columns()].reset_index(drop=True)


def parse_batches(batches):
    """Adaptador para `DataFrame.mapInPandas` en Spark."""
    for batch in batches:
        yield parse_raw(batch)
