"""Data drift (covariate shift) y prediction drift: PSI para numéricas y categóricas + test KS.

El perfil de referencia se guarda en Delta (`reference_profile`) por versión de
modelo, de modo que el monitoreo compara siempre contra la distribución con la
que se entrenó el champion vigente.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-6
OTHER = "__other__"


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _edges(values: pd.Series, bins: int) -> list[float]:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return [-np.inf, np.inf]
    qs = np.unique(np.quantile(clean, np.linspace(0, 1, bins + 1)[1:-1]))
    return [-np.inf, *qs.tolist(), np.inf]


def _hist(values: pd.Series, edges: list[float]) -> np.ndarray:
    counts, _ = np.histogram(values.dropna().to_numpy(dtype=float), bins=np.asarray(edges, dtype=float))
    return counts / max(counts.sum(), 1)


def _cat_shares(values: pd.Series, categories: list[str]) -> np.ndarray:
    v = values.fillna("__missing__").astype(str)
    v = v.where(v.isin(categories), OTHER)
    return v.value_counts(normalize=True).reindex([*categories, OTHER], fill_value=0.0).to_numpy()


def build_reference_profile(
    features: pd.DataFrame, scores: np.ndarray | None = None, bins: int = 10, sample_size: int = 5000
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    frame = features.copy()
    if scores is not None:
        frame["__score__"] = np.asarray(scores, dtype=float)
    profile: dict[str, Any] = {"features": {}}
    for col in frame.columns:
        series = frame[col]
        if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
            shares = series.fillna("__missing__").astype(str).value_counts(normalize=True)
            cats = sorted(shares[shares >= 0.005].index)
            profile["features"][col] = {
                "kind": "categorical",
                "categories": cats,
                "proportions": _cat_shares(series, cats).tolist(),
                "null_rate": float(series.isna().mean()),
            }
            continue
        clean = series.dropna().to_numpy(dtype=float)
        edges = _edges(series, bins)
        sample = rng.choice(clean, size=min(sample_size, clean.size), replace=False) if clean.size else clean
        profile["features"][col] = {
            "kind": "numeric",
            "edges": edges,
            "proportions": _hist(series, edges).tolist(),
            "null_rate": float(series.isna().mean()),
            "mean": float(clean.mean()) if clean.size else float("nan"),
            "sample": sample.tolist(),
        }
    return profile


def feature_drift(
    profile: dict[str, Any],
    current: pd.DataFrame,
    psi_warning: float = 0.10,
    psi_alert: float = 0.25,
    ks_pvalue_alert: float = 0.01,
) -> pd.DataFrame:
    rows = []
    for col, ref in profile["features"].items():
        if col not in current.columns:
            continue
        cur = current[col]
        ks_stat = ks_p = mean_cur = float("nan")
        if ref["kind"] == "categorical":
            value = psi(np.asarray(ref["proportions"]), _cat_shares(cur, ref["categories"]))
        else:
            value = psi(np.asarray(ref["proportions"]), _hist(cur, ref["edges"]))
            clean = pd.to_numeric(cur, errors="coerce").dropna().to_numpy(dtype=float)
            if clean.size and len(ref["sample"]):
                ks = ks_2samp(np.asarray(ref["sample"]), clean)
                ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
                mean_cur = float(clean.mean())
        status = "alerta" if value >= psi_alert else "warning" if value >= psi_warning else "estable"
        rows.append(
            {
                "feature": "prediction" if col == "__score__" else col,
                "kind": ref["kind"],
                "psi": value,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "ks_drift": bool(ks_p < ks_pvalue_alert) if not np.isnan(ks_p) else False,
                "null_rate_ref": ref["null_rate"],
                "null_rate_cur": float(cur.isna().mean()),
                "mean_ref": ref.get("mean", float("nan")),
                "mean_cur": mean_cur,
                "status": status,
            }
        )
    return pd.DataFrame(rows)
