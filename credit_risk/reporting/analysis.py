"""Análisis estadístico del modelo: funciones puras (NumPy/pandas/SciPy), sin gráficos ni I/O.

Cada función devuelve estructuras serializables a JSON para que el mismo resultado
alimente las figuras, el README y los artefactos de MLflow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from credit_risk.inference.delong import bootstrap_ci, delong_auc_ci
from credit_risk.models import metrics as M


def _wilson(k: float, n: float, alpha: float = 0.05) -> tuple[float, float]:
    """IC de Wilson para una proporción (estable con tasas bajas o n chico)."""
    if n == 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(center - half), float(center + half)


def discrimination(y: np.ndarray, p: np.ndarray, n_boot: int = 300) -> dict[str, Any]:
    """AUC (IC de DeLong), Gini, KS y PR-AUC con IC bootstrap."""
    auc = delong_auc_ci(y, p)
    ks = bootstrap_ci(y, p, M.ks_statistic, n_boot=n_boot)
    pr = bootstrap_ci(y, p, average_precision_score, n_boot=n_boot)
    return {
        "auc": auc["auc"],
        "auc_ci": [auc["ci_low"], auc["ci_high"]],
        "gini": 2 * auc["auc"] - 1,
        "ks": ks["estimate"],
        "ks_ci": [ks["ci_low"], ks["ci_high"]],
        "pr_auc": pr["estimate"],
        "pr_auc_ci": [pr["ci_low"], pr["ci_high"]],
        "prevalence": float(np.mean(y)),
    }


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Calibración: Brier, ECE, pendiente e intercepto, test de Spiegelhalter y tabla de fiabilidad.

    - Pendiente/intercepto: regresión logística de y sobre logit(p). Ideal 1 y 0.
    - Spiegelhalter (1986): z ~ N(0, 1) bajo H0 "las probabilidades están bien calibradas".
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    z = float(num / den) if den > 0 else 0.0
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    table = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n, k = int(mask.sum()), float(y[mask].sum())
        if n == 0:
            continue
        lo, hi = _wilson(k, n)
        table.append({"bin": b + 1, "n": n, "mean_pd": float(p[mask].mean()), "observed": k / n, "ci": [lo, hi]})
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "brier_reference": float(np.mean(y) * (1 - np.mean(y))),
        "ece": M.expected_calibration_error(y, p, bins),
        "slope": float(lr.coef_[0][0]),
        "intercept": float(lr.intercept_[0]),
        "spiegelhalter_z": z,
        "spiegelhalter_p": float(2 * stats.norm.sf(abs(z))),
        "mean_pd": float(p.mean()),
        "observed_rate": float(y.mean()),
        "table": table,
    }


def decile_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    """Tabla de ganancias: decil 1 = más riesgoso. Tasa de default, captura acumulada y lift."""
    y = np.asarray(y, dtype=float)
    order = np.argsort(-np.asarray(p, dtype=float), kind="stable")
    ys, ps = y[order], np.asarray(p, dtype=float)[order]
    groups = np.array_split(np.arange(len(ys)), n_bins)
    total_bad, base = ys.sum(), ys.mean()
    rows, cum_bad, cum_n = [], 0.0, 0
    for i, g in enumerate(groups, start=1):
        bad = float(ys[g].sum())
        cum_bad += bad
        cum_n += len(g)
        lo, hi = _wilson(bad, len(g))
        rows.append(
            {
                "decile": i,
                "n": len(g),
                "default_rate": bad / len(g),
                "default_rate_ci": [lo, hi],
                "mean_pd": float(ps[g].mean()),
                "cum_capture": cum_bad / total_bad,
                "cum_population": cum_n / len(ys),
                "lift": (bad / len(g)) / base,
            }
        )
    return rows


def threshold_curve(
    y: np.ndarray, p: np.ndarray, cost_fn: float, cost_fp: float, grid: np.ndarray | None = None
) -> list[dict[str, Any]]:
    """Por umbral: aprobación, morosidad entre aprobados, recall, precisión y costo esperado por préstamo."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    grid = np.round(np.linspace(0.02, 0.60, 59), 3) if grid is None else grid
    rows = []
    for t in grid:
        reject = p >= t
        tp = int((reject & (y == 1)).sum())
        fp = int((reject & (y == 0)).sum())
        fn = int((~reject & (y == 1)).sum())
        approved = int((~reject).sum())
        rows.append(
            {
                "threshold": float(t),
                "approval_rate": approved / len(y),
                "bad_rate_approved": fn / approved if approved else float("nan"),
                "recall": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "expected_cost": (cost_fn * fn + cost_fp * fp) / len(y),
            }
        )
    return rows


def curves(y: np.ndarray, p: np.ndarray, points: int = 200) -> dict[str, list[float]]:
    """ROC y Precision-Recall submuestreadas (para graficar sin guardar miles de puntos)."""
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    i = np.unique(np.linspace(0, len(fpr) - 1, points).astype(int))
    j = np.unique(np.linspace(0, len(rec) - 1, points).astype(int))
    return {
        "fpr": fpr[i].tolist(),
        "tpr": tpr[i].tolist(),
        "recall": rec[j].tolist(),
        "precision": prec[j].tolist(),
    }


def vintage_stability(
    y: np.ndarray, p: np.ndarray, issue_dates: pd.Series, min_defaults: int = 30
) -> list[dict[str, Any]]:
    """AUC (IC de DeLong), tasa observada y PD media por trimestre de emisión."""
    frame = pd.DataFrame({"y": np.asarray(y, dtype=int), "p": np.asarray(p, dtype=float)})
    frame["quarter"] = pd.to_datetime(issue_dates.to_numpy()).to_period("Q").astype(str)
    rows = []
    for quarter, g in frame.groupby("quarter", sort=True):
        if g["y"].sum() < min_defaults or g["y"].nunique() < 2:
            continue
        auc = delong_auc_ci(g["y"].to_numpy(), g["p"].to_numpy())
        rows.append(
            {
                "quarter": quarter,
                "n": len(g),
                "auc": auc["auc"],
                "auc_ci": [auc["ci_low"], auc["ci_high"]],
                "observed": float(g["y"].mean()),
                "mean_pd": float(g["p"].mean()),
            }
        )
    return rows


def shap_summary(contributions: pd.DataFrame, raw: pd.DataFrame, top: int = 12) -> list[dict[str, Any]]:
    """|SHAP| medio por variable original y dirección (Spearman entre valor y contribución)."""
    importance = contributions.abs().mean().sort_values(ascending=False).head(top)
    rows = []
    for name, value in importance.items():
        direction = float("nan")
        if name in raw.columns:
            values = pd.to_numeric(raw[name], errors="coerce")
            ok = values.notna()
            if ok.sum() > 10 and values[ok].nunique() > 1:
                direction = float(stats.spearmanr(values[ok], contributions.loc[ok, name]).statistic)
        rows.append({"feature": name, "mean_abs_shap": float(value), "spearman": direction})
    return rows
