"""Test de DeLong para AUC correlacionados e intervalos de confianza.

Funciones puras (NumPy/SciPy), baratas de probar en CI:

- ``delong_auc_ci``: AUC con su error estándar de DeLong et al. (1988) e IC.
- ``delong_roc_test``: test bilateral de H0: AUC_a = AUC_b sobre las MISMAS
  observaciones (los AUC están correlacionados; un test de dos muestras sería
  incorrecto). Usa el algoritmo O(n log n) de Sun y Xu (2014).
- ``bootstrap_ci``: IC percentil por bootstrap para cualquier métrica.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats


def _fast_delong(predictions_sorted: np.ndarray, n_positives: int) -> tuple[np.ndarray, np.ndarray]:
    """Sun y Xu (2014). ``predictions_sorted`` es (k modelos, n) con los positivos primero.

    Devuelve (AUC por modelo, matriz de covarianza de los AUC).
    """
    m = n_positives
    n = predictions_sorted.shape[1] - m
    positives = predictions_sorted[:, :m]
    negatives = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]

    tx = np.vstack([stats.rankdata(positives[r]) for r in range(k)])
    ty = np.vstack([stats.rankdata(negatives[r]) for r in range(k)])
    tz = np.vstack([stats.rankdata(predictions_sorted[r]) for r in range(k)])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def _sort_positives_first(y_true: Any, *scores: Any) -> tuple[np.ndarray, int]:
    y = np.asarray(y_true).astype(int)
    if y.min() == y.max():
        raise ValueError("DeLong necesita observaciones de ambas clases.")
    order = np.argsort(-y, kind="stable")
    matrix = np.vstack([np.asarray(score, dtype=float)[order] for score in scores])
    return matrix, int(y.sum())


def delong_auc_ci(y_true: Any, scores: Any, alpha: float = 0.05) -> dict[str, float]:
    """AUC con su error estándar de DeLong e IC al (1 - alpha)."""
    matrix, m = _sort_positives_first(y_true, scores)
    aucs, cov = _fast_delong(matrix, m)
    auc = float(aucs[0])
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    z = float(stats.norm.ppf(1 - alpha / 2))
    return {"auc": auc, "se": se, "ci_low": max(auc - z * se, 0.0), "ci_high": min(auc + z * se, 1.0)}


def delong_roc_test(y_true: Any, scores_a: Any, scores_b: Any) -> dict[str, float]:
    """Test bilateral de DeLong de H0: AUC_a = AUC_b (mismas observaciones)."""
    matrix, m = _sort_positives_first(y_true, scores_a, scores_b)
    aucs, cov = _fast_delong(matrix, m)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ cov @ contrast)
    diff = float(aucs[0] - aucs[1])
    if variance <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "se": 0.0, "z": 0.0, "p_value": 1.0}
    se = float(np.sqrt(variance))
    z = diff / se
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": diff,
        "se": se,
        "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


def bootstrap_ci(
    y_true: Any,
    scores: Any,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, float]:
    """IC percentil por bootstrap (remuestreo de observaciones con reemplazo).

    Se omiten los remuestreos con una sola clase (solo posible en muestras chicas).
    """
    y = np.asarray(y_true)
    s = np.asarray(scores)
    rng = np.random.default_rng(random_state)
    n = len(y)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        values.append(metric(y[idx], s[idx]))
    arr = np.asarray(values)
    return {
        "estimate": float(metric(y, s)),
        "ci_low": float(np.quantile(arr, alpha / 2)),
        "ci_high": float(np.quantile(arr, 1 - alpha / 2)),
        "n_boot": int(len(arr)),
    }
