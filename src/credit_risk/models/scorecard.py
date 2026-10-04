"""Scorecard tradicional: binning, WoE, Information Value, regresión logística
sobre WoE y escalado a puntos. Es el benchmark interpretable de la banca.

Portado de credit-risk-ml-platform (v1) y extendido a variables categóricas
(grado, propósito, vivienda, estado...). Todo se ajusta **solo con el periodo de
entrenamiento**; validación y test únicamente se transforman.

Convenciones (Siddiqi, 2017):

* ``WoE = ln(%buenos / %malos)`` por bin; positivo = menor riesgo.
* ``IV = sum((%buenos - %malos) * WoE)``.
* ``score = offset + factor * ln(odds buenos:malos)`` con ``factor = PDO / ln 2``
  y ``offset = base_score - factor * ln(base_odds)``. Por defecto 600 puntos
  equivalen a odds 50:1 y cada 20 puntos duplican los odds (parámetros).
* Cada bin tiene al menos ``min_share`` de las observaciones. En categóricas, las
  categorías raras se agrupan en ``__otros__``; los faltantes forman un bin propio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MISSING_CODE = 9999
OTHER = "__otros__"
_SMOOTHING = 0.5


@dataclass
class FeatureBinning:
    """Bins de una variable numérica (intervalos) o categórica (niveles)."""

    name: str
    kind: str = "numeric"
    edges: list[float] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    woe: dict[int, float] = field(default_factory=dict)
    iv: float = 0.0
    table: list[dict[str, Any]] = field(default_factory=list)

    def assign(self, values: pd.Series) -> np.ndarray:
        """Código de bin por fila (faltante = 9999; categoría no vista = OTHER)."""
        if self.kind == "categorical":
            v = values.astype(object)
            missing = v.isna().to_numpy()
            lookup = {level: i for i, level in enumerate(self.levels)}
            other = lookup.get(OTHER, len(self.levels))
            codes = np.array([lookup.get(str(x), other) for x in v], dtype=int)
            codes[missing] = MISSING_CODE
            return codes
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        codes = np.full(len(arr), MISSING_CODE, dtype=int)
        valid = ~np.isnan(arr)
        codes[valid] = np.digitize(arr[valid], self.edges) if self.edges else 0
        return codes

    def describe(self, code: int) -> str:
        if code == MISSING_CODE:
            return "faltante"
        if self.kind == "categorical":
            return self.levels[code] if code < len(self.levels) else OTHER
        if not self.edges:
            return "todos"
        if code == 0:
            return f"< {self.edges[0]:.4g}"
        if code == len(self.edges):
            return f">= {self.edges[-1]:.4g}"
        return f"[{self.edges[code - 1]:.4g}, {self.edges[code]:.4g})"


def _candidate_edges(values: np.ndarray, max_bins: int, max_discrete: int = 30) -> list[float]:
    """Fronteras entre valores observados distintos (por valor si es discreta, por cuantiles si no)."""
    uniques = np.unique(values)
    if len(uniques) <= 1:
        return []
    if len(uniques) <= max_discrete:
        return [float(x) for x in (uniques[:-1] + uniques[1:]) / 2.0]
    edges = set()
    for q in np.quantile(values, np.linspace(0, 1, max_bins + 1)[1:-1]):
        above = uniques[uniques > q]
        if len(above):
            edges.add(float((q + above[0]) / 2.0))
    return sorted(edges)


def _merge_small_bins(values: np.ndarray, edges: list[float], min_share: float) -> list[float]:
    """Une bins adyacentes con menos de ``min_share`` de las observaciones."""
    edges = list(edges)
    n = len(values)
    while edges:
        counts = np.bincount(np.digitize(values, edges), minlength=len(edges) + 1)
        smallest = int(np.argmin(counts))
        if counts[smallest] / n >= min_share:
            break
        if smallest == 0:
            drop = 0
        elif smallest == len(counts) - 1:
            drop = len(edges) - 1
        else:
            drop = smallest - 1 if counts[smallest - 1] <= counts[smallest + 1] else smallest
        edges.pop(drop)
    return edges


def _fill_woe(binning: FeatureBinning, codes: np.ndarray, target: np.ndarray) -> FeatureBinning:
    y = np.asarray(target, dtype=int)
    total_good, total_bad = float((y == 0).sum()), float((y == 1).sum())
    present = np.unique(codes)
    binning.iv, binning.table = 0.0, []
    for code in present:
        mask = codes == code
        good, bad = float(((y == 0) & mask).sum()), float(((y == 1) & mask).sum())
        dist_good = (good + _SMOOTHING) / (total_good + _SMOOTHING * len(present))
        dist_bad = (bad + _SMOOTHING) / (total_bad + _SMOOTHING * len(present))
        woe = float(np.log(dist_good / dist_bad))
        iv = float((dist_good - dist_bad) * woe)
        binning.iv += iv
        binning.woe[int(code)] = woe
        binning.table.append(
            {
                "feature": binning.name,
                "bin": binning.describe(int(code)),
                "code": int(code),
                "count": int(mask.sum()),
                "share": float(mask.mean()),
                "bad_rate": float(bad / max(good + bad, 1.0)),
                "woe": woe,
                "iv_contribution": iv,
            }
        )
    binning.table.sort(key=lambda row: row["code"])
    return binning


def fit_binning(
    name: str, values: pd.Series, target: Any, max_bins: int = 10, min_share: float = 0.01
) -> FeatureBinning:
    """Binning numérico por cuantiles (o por valor si hay pocos distintos) + WoE e IV."""
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    regular = v[~np.isnan(v)]
    edges = _candidate_edges(regular, max_bins) if len(regular) else []
    edges = _merge_small_bins(regular, edges, min_share) if edges else []
    binning = FeatureBinning(name=name, kind="numeric", edges=edges)
    return _fill_woe(binning, binning.assign(values), target)


def fit_categorical_binning(name: str, values: pd.Series, target: Any, min_share: float = 0.01) -> FeatureBinning:
    """Un bin por categoría; las de frecuencia < ``min_share`` se agrupan en OTHER."""
    v = values.astype(object)
    shares = v.dropna().astype(str).value_counts(normalize=True)
    levels = sorted(shares[shares >= min_share].index.tolist())
    if (shares < min_share).any() or not levels:
        levels.append(OTHER)
    binning = FeatureBinning(name=name, kind="categorical", levels=levels)
    return _fill_woe(binning, binning.assign(values), target)


def information_value_label(iv: float) -> str:
    """Regla empírica habitual (Siddiqi): poder predictivo según el IV."""
    if iv < 0.02:
        return "inútil"
    if iv < 0.1:
        return "débil"
    if iv < 0.3:
        return "medio"
    if iv < 0.5:
        return "fuerte"
    return "muy alto (> 0.5: verificar posible fuga)"


class CreditScorecard:
    """Scorecard WoE + regresión logística con escalado a puntos."""

    def __init__(
        self,
        pdo: float = 20.0,
        base_score: float = 600.0,
        base_odds: float = 50.0,
        max_bins: int = 10,
        min_share: float = 0.01,
        regularization_c: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.pdo, self.base_score, self.base_odds = pdo, base_score, base_odds
        self.max_bins, self.min_share = max_bins, min_share
        self.regularization_c, self.random_state = regularization_c, random_state
        self.factor = pdo / np.log(2.0)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.binnings: dict[str, FeatureBinning] = {}
        self.model: LogisticRegression | None = None
        self.features: list[str] = []

    def fit(self, x: pd.DataFrame, y: Any) -> CreditScorecard:
        target = np.asarray(y, dtype=int)
        self.features = list(x.columns)
        for name in self.features:
            col = x[name]
            if pd.api.types.is_numeric_dtype(col):
                self.binnings[name] = fit_binning(name, col, target, self.max_bins, self.min_share)
            else:
                self.binnings[name] = fit_categorical_binning(name, col, target, self.min_share)
        self.model = LogisticRegression(C=self.regularization_c, max_iter=2000, random_state=self.random_state)
        self.model.fit(self._woe_matrix(x), target)
        return self

    def _woe_matrix(self, x: pd.DataFrame) -> np.ndarray:
        columns = []
        for name in self.features:
            binning = self.binnings[name]
            columns.append(np.array([binning.woe.get(int(c), 0.0) for c in binning.assign(x[name])]))
        return np.column_stack(columns)

    def _check_fitted(self) -> LogisticRegression:
        if self.model is None:
            raise RuntimeError("El scorecard no está ajustado; llama a fit().")
        return self.model

    def predict_pd(self, x: pd.DataFrame) -> np.ndarray:
        return self._check_fitted().predict_proba(self._woe_matrix(x))[:, 1]

    def pd_to_score(self, pd_values: Any) -> np.ndarray:
        p = np.clip(np.asarray(pd_values, dtype=float), 1e-9, 1 - 1e-9)
        return self.offset + self.factor * np.log((1.0 - p) / p)

    def score_to_pd(self, scores: Any) -> np.ndarray:
        return 1.0 / (1.0 + np.exp((np.asarray(scores, dtype=float) - self.offset) / self.factor))

    def score(self, x: pd.DataFrame) -> np.ndarray:
        """Puntaje: mayor = menor riesgo."""
        return self.pd_to_score(self.predict_pd(x))

    def iv_table(self) -> list[dict[str, Any]]:
        rows = [
            {"feature": n, "iv": b.iv, "strength": information_value_label(b.iv), "n_bins": len(b.table)}
            for n, b in self.binnings.items()
        ]
        return sorted(rows, key=lambda row: row["iv"], reverse=True)

    def points_table(self) -> list[dict[str, Any]]:
        """Puntos por bin: ``-(coef * WoE + intercept/n) * factor + offset/n``."""
        model = self._check_fitted()
        n = len(self.features)
        intercept = float(model.intercept_[0])
        rows = []
        for j, name in enumerate(self.features):
            coef = float(model.coef_[0][j])
            for entry in self.binnings[name].table:
                points = -(coef * entry["woe"] + intercept / n) * self.factor + self.offset / n
                rows.append({**entry, "coefficient": coef, "points": float(points)})
        return rows
