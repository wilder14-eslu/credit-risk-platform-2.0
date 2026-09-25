"""Modelo servible: features + preprocesamiento + estimador + umbral + explicación.

`CreditRiskModel` es Python puro (testeable sin MLflow). Se registra en Unity
Catalog envuelto en `CreditRiskPyfunc`, que es lo que ejecuta el endpoint de
Databricks Model Serving: recibe las variables de la solicitud y devuelve
probabilidad, decisión, banda de riesgo y los factores que más pesaron.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.features.engineering import Preprocessor, build_features, feature_label


def risk_band(probability: float) -> str:
    for limit, band in ((0.05, "A"), (0.10, "B"), (0.20, "C"), (0.35, "D")):
        if probability < limit:
            return band
    return "E"


@dataclass
class CreditRiskModel:
    algorithm: str
    estimator: Any
    preprocessor: Preprocessor
    threshold: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def matrix(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(build_features(raw))

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.matrix(raw).to_numpy())[:, 1]

    def contributions(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Contribución al log-odds por variable ORIGINAL (one-hot agrupado)."""
        x = self.matrix(raw)
        arr = x.to_numpy()
        if self.algorithm == "xgboost":
            import xgboost as xgb

            values = self.estimator.get_booster().predict(xgb.DMatrix(arr), pred_contribs=True)[:, :-1]
        elif self.algorithm == "lightgbm":
            values = self.estimator.predict(arr, pred_contrib=True)[:, :-1]
        elif self.algorithm == "catboost":
            from catboost import Pool

            values = self.estimator.get_feature_importance(Pool(arr), type="ShapValues")[:, :-1]
        else:
            scaler = self.estimator.named_steps["scaler"]
            values = scaler.transform(arr) * self.estimator.named_steps["clf"].coef_[0]
        per_column = pd.DataFrame(values, columns=x.columns, index=raw.index)
        return per_column.T.groupby(Preprocessor.source_feature).sum().T

    def predict_frame(self, raw: pd.DataFrame, explain: bool = True, top_k: int = 3) -> pd.DataFrame:
        proba = self.predict_proba(raw)
        out = pd.DataFrame(
            {
                "probability": proba,
                "decision": np.where(proba >= self.threshold, "RECHAZAR", "APROBAR"),
                "risk_band": [risk_band(p) for p in proba],
            },
            index=raw.index,
        )
        if not explain:
            out["top_factors"] = "[]"
            return out
        contrib = self.contributions(raw)
        factors = []
        for _, row in contrib.iterrows():
            top = row.reindex(row.abs().sort_values(ascending=False).index)[:top_k]
            factors.append(
                json.dumps(
                    [
                        {
                            "feature": k,
                            "label": feature_label(k),
                            "impact": round(float(v), 4),
                            "direction": "aumenta_riesgo" if v > 0 else "reduce_riesgo",
                        }
                        for k, v in top.items()
                    ],
                    ensure_ascii=False,
                )
            )
        out["top_factors"] = factors
        return out

    def global_importance(self, sample: pd.DataFrame) -> pd.Series:
        return self.contributions(sample).abs().mean().sort_values(ascending=False)


try:  # MLflow es opcional en tests unitarios
    import mlflow.pyfunc

    class CreditRiskPyfunc(mlflow.pyfunc.PythonModel):
        """Wrapper que corre en Databricks Model Serving."""

        def load_context(self, context) -> None:
            import joblib

            self.model: CreditRiskModel = joblib.load(context.artifacts["credit_model"])

        def predict(self, context, model_input: pd.DataFrame, params: dict | None = None):
            explain = bool((params or {}).get("explain", True))
            return self.model.predict_frame(model_input, explain=explain).reset_index(drop=True)

except ImportError:  # pragma: no cover
    CreditRiskPyfunc = None  # type: ignore[assignment,misc]
