"""Feature engineering compartido por entrenamiento, batch scoring y serving.

`build_features` crea variables derivadas y `Preprocessor` (ajustado solo con
train) imputa numéricas y codifica categóricas. Ambos viajan dentro del modelo
registrado en Unity Catalog, así el endpoint aplica exactamente la misma
transformación que el entrenamiento (sin training/serving skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import categorical_features, input_features, numeric_features

DERIVED_FEATURES: dict[str, str] = {
    "loan_to_income": "Monto / ingreso anual",
    "installment_to_income": "Cuota anual / ingreso anual",
    "revol_bal_to_income": "Saldo revolvente / ingreso anual",
    "open_acc_ratio": "Líneas abiertas / líneas totales",
    "log_annual_inc": "Log del ingreso anual",
}
OTHER = "__other__"
MISSING = "__missing__"


def feature_label(name: str) -> str:
    specs = {**numeric_features(), **categorical_features()}
    if name in specs:
        return specs[name].get("label", name)
    return DERIVED_FEATURES.get(name, name)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Numéricas + derivadas (float) y categóricas (texto), sin imputar."""
    missing = [c for c in input_features() if c not in data.columns]
    if missing:
        raise ValueError(f"Faltan variables de entrada: {missing}")
    num = data[list(numeric_features())].apply(pd.to_numeric, errors="coerce").astype("float64")
    income = num["annual_inc"].where(num["annual_inc"] > 0)
    num["loan_to_income"] = num["loan_amnt"] / income
    num["installment_to_income"] = num["installment"] * 12 / income
    num["revol_bal_to_income"] = num["revol_bal"] / income
    num["open_acc_ratio"] = num["open_acc"] / num["total_acc"].where(num["total_acc"] > 0)
    num["log_annual_inc"] = np.log1p(num["annual_inc"].clip(lower=0))
    cat = data[list(categorical_features())].astype(object).where(data[list(categorical_features())].notna(), None)
    return pd.concat([num, cat], axis=1)


class Preprocessor:
    """Imputación por mediana + one-hot con vocabulario aprendido en train."""

    def __init__(self, min_category_share: float = 0.005) -> None:
        self.min_category_share = min_category_share
        self.medians_: dict[str, float] = {}
        self.vocab_: dict[str, list[str]] = {}
        self.columns_: list[str] = []

    @property
    def numeric_columns(self) -> list[str]:
        return list(numeric_features()) + list(DERIVED_FEATURES)

    def fit(self, features: pd.DataFrame) -> Preprocessor:
        for col in self.numeric_columns:
            self.medians_[col] = float(np.nan_to_num(features[col].median(), nan=0.0))
        for col in categorical_features():
            share = features[col].fillna(MISSING).value_counts(normalize=True)
            self.vocab_[col] = sorted(share[share >= self.min_category_share].index.astype(str))
        self.columns_ = list(self.transform(features.head(1)).columns)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.medians_:
            raise RuntimeError("Preprocessor no está ajustado")
        num = features[self.numeric_columns].fillna(value=self.medians_)
        parts = [num]
        for col, vocab in self.vocab_.items():
            values = features[col].fillna(MISSING).astype(str)
            values = values.where(values.isin(vocab), OTHER)
            for cat in [*vocab, OTHER]:
                parts.append(
                    pd.Series((values == cat).astype("float64").to_numpy(), index=features.index, name=f"{col}={cat}")
                )
        out = pd.concat(parts, axis=1)
        if self.columns_:
            out = out.reindex(columns=self.columns_, fill_value=0.0)
        return out

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)

    @staticmethod
    def source_feature(column: str) -> str:
        """`grade=B` -> `grade` (para agrupar contribuciones SHAP por variable original)."""
        return column.split("=", 1)[0]
