"""Algoritmos candidatos del benchmark y sus espacios de búsqueda (Optuna)."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def is_available(name: str) -> bool:
    module = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}.get(name)
    return module is None or importlib.util.find_spec(module) is not None


def default_params(name: str) -> dict[str, Any]:
    """Valores regularizados: en crédito el sobreajuste se paga caro en producción."""
    return {
        "logistic_regression": {"C": 0.5},
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 30,
            "reg_lambda": 5.0,
        },
        "lightgbm": {
            "n_estimators": 300,
            "num_leaves": 15,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_samples": 200,
            "reg_lambda": 5.0,
        },
        "catboost": {"iterations": 400, "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 10},
    }[name]


def suggest_params(trial, name: str) -> dict[str, Any]:
    """Espacio de búsqueda de hiperparámetros por algoritmo."""
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 300, 900, step=100),
            "depth": trial.suggest_int("depth", 4, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    raise ValueError(f"Algoritmo desconocido: {name}")


KNOWN = ("logistic_regression", "xgboost", "lightgbm", "catboost")


def build_estimator(name: str, params: dict[str, Any] | None = None, seed: int = 42):
    if name not in KNOWN:
        raise ValueError(f"Algoritmo desconocido: {name}")
    params = {**default_params(name), **(params or {})}
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, **params)),
            ]
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            eval_metric="auc",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **params, random_seed=seed, verbose=False, eval_metric="AUC", allow_writing_files=False
        )
    raise ValueError(f"Algoritmo desconocido: {name}")


def available_candidates(names: list[str]) -> list[str]:
    usable = []
    for name in names:
        if is_available(name):
            usable.append(name)
        else:
            logger.warning("Candidato %s omitido: dependencia no instalada", name)
    return usable
