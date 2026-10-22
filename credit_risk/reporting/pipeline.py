"""Orquesta el análisis completo de un entrenamiento y devuelve un dict serializable.

Reproduce la misma secuencia que `jobs/03_train_register.py` (benchmark, selección
estadística en validación, tuning, reporte en test) y agrega el análisis que explica
por qué se eligió el modelo: inferencia, calibración, ordenamiento de cartera,
umbral por costos, estabilidad por cosecha, ablación de variables y SHAP.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.config import date_column, platform_config, target_name
from credit_risk.features.engineering import build_features
from credit_risk.inference.delong import delong_roc_test
from credit_risk.models import training as T
from credit_risk.models.credit_model import CreditRiskModel
from credit_risk.reporting import analysis as A

logger = logging.getLogger(__name__)

# Salidas del scoring interno de Lending Club (la tasa se fija según el grado).
LENDING_CLUB_SCORE_FEATURES = ("grade", "sub_grade", "int_rate")


def _neutralize(x: pd.DataFrame) -> pd.DataFrame:
    """Vuelve constantes las variables de Lending Club: el modelo no puede usarlas."""
    x = x.copy()
    x["grade"] = "N/A"
    x["sub_grade"] = "N/A"
    x["int_rate"] = 0.0
    return x


def ablation_without_lc_score(model: CreditRiskModel, splits: T.Splits) -> dict[str, Any]:
    """Reentrena el mismo algoritmo y parámetros sin grade, sub_grade ni int_rate.

    Cuantifica cuánto del poder del modelo viene del scoring propio de Lending Club.
    """
    reduced = replace(
        splits,
        x_train=_neutralize(splits.x_train),
        x_val=_neutralize(splits.x_val),
        x_test=_neutralize(splits.x_test),
    )
    params = model.metadata.get("params", {})
    without = T.fit_model(model.algorithm, params, reduced)
    y = splits.y_test.to_numpy()
    test = delong_roc_test(y, model.predict_proba(splits.x_test), without.predict_proba(reduced.x_test))
    z = 1.959963984540054
    return {
        "dropped": list(LENDING_CLUB_SCORE_FEATURES),
        "auc_full": test["auc_a"],
        "auc_without": test["auc_b"],
        "delta": test["diff"],
        "delta_ci": [test["diff"] - z * test["se"], test["diff"] + z * test["se"]],
        "p_value": test["p_value"],
    }


def tuning_effect(default: CreditRiskModel, tuned: CreditRiskModel, splits: T.Splits) -> dict[str, Any]:
    """Default vs ajustado con Optuna, en validación (donde se decide) y en test (informativo)."""
    out = {}
    for split, x, y in (("validation", splits.x_val, splits.y_val), ("test", splits.x_test, splits.y_test)):
        t = delong_roc_test(y.to_numpy(), tuned.predict_proba(x), default.predict_proba(x))
        out[split] = {"auc_tuned": t["auc_a"], "auc_default": t["auc_b"], "delta": t["diff"], "p_value": t["p_value"]}
    return out


def build_results(canonical: pd.DataFrame, optuna_trials: int | None = None, seed: int = 42) -> dict[str, Any]:
    """Corre el análisis completo sobre el dataset canónico (salida de parse_raw + clean)."""
    cfg = platform_config()
    tcfg = cfg["training"]
    trials = tcfg["optuna_trials"] if optuna_trials is None else optuna_trials
    labeled = canonical.dropna(subset=[target_name()])
    dates = pd.to_datetime(labeled[date_column()])
    split_cfg = cfg["data"]
    splits = T.make_splits(labeled[dates < pd.Timestamp(split_cfg["test_end"])], split_cfg)

    table, models = T.run_benchmark(splits)
    selection = T.statistical_selection(table, models, splits)
    scorecard = T.fit_scorecard(splits)
    comparison = T.compare_on_test(models, splits, scorecard)
    best, final, result, origin = T.choose_final_model(
        table, models, splits, trials, tcfg["optuna_timeout_seconds"], best=selection["selected"]
    )
    logger.info("Modelo final: %s (%s)", best, origin)
    later = labeled[dates >= pd.Timestamp(split_cfg["train_end"])]
    return assemble_results(
        splits, table, models, selection, comparison, scorecard, best, final, result, origin, later, seed
    )


def assemble_results(
    splits: T.Splits,
    table: pd.DataFrame,
    models: dict[str, CreditRiskModel],
    selection: dict[str, Any],
    comparison: dict[str, Any],
    scorecard,
    best: str,
    final: CreditRiskModel,
    result: dict[str, Any],
    origin: str,
    later: pd.DataFrame | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Arma el dict de resultados a partir de un entrenamiento ya hecho (lo usa también el job 03).

    `later` son préstamos etiquetados desde el inicio de la validación (incluida la
    producción 2014-2015) para la estabilidad por cosecha; si es None se omite.
    """
    tcfg = platform_config()["training"]
    y_test = splits.y_test.to_numpy()
    p_test = final.predict_proba(splits.x_test)
    p_score = scorecard.predict_pd(build_features(splits.x_test))
    p_lr = models["logistic_regression"].predict_proba(splits.x_test) if "logistic_regression" in models else None

    vintages = []
    if later is not None and len(later):
        vintages = A.vintage_stability(
            later[target_name()].to_numpy(), final.predict_proba(later), later[date_column()]
        )

    sample = splits.x_test.sample(min(3000, len(splits.x_test)), random_state=seed)
    contrib = final.contributions(sample)
    feats = build_features(sample)
    shap_rows = A.shap_summary(contrib, feats)
    dependence = {}
    for row in shap_rows:
        name = row["feature"]
        if name in feats.columns and pd.api.types.is_numeric_dtype(feats[name]) and len(dependence) < 3:
            ok = feats[name].notna()
            dependence[name] = {
                "values": feats.loc[ok, name].astype(float).round(4).tolist()[:1500],
                "shap": contrib.loc[ok, name].round(5).tolist()[:1500],
            }

    curve_models = {f"{best} (final)": p_test, "scorecard_woe": p_score}
    if p_lr is not None and best != "logistic_regression":
        curve_models["logistic_regression"] = p_lr
    passed, reasons = T.check_quality_gates(result)

    results: dict[str, Any] = {
        "periods": splits.periods,
        "sizes": {
            "train": int(len(splits.y_train)),
            "validation": int(len(splits.y_val)),
            "test": int(len(splits.y_test)),
            "default_rate_train": float(splits.y_train.mean()),
            "default_rate_test": float(y_test.mean()),
        },
        "benchmark": [
            {
                "algorithm": r["algorithm"],
                "train_auc": float(r["train_roc_auc"]),
                "val_auc": float(r["val_roc_auc"]),
                "test_auc": float(r["test_roc_auc"]),
                "val_gap": float(r["val_auc_gap"]),
                "test_ks": float(r["test_ks"]),
                "test_brier": float(r["test_brier"]),
                "latency_ms": float(r["latency_ms"]),
            }
            for _, r in table.iterrows()
        ],
        "selection": selection,
        "comparison": comparison,
        "final": {
            "algorithm": best,
            "origin": origin,
            "params": final.metadata.get("params", {}),
            "threshold": float(final.threshold),
            "discrimination": A.discrimination(y_test, p_test),
            "calibration": A.calibration(y_test, p_test),
            "deciles": A.decile_table(y_test, p_test),
            "thresholds": A.threshold_curve(y_test, p_test, tcfg["cost_false_negative"], tcfg["cost_false_positive"]),
            "gates": {"passed": passed, "reasons": reasons},
        },
        "curves": {name: A.curves(y_test, p) for name, p in curve_models.items()},
        "scorecard": {"discrimination": A.discrimination(y_test, p_score), "iv": scorecard.iv_table()},
        "vintages": vintages,
        "shap": shap_rows,
        "shap_dependence": dependence,
        "ablation": ablation_without_lc_score(final, splits),
        "costs": {"fn": tcfg["cost_false_negative"], "fp": tcfg["cost_false_positive"]},
        "selection_config": tcfg["selection"],
    }
    if origin == "optuna" and best in models:
        results["tuning"] = tuning_effect(models[best], final, splits)
    return results


def json_safe(obj: Any) -> Any:
    """Convierte NaN/inf y tipos NumPy a valores JSON estándar."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
