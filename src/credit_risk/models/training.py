"""Split temporal out-of-time, benchmark, tuning con Optuna, quality gates e informe."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.config import date_column, input_features, platform_config, target_name
from credit_risk.features.engineering import Preprocessor, build_features
from credit_risk.models import metrics as M
from credit_risk.models.candidates import available_candidates, build_estimator, suggest_params
from credit_risk.models.credit_model import CreditRiskModel

logger = logging.getLogger(__name__)


@dataclass
class Splits:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    periods: dict[str, str]


def split_config(as_of: str | None = None) -> dict[str, Any]:
    """Cortes temporales. Sin `as_of` usa config/platform.yaml (entrenamiento inicial).

    Con `as_of` (reloj del replay cuando el monitoreo dispara Continuous Training)
    las ventanas se desplazan para incorporar los desenlaces más recientes ya
    observados: test = últimos 6 meses etiquetados, validación = 6 meses previos,
    train = todo lo anterior. Así el reentrenamiento aprende del drift real.
    """
    cfg = dict(platform_config()["data"])
    if not as_of:
        return cfg
    delay = platform_config()["replay"]["label_delay_months"]
    label_end = pd.Timestamp(as_of).to_period("M").to_timestamp() - pd.DateOffset(months=delay - 1)
    cfg["test_end"] = str(label_end.date())
    cfg["validation_end"] = str((label_end - pd.DateOffset(months=6)).date())
    cfg["train_end"] = str((label_end - pd.DateOffset(months=12)).date())
    return cfg


def make_splits(data: pd.DataFrame, cfg: dict | None = None) -> Splits:
    """Split por fecha de originación (como se valida un scorecard): nunca aleatorio.

    train < train_end <= validación < validation_end <= test < test_end.
    Solo préstamos con desenlace conocido (target no nulo).
    """
    cfg = cfg or platform_config()["data"]
    labeled = data.dropna(subset=[target_name()])
    dates = pd.to_datetime(labeled[date_column()])
    t1, t2, t3 = (pd.Timestamp(cfg[k]) for k in ("train_end", "validation_end", "test_end"))
    parts = {
        "train": labeled[dates < t1],
        "val": labeled[(dates >= t1) & (dates < t2)],
        "test": labeled[(dates >= t2) & (dates < t3)],
    }
    if len(parts["train"]) > cfg["max_train_rows"]:
        parts["train"] = parts["train"].sample(cfg["max_train_rows"], random_state=cfg["random_state"])
    for name, part in parts.items():
        if part.empty or part[target_name()].nunique() < 2:
            raise ValueError(f"El periodo '{name}' no tiene datos etiquetados suficientes")
    cols = list(input_features())
    return Splits(
        parts["train"][cols],
        parts["train"][target_name()].astype(int),
        parts["val"][cols],
        parts["val"][target_name()].astype(int),
        parts["test"][cols],
        parts["test"][target_name()].astype(int),
        periods={
            k: f"{pd.to_datetime(v[date_column()]).min():%Y-%m} a "
            f"{pd.to_datetime(v[date_column()]).max():%Y-%m} ({len(v)} préstamos)"
            for k, v in parts.items()
        },
    )


def _fit_preprocessor(splits: Splits) -> tuple[Preprocessor, np.ndarray]:
    pre = Preprocessor()
    x = pre.fit_transform(build_features(splits.x_train))
    return pre, x.to_numpy()


def fit_model(name: str, params: dict[str, Any], splits: Splits, seed: int = 42) -> CreditRiskModel:
    pre, x_train = _fit_preprocessor(splits)
    estimator = build_estimator(name, params, seed=seed)
    estimator.fit(x_train, splits.y_train.to_numpy())
    model = CreditRiskModel(name, estimator, pre, metadata={"params": params, "periods": splits.periods})
    tcfg = platform_config()["training"]
    model.threshold = M.optimal_threshold(
        splits.y_val, model.predict_proba(splits.x_val), tcfg["cost_false_negative"], tcfg["cost_false_positive"]
    )
    return model


def evaluate(model: CreditRiskModel, splits: Splits) -> dict[str, Any]:
    out: dict[str, Any] = {"threshold": model.threshold}
    for prefix, x, y in (("train", splits.x_train, splits.y_train), ("test", splits.x_test, splits.y_test)):
        for k, v in M.classification_metrics(y, model.predict_proba(x), model.threshold).items():
            out[f"{prefix}_{k}"] = v
    out["latency_ms"] = M.inference_latency_ms(model.predict_proba, splits.x_test)
    out.update(M.diagnose_fit(out["train_roc_auc"], out["test_roc_auc"]))
    return out


def tune(name: str, splits: Splits, n_trials: int, timeout: int, seed: int = 42) -> dict[str, Any]:
    """Optuna TPE maximizando AUC en el periodo de validación (el test no se toca)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pre, x_train = _fit_preprocessor(splits)
    x_val = pre.transform(build_features(splits.x_val)).to_numpy()

    def objective(trial) -> float:
        est = build_estimator(name, suggest_params(trial, name), seed=seed)
        est.fit(x_train, splits.y_train.to_numpy())
        return M.classification_metrics(splits.y_val, est.predict_proba(x_val)[:, 1])["roc_auc"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    logger.info("Optuna %s: mejor AUC validación=%.4f", name, study.best_value)
    return dict(study.best_params)


def run_benchmark(
    splits: Splits, candidates: list[str] | None = None
) -> tuple[pd.DataFrame, dict[str, CreditRiskModel]]:
    tcfg = platform_config()["training"]
    rows, models = [], {}
    for name in available_candidates(candidates or tcfg["candidates"]):
        model = fit_model(name, {}, splits)
        result = evaluate(model, splits)
        result.update(algorithm=name, params={})
        rows.append(result)
        models[name] = model
        logger.info("%s -> AUC test OOT %.4f", name, result["test_roc_auc"])
    table = pd.DataFrame(rows).sort_values(tcfg["selection_metric"], ascending=False)
    return table.reset_index(drop=True), models


def select_best(table: pd.DataFrame, metric: str | None = None) -> str:
    metric = metric or platform_config()["training"]["selection_metric"]
    gates = platform_config()["quality_gates"]
    eligible = table[(table["latency_ms"] <= gates["max_latency_ms"]) & (table["auc_gap"] <= gates["max_overfit_gap"])]
    source = eligible if not eligible.empty else table
    return str(source.sort_values(metric, ascending=False).iloc[0]["algorithm"])


def check_quality_gates(result: dict[str, Any], gates: dict | None = None) -> tuple[bool, list[str]]:
    gates = gates or platform_config()["quality_gates"]
    checks = [
        (
            result["test_roc_auc"] >= gates["min_test_roc_auc"],
            f"AUC test {result['test_roc_auc']:.4f} < {gates['min_test_roc_auc']}",
        ),
        (
            result["auc_gap"] <= gates["max_overfit_gap"],
            f"brecha train-test {result['auc_gap']:.4f} > {gates['max_overfit_gap']}",
        ),
        (result["test_brier"] <= gates["max_brier"], f"Brier {result['test_brier']:.4f} > {gates['max_brier']}"),
        (
            result["latency_ms"] <= gates["max_latency_ms"],
            f"latencia {result['latency_ms']:.2f}ms > {gates['max_latency_ms']}",
        ),
    ]
    reasons = [msg for ok, msg in checks if not ok]
    return not reasons, reasons


def champion_vs_challenger(
    challenger_auc: float, champion_auc: float | None, min_improvement: float | None = None
) -> tuple[bool, str]:
    if min_improvement is None:
        min_improvement = platform_config()["quality_gates"]["min_auc_improvement"]
    if champion_auc is None or np.isnan(champion_auc):
        return True, "no existe champion: el primer modelo válido se promueve"
    delta = challenger_auc - champion_auc
    if delta >= min_improvement:
        return True, f"challenger mejora AUC en {delta:+.4f}"
    return False, f"mejora insuficiente ({delta:+.4f} < {min_improvement})"


def training_report(
    table: pd.DataFrame, best: str, result: dict[str, Any], importance: pd.Series, periods: dict[str, str]
) -> str:
    cols = ["algorithm", "test_roc_auc", "test_pr_auc", "test_ks", "test_brier", "latency_ms", "fit_diagnosis"]
    lines = [
        "# Informe de entrenamiento - Credit Risk Platform 2.0 (Lending Club)",
        "",
        f"**Modelo ganador:** `{best}`  ",
        f"**Umbral de decisión (costo FN/FP):** {result['threshold']:.2f}  ",
        f"**Diagnóstico de ajuste:** {result['fit_diagnosis']} (brecha AUC {result['auc_gap']:+.4f})",
        "",
        "## Periodos (validación out-of-time)",
        "",
        *[f"- **{k}:** {v}" for k, v in periods.items()],
        "",
        "## Benchmark de candidatos (mismo split temporal)",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, row in table.iterrows():
        lines.append(
            "| " + " | ".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |"
        )
    lines += ["", "## Train vs test OOT", "", "| métrica | train | test |", "|---|---|---|"]
    for m in ("roc_auc", "gini", "pr_auc", "ks", "brier", "ece", "recall", "precision"):
        lines.append(f"| {m} | {result[f'train_{m}']:.4f} | {result[f'test_{m}']:.4f} |")
    lines += ["", "## Variables más influyentes (|SHAP| medio)", ""]
    lines += [f"- `{name}`: {value:.4f}" for name, value in importance.head(12).items()]
    return "\n".join(lines) + "\n"
