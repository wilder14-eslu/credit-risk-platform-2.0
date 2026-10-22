"""Casos borde y ramas defensivas: cada línea del código medido queda ejecutada y verificada."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from credit_risk import config as C
from credit_risk.config import platform_config
from credit_risk.data.lending_club import parse_batches, parse_raw
from credit_risk.data.quality import Expectation, assert_quality
from credit_risk.features.engineering import Preprocessor, build_features
from credit_risk.inference.delong import bootstrap_ci
from credit_risk.models import candidates as K
from credit_risk.models import training as T
from credit_risk.models.metrics import diagnose_fit, ks_statistic
from credit_risk.models.scorecard import FeatureBinning, _candidate_edges
from credit_risk.monitoring.ab_testing import bootstrap_auc_diff, evaluate_ab_test
from credit_risk.monitoring.concept_drift import two_proportion_ztest
from credit_risk.monitoring.data_drift import _edges
from credit_risk.monitoring.decision import retrain_decision
from credit_risk.reporting import analysis as A
from credit_risk.reporting import figures, readme
from credit_risk.reporting.pipeline import build_results, json_safe

# --- Configuración y datos ---------------------------------------------------------------------


def test_set_config_dir_and_volume_path():
    original = C.CONFIG_DIR
    try:
        C.set_config_dir(original)
        assert C.data_schema()["target"] == C.target_name()
    finally:
        C.set_config_dir(original)
    names = C.UCNames(catalog="workspace", schema="credit_risk")
    assert names.volume_path == "/Volumes/workspace/credit_risk"


def test_parse_batches_adapter_for_spark(raw_lc):
    batches = list(parse_batches([raw_lc.head(50), raw_lc.iloc[50:80]]))
    assert [len(b) for b in batches] == [len(parse_raw(raw_lc.head(50))), len(parse_raw(raw_lc.iloc[50:80]))]


def test_quality_warnings_do_not_block(caplog):
    warning = Expectation("nulls", "dti", passed=False, observed=0.2, threshold=0.1, severity="warning")
    assert warning.to_dict()["severity"] == "warning"
    assert_quality([warning])  # un warning se registra pero no detiene el pipeline
    assert "Expectativa en warning" in caplog.text


def test_feature_label_falls_back_to_the_name():
    from credit_risk.features.engineering import feature_label

    assert feature_label("variable_desconocida") == "variable_desconocida"


def test_unfitted_preprocessor_fails_loudly(canonical):
    with pytest.raises(RuntimeError, match="no está ajustado"):
        Preprocessor().transform(build_features(canonical.head(5)))


# --- Métricas, inferencia y scorecard ----------------------------------------------------------


def test_metric_edge_cases():
    assert math.isnan(ks_statistic(np.zeros(5), np.linspace(0, 1, 5)))
    assert diagnose_fit(0.60, 0.59)["fit_diagnosis"] == "subajuste"
    tiny = bootstrap_ci([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], lambda y, s: float(np.mean(s[y == 1])), n_boot=200)
    assert tiny["ci_low"] <= tiny["ci_high"]  # los remuestreos de una sola clase se omiten


def test_scorecard_binning_edges():
    assert _candidate_edges(np.array([3.0, 3.0, 3.0]), max_bins=5) == []
    flat = FeatureBinning(name="x", kind="numeric", edges=[])
    assert flat.describe(0) == "todos"


# --- Candidatos y entrenamiento ----------------------------------------------------------------


@pytest.mark.parametrize("name", K.KNOWN)
def test_search_spaces_and_estimators(name):
    import optuna

    trial = optuna.create_study().ask()
    params = K.suggest_params(trial, name)
    assert params and set(params) <= set(trial.params)
    assert K.build_estimator(name, K.default_params(name)) is not None


def test_unknown_candidates_and_missing_dependencies(monkeypatch, caplog):
    import optuna

    with pytest.raises(ValueError):
        K.suggest_params(optuna.create_study().ask(), "svm")
    with pytest.raises(ValueError):
        K.build_estimator("svm", {})
    monkeypatch.setattr(K, "is_available", lambda name: name != "catboost")
    assert K.available_candidates(["xgboost", "catboost"]) == ["xgboost"]
    assert "catboost omitido" in caplog.text


@pytest.mark.parametrize(
    ("name", "params"),
    [("lightgbm", {"n_estimators": 60}), ("catboost", {"iterations": 60})],
)
def test_tree_contributions_add_up_to_the_logit(name, params, splits):
    """SHAP exacto de LightGBM (el champion en producción) y CatBoost: suma + base = log-odds."""
    model = T.fit_model(name, params, splits)
    x = splits.x_test.head(40)
    contrib = model.contributions(x)
    p = np.clip(model.predict_proba(x), 1e-9, 1 - 1e-9)
    logit = np.log(p / (1 - p))
    base = logit - contrib.sum(axis=1).to_numpy()
    assert np.allclose(base, base[0], atol=1e-4)  # el valor base es el mismo para todas las filas


def test_split_caps_train_rows_and_rejects_empty_periods(canonical):
    cfg = {**platform_config()["data"], "max_train_rows": 500}
    splits = T.make_splits(canonical, cfg)
    assert len(splits.y_train) == 500
    with pytest.raises(ValueError, match="no tiene datos etiquetados"):
        T.make_splits(canonical, {**platform_config()["data"], "test_end": platform_config()["data"]["validation_end"]})


def test_choose_final_model_when_no_candidate_passes_the_gates(splits, monkeypatch):
    table, models = T.run_benchmark(splits, ["logistic_regression"])
    monkeypatch.setattr(T, "check_quality_gates", lambda result, gates=None: (False, ["gate"]))
    name, _, _, origin = T.choose_final_model(table, models, splits, n_trials=0, timeout=60)
    assert name == "logistic_regression" and origin == "default"  # el job falla luego con un motivo claro


def test_pyfunc_switches_to_packaged_config(monkeypatch, tmp_path, lr_model):
    import joblib

    from credit_risk.models.credit_model import CreditRiskPyfunc

    model_file = tmp_path / "m.joblib"
    joblib.dump(lr_model, model_file)
    switched = []
    monkeypatch.setattr(C, "CONFIG_DIR", tmp_path / "no-existe")
    monkeypatch.setattr(C, "set_config_dir", switched.append)
    wrapper = CreditRiskPyfunc()
    wrapper.load_context(SimpleNamespace(artifacts={"credit_model": str(model_file), "config": "/model/config"}))
    assert switched == ["/model/config"] and wrapper.model.algorithm == lr_model.algorithm


# --- Monitoreo y A/B ---------------------------------------------------------------------------


def test_proportion_test_and_drift_edges():
    assert all(math.isnan(v) for v in two_proportion_ztest(0, 0, 1, 1))
    assert two_proportion_ztest(0, 10, 0, 10) == (0.0, 1.0)
    assert _edges(pd.Series([np.nan, np.nan]), bins=10) == [-np.inf, np.inf]


def test_retrain_on_prediction_drift_and_concept_signals():
    cfg = platform_config()["monitoring"]
    drift = pd.DataFrame({"feature": ["prediction", "dti"], "psi": [0.5, 0.01]})
    concept = {"signals": {"ddm": True, "page_hinkley": False}, "concept_drift": True}
    out = retrain_decision(drift, concept, cfg, n_rows=5000)
    assert out["retrain"] and out["severity"] == "critico"
    assert any("prediction drift" in r for r in out["reasons"]) and "concept drift: ddm" in out["reasons"]


def _arm(rng, n, quality, approve_rate=0.8):
    y = rng.binomial(1, 0.15, n)
    p = np.clip(0.15 + quality * (y - 0.15) + rng.normal(0, 0.1, n), 0.001, 0.999)
    return {"y": y, "p": p, "approved": rng.random(n) < approve_rate}


def test_ab_decisions_cover_every_outcome():
    cfg = {**platform_config()["ab_testing"], "bootstrap_iterations": 200}
    rng = np.random.default_rng(0)
    worse = evaluate_ab_test(_arm(rng, 4000, 0.6), _arm(rng, 4000, 0.05), cfg)
    assert worse["decision"] == "detener" and "inferior" in worse["reason"]
    same_a, same_b = _arm(rng, 4000, 0.3), _arm(rng, 4000, 0.3)
    closed = evaluate_ab_test(same_a, same_b, cfg, months_running=cfg["max_test_months"])
    running = evaluate_ab_test(same_a, same_b, cfg, months_running=1)
    if closed["decision"] == "detener" and "cierre" in closed["reason"]:
        assert running["decision"] == "continuar" and running["reason"] == "aún sin significancia"
    tiny = bootstrap_auc_diff([0, 1], [0.2, 0.8], [0, 1], [0.3, 0.7], iterations=50)
    assert set(tiny) == {"auc_diff", "ci_low", "ci_high", "p_value"}  # remuestreos de una clase se omiten


# --- Análisis estadístico y reporte ------------------------------------------------------------


def test_calibration_skips_empty_bins_and_small_vintages():
    # Cuantiles interpolados crean un bin sin observaciones (entre 0.16 y 0.2): se omite de la tabla.
    p = np.r_[np.full(400, 0.1), np.full(200, 0.2), np.full(400, 0.3)]
    y = (np.random.default_rng(1).random(1000) < p).astype(int)
    cal = A.calibration(y, p, bins=10)
    assert [r["n"] for r in cal["table"]] == [400, 200, 400]
    dates = pd.Series(pd.to_datetime(["2013-01-15"] * 990 + ["2013-07-15"] * 10))
    rows = A.vintage_stability(y, p, dates, min_defaults=30)
    assert [r["quarter"] for r in rows] == ["2013Q1"]  # el trimestre con 10 préstamos se omite


def test_phase_of_ignores_malformed_periods():
    assert figures._phase_of("2013Q1", {"val": "sin fechas", "test": "2013-01 a 2013-06 (n)"}) == "test"


@pytest.fixture(scope="module")
def tuned_results(canonical):
    """Corrida donde gana xgboost con Optuna: ejercita tuning, curvas vs logística y su sección del reporte."""
    training = platform_config()["training"]
    original = training["candidates"]
    training["candidates"] = ["logistic_regression", "xgboost"]
    real_selection, real_gates = T.statistical_selection, T.check_quality_gates
    T.statistical_selection = lambda table, models, splits: {
        **real_selection(table, models, splits),
        "selected": "xgboost",
    }
    T.check_quality_gates = lambda result, gates=None: (True, [])  # datos sintéticos: el gate de brecha no aplica
    try:
        return json_safe(build_results(canonical, optuna_trials=1))
    finally:
        training["candidates"] = original
        T.statistical_selection, T.check_quality_gates = real_selection, real_gates


def test_results_with_tuning_and_report_variants(tuned_results, tmp_path):
    r = tuned_results
    assert r["final"]["algorithm"] == "xgboost" and r["final"]["origin"] == "optuna"
    assert set(r["tuning"]) == {"validation", "test"} and "logistic_regression" in r["curves"]

    # Variantes del texto: empates, descartes, calibración adecuada y sin figuras opcionales.
    rows = [dict(row) for row in r["selection"]["rows"]]
    for row in rows:
        row.update(eligible=True, retained=row["model"] == "xgboost")
    tied = {**rows[0], "model": "lightgbm", "eligible": True, "retained": True, "p_holm": 0.5}
    variant = {
        **r,
        "selection": {**r["selection"], "rows": [*rows, tied]},
        "final": {
            **r["final"],
            "calibration": {**r["final"]["calibration"], "spiegelhalter_p": 0.5, "intercept": 0.0},
        },
        "shap_dependence": {},
        "vintages": [],
        "generated_at": "hoy",
        "data_file": "x.csv",
    }
    files = figures.render_all(variant, tmp_path)
    assert "shap_dependence" not in files and "vintages" not in files
    section = readme.render(variant, files)
    assert "Efecto del tuning con Optuna" in section and "Calibración adecuada" in section
    assert "Empate práctico" in section and "queda descartado en validación" in section


def test_pairwise_lookup_and_readings():
    results = {
        "comparison": {"pairwise": [{"model_a": "a", "model_b": "b", "delta": 0.01, "ci_low": 0.0, "ci_high": 0.02}]}
    }
    flipped = readme._pair(results, "b", "a")
    assert flipped["delta"] == -0.01 and flipped["ci_low"] == -0.02 and flipped["ci_high"] == -0.0
    assert readme._pair(results, "a", "z") is None
    assert readme._reading(0.01, 0.01, 0.005) == "diferencia significativa y material"
    assert readme._reading(0.001, 0.01, 0.005) == "significativa pero prácticamente irrelevante"
    assert readme._reading(0.01, 0.5, 0.005) == "de tamaño relevante pero no concluyente"
    assert readme._reading(0.001, 0.5, 0.005) == "sin evidencia de diferencia"
