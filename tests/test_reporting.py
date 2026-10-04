"""Reporte estadístico: análisis puro, pipeline completo, figuras y sección del README."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import platform_config
from credit_risk.reporting import analysis as A
from credit_risk.reporting import figures, readme
from credit_risk.reporting.pipeline import build_results, json_safe


@pytest.fixture(scope="module")
def scored():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.6, 20_000)
    y = rng.binomial(1, p)  # perfectamente calibrado por construcción
    return y, p


def test_wilson_interval_contains_proportion():
    lo, hi = A._wilson(15, 100)
    assert lo < 0.15 < hi
    assert np.isnan(A._wilson(0, 0)[0])


def test_calibration_of_calibrated_scores(scored):
    y, p = scored
    cal = A.calibration(y, p)
    assert cal["slope"] == pytest.approx(1.0, abs=0.1)
    assert abs(cal["intercept"]) < 0.1
    assert cal["spiegelhalter_p"] > 0.01
    assert cal["brier"] < cal["brier_reference"]
    assert len(cal["table"]) == 10


def test_deciles_thresholds_curves_and_discrimination(scored):
    y, p = scored
    deciles = A.decile_table(y, p)
    rates = [d["default_rate"] for d in deciles]
    assert rates[0] > rates[-1] and deciles[-1]["cum_capture"] == pytest.approx(1.0)
    rows = A.threshold_curve(y, p, 5, 1)
    approvals = [r["approval_rate"] for r in rows]
    assert approvals == sorted(approvals)  # umbral más alto, más aprobación
    c = A.curves(y, p, points=50)
    assert len(c["fpr"]) <= 50 and c["tpr"][-1] == pytest.approx(1.0)
    d = A.discrimination(y, p, n_boot=50)
    assert d["auc_ci"][0] < d["auc"] < d["auc_ci"][1] and d["gini"] == pytest.approx(2 * d["auc"] - 1)


def test_vintage_stability_skips_small_quarters(scored):
    y, p = scored
    dates = pd.Series(pd.date_range("2013-01-01", periods=len(y), freq="h"))
    rows = A.vintage_stability(y, p, dates, min_defaults=30)
    assert rows and all(r["auc_ci"][0] <= r["auc"] <= r["auc_ci"][1] for r in rows)


def test_json_safe_handles_nan_and_numpy():
    out = json_safe({"a": np.float64(1.5), "b": [float("nan"), np.int64(2)], "c": (1, 2)})
    assert out == {"a": 1.5, "b": [None, 2], "c": [1, 2]}
    json.dumps(out)


@pytest.fixture(scope="module")
def results(canonical):
    training = platform_config()["training"]
    original = training["candidates"]
    training["candidates"] = ["logistic_regression", "xgboost"]
    try:
        return json_safe(build_results(canonical, optuna_trials=0))
    finally:
        training["candidates"] = original


def test_build_results_is_complete(results):
    expected = {"periods", "sizes", "benchmark", "selection", "comparison", "final", "curves", "scorecard",
                "vintages", "shap", "shap_dependence", "ablation"}  # fmt: skip
    assert expected <= set(results)
    assert results["final"]["algorithm"] in {"logistic_regression", "xgboost"}
    assert results["ablation"]["dropped"] == ["grade", "sub_grade", "int_rate"]
    assert results["vintages"] and results["shap"]
    json.dumps(results)


def test_figures_and_readme_section(results, tmp_path):
    files = figures.render_all(results, tmp_path)
    assert set(files) >= {"auc_ci", "pairwise", "roc_pr", "calibration", "deciles", "threshold", "shap", "iv"}
    assert all((tmp_path / name).stat().st_size > 5_000 for name in files.values())
    section = readme.render({**results, "generated_at": "hoy", "data_file": "x.csv"}, files)
    assert section.startswith(readme.START) and section.endswith(readme.END)
    for title in ("Resumen de la decisión", "Por qué este modelo", "Selección estadística", "Calibración",
                  "Umbral de decisión", "Explicabilidad", "Ablación"):  # fmt: skip
        assert title in section
    assert not re.search(r"\bnan\b", section.lower())
    doc = f"# T\n\n{readme.START}\nviejo\n{readme.END}\n\nfin\n"
    updated = readme.update_readme(doc, section)
    assert "viejo" not in updated and updated.endswith("fin\n") and updated.count(readme.START) == 1
    with pytest.raises(ValueError):
        readme.update_readme("# sin marcadores", section)
