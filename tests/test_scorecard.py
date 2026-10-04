"""Scorecard WoE: binning numérico y categórico, IV, puntos y benchmark en el pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk.models import training as T
from credit_risk.models.scorecard import (
    MISSING_CODE,
    OTHER,
    CreditScorecard,
    fit_binning,
    fit_categorical_binning,
    information_value_label,
)


def _synthetic(n: int = 8000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    util = rng.uniform(0, 1.5, n)
    fico = rng.normal(690, 40, n)
    grade = rng.choice(list("ABCDE"), n, p=[0.3, 0.3, 0.2, 0.15, 0.05])
    rare = rng.choice(["x", "y", "z"], n, p=[0.6, 0.395, 0.005])
    noise = rng.normal(size=n)
    logit = (
        -2.5 + 1.8 * util - 0.01 * (fico - 690) + 0.5 * pd.Series(grade).map(dict(zip("ABCDE", range(5)))).to_numpy()
    )
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    x = pd.DataFrame({"util": util, "fico": fico, "grade": grade, "rare": rare, "noise": noise})
    x.loc[rng.choice(n, 300, replace=False), "fico"] = np.nan
    x.loc[rng.choice(n, 200, replace=False), "grade"] = None
    return x, y


def test_numeric_woe_sign_follows_risk_and_missing_bin():
    x, y = _synthetic()
    util = fit_binning("util", x["util"], y)
    woes = [r["woe"] for r in util.table if r["code"] < MISSING_CODE]
    assert woes[0] > 0 > woes[-1]  # baja utilización = menor riesgo = WoE positivo
    assert util.iv > 0.1
    fico = fit_binning("fico", x["fico"], y)
    assert any(r["bin"] == "faltante" for r in fico.table)


def test_categorical_binning_orders_risk_and_groups_rare_levels():
    x, y = _synthetic()
    grade = fit_categorical_binning("grade", x["grade"], y)
    woe = {r["bin"]: r["woe"] for r in grade.table}
    assert woe["A"] > woe["C"] > woe["E"]
    assert "faltante" in woe
    rare = fit_categorical_binning("rare", x["rare"], y)
    assert OTHER in rare.levels and "z" not in rare.levels
    codes = rare.assign(pd.Series(["x", "nunca_visto", None]))
    assert codes[1] == rare.levels.index(OTHER) and codes[2] == MISSING_CODE


def test_scorecard_ranks_iv_scores_and_points():
    x, y = _synthetic()
    sc = CreditScorecard().fit(x, y)
    ivs = {r["feature"]: r["iv"] for r in sc.iv_table()}
    assert ivs["util"] > ivs["noise"] and ivs["noise"] < 0.02
    pd_hat = sc.predict_pd(x)
    assert roc_auc_score(y, pd_hat) > 0.7
    scores = sc.score(x)
    assert np.corrcoef(scores, pd_hat)[0, 1] < 0  # más puntos = menos riesgo
    np.testing.assert_allclose(sc.score_to_pd(sc.pd_to_score([0.02, 0.2])), [0.02, 0.2], rtol=1e-6)
    assert sc.pd_to_score([1 / 51])[0] == pytest.approx(600.0)  # 600 puntos = odds 50:1
    assert {"points", "coefficient"} <= set(sc.points_table()[0])


def test_scorecard_requires_fit_and_iv_labels():
    with pytest.raises(RuntimeError):
        CreditScorecard().predict_pd(pd.DataFrame({"a": [1.0]}))
    assert information_value_label(0.01) == "inútil"
    assert information_value_label(0.2) == "medio"
    assert "fuga" in information_value_label(0.8)


def test_selection_and_test_comparison_in_pipeline(splits):
    table, models = T.run_benchmark(splits, ["logistic_regression", "xgboost"])
    assert {"val_roc_auc", "val_auc_gap"} <= set(table.columns)
    selection = T.statistical_selection(table, models, splits)
    assert selection["selected"] in models
    assert sum(r["selected"] for r in selection["rows"]) == 1
    scorecard = T.fit_scorecard(splits)
    comparison = T.compare_on_test(models, splits, scorecard)
    names = {r["model"] for r in comparison["auc"]}
    assert names == {"logistic_regression", "xgboost", "scorecard_woe"}
    assert len(comparison["pairwise"]) == 3
    assert all(r["ci_low"] <= r["auc"] <= r["ci_high"] for r in comparison["auc"])
    best, *_ = T.choose_final_model(table, models, splits, 0, 10, best=selection["selected"])
    assert best in models
    md = T.training_report(
        table,
        best,
        T.evaluate(models[best], splits),
        models[best].global_importance(splits.x_test.head(100)),
        splits.periods,
        selection=selection,
        comparison=comparison,
        iv=scorecard.iv_table(),
    )
    assert "Selección estadística" in md and "DeLong + Holm" in md and "Information Value" in md
