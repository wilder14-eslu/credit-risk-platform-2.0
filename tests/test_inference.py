"""Inferencia para comparar y seleccionar modelos: DeLong, Holm, significancia práctica, parsimonia."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk.inference.comparison import (
    auc_table,
    holm_adjust,
    interpret_difference,
    pairwise_delong_comparison,
    select_parsimonious,
)
from credit_risk.inference.delong import bootstrap_ci, delong_auc_ci, delong_roc_test


@pytest.fixture(scope="module")
def scored():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.15, 6000)
    strong = y * 1.2 + rng.normal(0, 1, len(y))
    twin = strong + rng.normal(0, 0.05, len(y))  # prácticamente el mismo modelo
    weak = y * 0.4 + rng.normal(0, 1, len(y))
    return y, strong, twin, weak


def test_delong_auc_matches_sklearn_and_ci_contains_it(scored):
    y, strong, _, _ = scored
    result = delong_auc_ci(y, strong)
    assert result["auc"] == pytest.approx(roc_auc_score(y, strong), abs=1e-10)
    assert result["ci_low"] < result["auc"] < result["ci_high"]
    assert result["se"] > 0


def test_delong_test_detects_difference_and_not_self(scored):
    y, strong, _, weak = scored
    assert delong_roc_test(y, strong, weak)["p_value"] < 1e-6
    same = delong_roc_test(y, strong, strong)
    assert same["p_value"] == pytest.approx(1.0) and same["diff"] == pytest.approx(0.0)


def test_delong_requires_both_classes():
    with pytest.raises(ValueError):
        delong_auc_ci(np.zeros(10), np.arange(10))


def test_bootstrap_ci_contains_estimate(scored):
    y, strong, _, _ = scored
    ci = bootstrap_ci(y, strong, roc_auc_score, n_boot=200)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"]
    assert ci["n_boot"] == 200


def test_holm_known_values_order_and_validation():
    # Ordenados: 0.001*3=0.003, 0.03*2=0.06, 0.04*1=0.04 -> máximo acumulado 0.06
    assert holm_adjust([0.04, 0.001, 0.03]) == pytest.approx([0.06, 0.003, 0.06])
    assert holm_adjust([]) == []
    with pytest.raises(ValueError):
        holm_adjust([0.2, 1.5])
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.3, 10)
    adjusted = holm_adjust(p)
    assert all(a >= raw for a, raw in zip(adjusted, p, strict=True))
    assert all(a <= min(1.0, raw * len(p)) + 1e-12 for a, raw in zip(adjusted, p, strict=True))


def test_interpretation_separates_statistical_from_practical():
    assert "irrelevante" in interpret_difference(0.001, 0.001)
    assert "material a favor de A" in interpret_difference(0.02, 0.001)
    assert "material a favor de B" in interpret_difference(-0.02, 0.001)
    assert "equivalentes" in interpret_difference(0.001, 0.5)
    assert "no concluyente" in interpret_difference(0.02, 0.5)


def test_pairwise_comparison_has_all_pairs_and_holm(scored):
    y, strong, twin, weak = scored
    rows = pairwise_delong_comparison(y, {"strong": strong, "twin": twin, "weak": weak})
    assert len(rows) == 3
    assert all(r["p_holm"] >= r["p_value"] for r in rows)
    sw = next(r for r in rows if (r["model_a"], r["model_b"]) == ("strong", "weak"))
    assert sw["ci_low"] > 0 and "material a favor de A" in sw["interpretation"]
    st = next(r for r in rows if (r["model_a"], r["model_b"]) == ("strong", "twin"))
    assert abs(st["delta"]) < 0.005
    table = auc_table(y, {"weak": weak, "strong": strong})
    assert [r["model"] for r in table] == ["strong", "weak"]


COMPLEXITY = {"simple": 1, "complex": 2, "complex_b": 2}


def test_parsimony_keeps_complex_model_when_materially_better(scored):
    y, strong, _, weak = scored
    result = select_parsimonious(y, {"complex": strong, "simple": weak}, COMPLEXITY)
    assert result["selected"] == "complex"
    simple = next(r for r in result["rows"] if r["model"] == "simple")
    assert not simple["retained"] and simple["p_holm"] < 0.05


def test_parsimony_prefers_simple_model_when_equivalent(scored):
    y, strong, twin, _ = scored
    # El complejo es apenas mejor: diferencia sin relevancia práctica -> gana el simple.
    result = select_parsimonious(y, {"complex": strong, "simple": twin}, COMPLEXITY)
    assert result["selected"] == "simple"
    assert "más simple" in result["reason"]


def test_parsimony_ignores_ineligible_and_breaks_ties_by_auc(scored):
    y, strong, twin, weak = scored
    scores = {"complex": twin, "complex_b": strong, "simple": weak}
    result = select_parsimonious(y, scores, COMPLEXITY, eligible=["complex", "complex_b"])
    assert result["selected"] in {"complex", "complex_b"}
    simple = next(r for r in result["rows"] if r["model"] == "simple")
    assert not simple["eligible"] and not simple["retained"] and np.isnan(simple["p_holm"])
    assert sum(r["selected"] for r in result["rows"]) == 1
    with pytest.raises(ValueError):
        select_parsimonious(y, scores, COMPLEXITY, eligible=["unknown"])
