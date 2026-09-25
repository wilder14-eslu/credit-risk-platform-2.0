import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import input_features, platform_config, target_name
from credit_risk.models.metrics import classification_metrics
from credit_risk.monitoring.ab_testing import assign_variant, bootstrap_auc_diff, evaluate_ab_test
from credit_risk.monitoring.concept_drift import concept_drift_report, ddm, page_hinkley, two_proportion_ztest
from credit_risk.monitoring.data_drift import build_reference_profile, feature_drift, psi
from credit_risk.monitoring.decision import retrain_decision
from credit_risk.monitoring.replay import monitoring_windows, months_between, next_months

CFG = platform_config()["monitoring"]


@pytest.fixture(scope="module")
def profile(splits, xgb_model):
    return build_reference_profile(splits.x_train, xgb_model.predict_proba(splits.x_train))


def test_psi_basics():
    e = np.array([0.25] * 4)
    assert psi(e, e) == pytest.approx(0)
    assert psi(e, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25


def test_profile_covers_numeric_and_categorical(profile):
    restored = json.loads(json.dumps(profile))
    assert restored["features"]["grade"]["kind"] == "categorical"
    assert restored["features"]["int_rate"]["kind"] == "numeric"
    assert "__score__" in restored["features"]


def test_real_temporal_drift_is_detected(profile, canonical, xgb_model, splits):
    """Las originaciones 2017-2018 (sintéticas con drift) se alejan del entrenamiento."""
    late = canonical[canonical["issue_month"] >= "2017-01-01"]
    cur = late[list(input_features())].copy()
    cur["__score__"] = xgb_model.predict_proba(late)
    table = feature_drift(profile, cur)
    same = splits.x_test.copy()
    same["__score__"] = xgb_model.predict_proba(splits.x_test)
    baseline = feature_drift(profile, same)
    assert table.set_index("feature").loc["grade", "psi"] > baseline.set_index("feature").loc["grade", "psi"]
    assert {"numeric", "categorical"} == set(table["kind"])


def test_categorical_psi_detects_mix_change(profile, splits):
    cur = splits.x_test.copy()
    cur["grade"] = "G"
    table = feature_drift(profile, cur).set_index("feature")
    assert table.loc["grade", "status"] == "alerta"


def test_sequential_detectors():
    rng = np.random.default_rng(0)
    stream = np.concatenate([rng.random(2000) < 0.05, rng.random(2000) < 0.3]).astype(int)
    assert ddm(stream).state == "drift"
    assert ddm((rng.random(3000) < 0.05).astype(int)).state != "drift"
    assert page_hinkley(np.concatenate([rng.normal(0.2, 0.05, 1000), rng.normal(1, 0.05, 1000)]), lamb=20)["drift"]
    z, p = two_proportion_ztest(150, 1000, 70, 1000)
    assert z > 0 and p < 0.001


def test_concept_drift_on_late_vintages(canonical, xgb_model, splits):
    reference = classification_metrics(splits.y_test, xgb_model.predict_proba(splits.x_test))
    test = canonical.loc[splits.x_test.index]
    assert not concept_drift_report(test[target_name()], xgb_model.predict_proba(test), reference, CFG)["concept_drift"]
    rng = np.random.default_rng(1)
    shocked = test.copy()
    shocked[target_name()] = np.where(
        rng.random(len(shocked)) < 0.5, 1 - shocked[target_name()], shocked[target_name()]
    )
    report = concept_drift_report(shocked[target_name()], xgb_model.predict_proba(shocked), reference, CFG)
    assert report["concept_drift"] and report["signals"]["auc_degradation"]


def test_retrain_policy_with_simulated_clock():
    drift = pd.DataFrame({"feature": ["a", "b", "c", "prediction"], "psi": [0.3, 0.4, 0.5, 0.05]})
    assert retrain_decision(drift, None, CFG, 5000, now=datetime(2016, 6, 1))["retrain"]
    cooled = retrain_decision(drift, None, CFG, 5000, last_retrain_at=datetime(2016, 5, 1), now=datetime(2016, 6, 1))
    assert not cooled["retrain"]
    assert retrain_decision(drift, None, CFG, 5000, last_retrain_at=datetime(2015, 1, 1), now=datetime(2016, 6, 1))[
        "retrain"
    ]
    assert retrain_decision(drift, None, CFG, 10)["severity"] == "sin_datos"


def test_replay_calendar():
    months = next_months(None, "2015-01-01", 3)
    assert [m.strftime("%Y-%m") for m in months] == ["2015-01", "2015-02", "2015-03"]
    assert next_months("2015-03-01", "2015-01-01", 2)[0] == pd.Timestamp("2015-04-01")
    assert next_months("2018-11-01", "2015-01-01", 3, last_available="2018-12-01") == [pd.Timestamp("2018-12-01")]
    win = monitoring_windows("2016-12-01", window_months=3, label_delay_months=6)
    assert win["drift_start"] == pd.Timestamp("2016-10-01") and win["label_end"] == pd.Timestamp("2016-06-01")
    assert months_between("2015-01-01", "2016-03-01") == 14


def _arm(rng, n, strength):
    y = (rng.random(n) < 0.15).astype(int)
    p = np.clip(0.1 + strength * y + rng.normal(0, 0.1, n), 0.001, 0.999)
    return {"y": y, "p": p, "approved": p < 0.3}


def test_ab_testing():
    cfg = platform_config()["ab_testing"]
    rng = np.random.default_rng(0)
    ids = [f"L{i}" for i in range(20000)]
    share = sum(assign_variant(i, 0.2) == "challenger" for i in ids) / len(ids)
    assert 0.18 < share < 0.22 and assign_variant("x", 0) == "champion"
    assert evaluate_ab_test(_arm(rng, 100, 0.2), _arm(rng, 100, 0.3), cfg)["decision"] == "continuar"
    assert evaluate_ab_test(_arm(rng, 4000, 0.1), _arm(rng, 4000, 0.3), cfg)["decision"] == "promover"
    assert evaluate_ab_test(_arm(rng, 4000, 0.3), _arm(rng, 4000, 0.05), cfg)["decision"] == "detener"
    a, b = _arm(rng, 3000, 0.2), _arm(rng, 3000, 0.2)
    res = bootstrap_auc_diff(a["y"], a["p"], b["y"], b["p"], iterations=200)
    assert res["ci_low"] < 0 < res["ci_high"]
