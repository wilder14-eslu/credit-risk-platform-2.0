import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import input_features, target_name
from credit_risk.features.engineering import OTHER, Preprocessor, build_features
from credit_risk.models import metrics as M
from credit_risk.models import training as T
from credit_risk.models.candidates import build_estimator
from credit_risk.models.credit_model import risk_band


def test_build_features_derived(canonical):
    feats = build_features(canonical)
    row, src = feats.iloc[0], canonical.iloc[0]
    assert row["loan_to_income"] == pytest.approx(src["loan_amnt"] / src["annual_inc"])
    assert row["installment_to_income"] == pytest.approx(src["installment"] * 12 / src["annual_inc"])
    with pytest.raises(ValueError):
        build_features(canonical.drop(columns=["grade"]))


def test_preprocessor_handles_unseen_and_missing(canonical):
    feats = build_features(canonical)
    pre = Preprocessor().fit(feats)
    new = feats.head(3).copy()
    new.loc[new.index[0], "purpose"] = "time_machine"
    new.loc[new.index[1], "emp_length_years"] = np.nan
    x = pre.transform(new)
    assert list(x.columns) == pre.columns_
    assert x.loc[new.index[0], f"purpose={OTHER}"] == 1.0
    assert not x.isna().any().any()
    assert Preprocessor.source_feature("grade=B") == "grade"


def test_time_based_split_has_no_leakage(splits, canonical):
    dates = canonical.set_index(canonical.index)["issue_month"]
    assert dates.loc[splits.x_train.index].max() < dates.loc[splits.x_val.index].min()
    assert dates.loc[splits.x_val.index].max() < dates.loc[splits.x_test.index].min()
    assert list(splits.x_train.columns) == list(input_features())


def test_split_config_moves_with_as_of():
    base = T.split_config()
    moved = T.split_config("2016-12-01")
    assert moved["test_end"] > base["test_end"]
    assert moved["train_end"] < moved["validation_end"] < moved["test_end"]


def test_benchmark_and_gates(splits):
    table, models = T.run_benchmark(splits, ["logistic_regression", "xgboost"])
    assert table["test_roc_auc"].is_monotonic_decreasing
    best = T.select_best(table)
    assert best in models
    passed, reasons = T.check_quality_gates(table[table.algorithm == best].iloc[0].to_dict())
    assert passed, reasons
    weak = {"test_roc_auc": 0.5, "auc_gap": 0.3, "test_brier": 0.4, "latency_ms": 99}
    assert len(T.check_quality_gates(weak)[1]) == 4


def test_champion_vs_challenger():
    assert T.champion_vs_challenger(0.70, None)[0]
    assert T.champion_vs_challenger(0.705, 0.70)[0]
    assert not T.champion_vs_challenger(0.701, 0.70)[0]


def test_optuna(splits):
    assert "C" in T.tune("logistic_regression", splits, n_trials=2, timeout=60)


def test_predict_frame_and_explanations(xgb_model, splits):
    frame = xgb_model.predict_frame(splits.x_test.head(10))
    assert frame["probability"].between(0, 1).all()
    factors = json.loads(frame["top_factors"].iloc[0])
    assert len(factors) == 3 and all(
        f["feature"]
        in [
            *input_features(),
            "loan_to_income",
            "installment_to_income",
            "revol_bal_to_income",
            "open_acc_ratio",
            "log_annual_inc",
        ]
        for f in factors
    )


def test_logistic_contributions_sum_to_logit(lr_model, splits):
    x = splits.x_test.head(30)
    p = lr_model.predict_proba(x)
    intercept = lr_model.estimator.named_steps["clf"].intercept_[0]
    np.testing.assert_allclose(lr_model.contributions(x).sum(axis=1) + intercept, np.log(p / (1 - p)), atol=1e-6)


def test_training_report(splits, lr_model):
    table, _ = T.run_benchmark(splits, ["logistic_regression"])
    md = T.training_report(
        table,
        "logistic_regression",
        T.evaluate(lr_model, splits),
        lr_model.global_importance(splits.x_test.head(100)),
        splits.periods,
    )
    assert "out-of-time" in md and "Benchmark" in md


def test_metrics_basics():
    y = np.array([0, 0, 1, 1])
    m = M.classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["ks"] == 1.0
    assert M.diagnose_fit(0.95, 0.70)["fit_diagnosis"] == "sobreajuste"


@pytest.mark.parametrize("p,band", [(0.01, "A"), (0.07, "B"), (0.15, "C"), (0.3, "D"), (0.5, "E")])
def test_risk_band(p, band):
    assert risk_band(p) == band


def test_unknown_algorithm():
    with pytest.raises(ValueError):
        build_estimator("svm")


def test_pyfunc_roundtrip(tmp_path, lr_model, splits):
    mlflow = pytest.importorskip("mlflow")
    import joblib
    from mlflow.models import infer_signature

    from credit_risk.config import PROJECT_ROOT
    from credit_risk.models.credit_model import CreditRiskPyfunc

    sample = splits.x_test.head(5).reset_index(drop=True)
    joblib.dump(lr_model, tmp_path / "m.joblib")
    mlflow.pyfunc.save_model(
        path=str(tmp_path / "pyfunc"),
        python_model=CreditRiskPyfunc(),
        artifacts={"credit_model": str(tmp_path / "m.joblib")},
        code_paths=[str(PROJECT_ROOT / "src" / "credit_risk")],
        signature=infer_signature(sample, lr_model.predict_frame(sample), params={"explain": True}),
    )
    out = mlflow.pyfunc.load_model(str(tmp_path / "pyfunc")).predict(sample, params={"explain": False})
    assert isinstance(out, pd.DataFrame) and (out["top_factors"] == "[]").all()
    assert target_name() not in out
