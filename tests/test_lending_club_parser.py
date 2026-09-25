import numpy as np
import pandas as pd
import pytest

from credit_risk.config import data_schema, input_features, target_name
from credit_risk.data.lending_club import canonical_columns, label_from_status, parse_raw, spark_schema_ddl
from credit_risk.data.quality import assert_quality, clean, run_expectations

RAW_ROW = {
    "id": "68407277",
    "loan_amnt": "3600.0",
    "term": " 36 months",
    "int_rate": "13.99",
    "installment": "123.03",
    "grade": "C",
    "sub_grade": "C4",
    "emp_length": "10+ years",
    "home_ownership": "MORTGAGE",
    "annual_inc": "55000.0",
    "verification_status": "Not Verified",
    "issue_d": "Dec-2015",
    "loan_status": "Fully Paid",
    "purpose": "debt_consolidation",
    "addr_state": "PA",
    "dti": "5.91",
    "delinq_2yrs": "0.0",
    "earliest_cr_line": "Aug-2003",
    "fico_range_low": "675.0",
    "fico_range_high": "679.0",
    "inq_last_6mths": "1.0",
    "open_acc": "7.0",
    "pub_rec": "0.0",
    "revol_bal": "2765.0",
    "revol_util": "29.7",
    "total_acc": "13.0",
    "initial_list_status": "w",
    "application_type": "Individual",
    "mort_acc": "1.0",
    "pub_rec_bankruptcies": "0.0",
    "total_pymnt": "4421.72",
}


def test_parses_a_real_lending_club_row():
    out = parse_raw(pd.DataFrame([RAW_ROW]))
    row = out.iloc[0]
    assert row["loan_id"] == "68407277"
    assert row["issue_month"] == pd.Timestamp("2015-12-01")
    assert row["term_months"] == 36
    assert row["emp_length_years"] == 10
    assert row["fico_score"] == pytest.approx(677.0)
    assert row["credit_history_months"] == (2015 - 2003) * 12 + (12 - 8)
    assert row[target_name()] == 0.0
    assert "total_pymnt" not in out.columns  # leakage fuera


def test_handles_percent_strings_and_special_values():
    row = {
        **RAW_ROW,
        "int_rate": "13.99%",
        "revol_util": "29.7%",
        "emp_length": "< 1 year",
        "home_ownership": "ANY",
        "loan_status": "Charged Off",
    }
    out = parse_raw(pd.DataFrame([row])).iloc[0]
    assert out["int_rate"] == pytest.approx(13.99) and out["revol_util"] == pytest.approx(29.7)
    assert out["emp_length_years"] == 0 and out["home_ownership"] == "OTHER"
    assert out[target_name()] == 1.0


def test_drops_footer_rows(raw_lc):
    assert raw_lc["loan_amnt"].isna().sum() == 1
    assert len(parse_raw(raw_lc)) == len(raw_lc) - 1


def test_label_mapping():
    status = pd.Series(
        [
            "Fully Paid",
            "Charged Off",
            "Current",
            "Default",
            "Late (31-120 days)",
            "Does not meet the credit policy. Status:Charged Off",
        ]
    )
    labels = label_from_status(status)
    assert labels.tolist()[:2] == [0.0, 1.0]
    assert np.isnan(labels[2]) and labels[3] == 1.0 and np.isnan(labels[4]) and labels[5] == 1.0


def test_schema_contract():
    out_cols = canonical_columns()
    assert set(input_features()) <= set(out_cols)
    assert len(spark_schema_ddl().split(", ")) == len(out_cols)
    assert not set(data_schema()["leakage_columns"]) & set(out_cols)


def test_quality_gate_passes_and_blocks(canonical):
    assert_quality(run_expectations(canonical))
    broken = canonical.copy()
    broken["fico_score"] = np.nan
    with pytest.raises(ValueError, match="calidad"):
        assert_quality(run_expectations(broken))
    dup = canonical.copy()
    dup.loc[dup.index[:5], "loan_id"] = dup["loan_id"].iloc[0]
    assert any(r.name == "unique_id" and not r.passed for r in run_expectations(dup))


def test_quality_detects_leakage_columns(canonical):
    leaky = canonical.assign(total_pymnt=1.0)
    assert any(r.name == "no_leakage_columns" and not r.passed for r in run_expectations(leaky))


def test_clean_removes_impossible_values(canonical):
    bad = canonical.copy()
    bad.loc[bad.index[0], "fico_score"] = 100
    assert len(clean(bad)) == len(canonical) - 1
