"""Capa de política: el modelo estima la PD, la política decide (determinista y versionada)."""

import math

import pytest

from credit_risk import policy as P
from credit_risk.models.credit_model import risk_band

APP = {"annual_inc": 60000.0, "installment": 400.0, "dti": 18.0}
FACTORS = [
    {"feature": "int_rate", "impact": 0.40},
    {"feature": "fico_score", "impact": -0.10},
    {"feature": "dti", "impact": 0.15},
]


@pytest.mark.parametrize(
    ("p", "expected", "first_reason"),
    [
        (0.05, P.APPROVE, None),
        (0.159, P.APPROVE, None),
        (0.16, P.REVIEW, P.GREY_ZONE),  # [0.18 - 0.02, 0.18)
        (0.179, P.REVIEW, P.GREY_ZONE),
        (0.18, P.REJECT, P.PD_HIGH),
        (0.60, P.REJECT, P.PD_HIGH),
    ],
)
def test_decide_threshold_and_grey_zone(p, expected, first_reason):
    d = P.decide(p, 0.18, APP, FACTORS)
    assert d.decision == expected and d.policy_version == P.POLICY_VERSION
    if first_reason is None:
        assert d.reasons == []
    else:
        assert d.reasons[0] == first_reason
        assert d.reasons[1:] == ["RIESGO_INT_RATE", "RIESGO_DTI"]  # solo factores que suben el riesgo, por impacto


@pytest.mark.parametrize(
    ("app", "reason"),
    [
        ({**APP, "installment": 2500.0}, P.PAYMENT_TO_INCOME),  # 2500 / 5000 = 50 % del ingreso mensual
        ({**APP, "dti": 50.0}, P.DTI_HIGH),
    ],
)
def test_knockout_rules_win_over_a_low_pd(app, reason):
    d = P.decide(0.02, 0.18, app)
    assert d.decision == P.REJECT and d.reasons[0] == reason


def test_reason_codes_are_capped():
    many = [{"feature": f"f{i}", "impact": 1.0 + i} for i in range(10)]
    assert len(P.decide(0.9, 0.18, APP, many).reasons) == P.POLICY["max_reason_codes"]


def test_decision_contract():
    with pytest.raises(ValueError):
        P.Decision(P.REJECT, [])  # un rechazo sin motivo no es auditable
    with pytest.raises(ValueError):
        P.Decision("TAL_VEZ", ["x"])
    assert P.fallback().decision == P.REVIEW and P.fallback().reasons == [P.MODEL_UNAVAILABLE]


@pytest.mark.parametrize(
    ("p", "band", "threshold", "model_decision"),
    [
        (1.5, None, 0.18, None),
        (-0.1, None, 0.18, None),
        (math.nan, None, 0.18, None),
        ("abc", None, 0.18, None),
        (0.10, "A", 0.18, None),  # 0.10 es banda C
        (0.30, None, 0.18, "APROBAR"),  # el modelo aprueba con PD sobre el umbral
        (0.10, None, 1.5, None),
    ],
)
def test_invariants_reject_incoherent_output(p, band, threshold, model_decision):
    with pytest.raises(P.InvariantError):
        P.check_invariants(p, band, threshold, model_decision)


def test_invariants_accept_coherent_output():
    assert P.check_invariants("0.12", "C", 0.18, "APROBAR") == pytest.approx(0.12)
    assert P.check_invariants(0.25, "D", 0.18, "RECHAZAR") == pytest.approx(0.25)


def test_risk_band_is_shared_with_the_model():
    assert risk_band is P.risk_band


def test_input_hash_is_canonical():
    a = {"x": 1, "y": "b"}
    assert P.input_hash(a) == P.input_hash({"y": "b", "x": 1}) and len(P.input_hash(a)) == 64
    assert P.input_hash(a) != P.input_hash({"x": 2, "y": "b"})


def test_model_reports_its_threshold(lr_model):

    from credit_risk.data.synthetic import generate_raw

    out = lr_model.predict_frame(generate_raw(50, seed=1).pipe(_canonical), explain=False)
    assert (out["threshold"] == lr_model.threshold).all()


def _canonical(raw):
    from credit_risk.data.lending_club import parse_raw
    from credit_risk.data.quality import clean

    return clean(parse_raw(raw))
