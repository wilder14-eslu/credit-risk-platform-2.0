"""Demo pública (Streamlit Community Cloud): lógica de scoring, app y carpeta autocontenida."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import joblib
import pytest

from credit_risk import config as C

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "apps" / "demo"


@pytest.fixture(scope="module")
def demo_core():
    mp = pytest.MonkeyPatch()
    mp.setenv("CREDIT_RISK_CONFIG_DIR", str(C.CONFIG_DIR))  # demo_core no debe alterar la config del proceso
    mp.syspath_prepend(str(DEMO))
    yield importlib.import_module("demo_core")
    mp.undo()


def _form(demo_core, profile: str) -> dict:
    return dict(zip(demo_core.FORM_FIELDS, demo_core.EXAMPLES[profile], strict=True))


def test_build_record_has_every_model_input(demo_core):
    record = demo_core.build_record(**_form(demo_core, "Perfil típico"))
    assert set(C.input_features()) <= set(record)
    assert record["installment"] == pytest.approx(407.1, abs=0.5)  # 12 000 a 36 meses al 13.5 %
    assert record["mort_acc"] == 0.0 and record["addr_state"] == "CA"


def test_installment_without_interest(demo_core):
    assert demo_core.installment(1200, 0, 12) == 100


def test_score_orders_profiles_by_risk(demo_core, xgb_model):
    pds = {
        name: demo_core.score(demo_core.build_record(**_form(demo_core, name)), model=xgb_model)
        for name in demo_core.EXAMPLES
    }
    typical = pds["Perfil típico"]
    assert 0 < typical["probability"] < 1 and typical["decision"] in ("APROBAR", "RECHAZAR")
    assert typical["factors"] and {"feature", "label", "impact", "direction"} <= set(typical["factors"][0])
    assert pds["Perfil conservador"]["probability"] < pds["Perfil riesgoso"]["probability"]


@pytest.fixture(scope="module")
def built_demo(tmp_path_factory, lr_model):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        build_demo = importlib.import_module("build_demo")
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    tmp = tmp_path_factory.mktemp("demo")
    model_file = tmp / "model.joblib"
    joblib.dump(lr_model, model_file)
    return build_demo.build(tmp / "out", model_file, "champion", "workspace", "credit_risk")


def test_build_demo_is_self_contained(built_demo, lr_model):
    """La carpeta generada puntúa sola (lo verifica build) y lleva todo lo que necesita Streamlit Cloud."""
    out = built_demo
    for name in ("app.py", "demo_core.py", "README.md", "requirements.txt", ".streamlit/config.toml"):
        assert (out / name).exists(), name
    assert (out / "config" / "data_schema.yaml").exists()
    assert (out / "credit_risk" / "models" / "credit_model.py").exists()
    assert not list(out.rglob("__pycache__"))
    info = json.loads((out / "model" / "model_info.json").read_text(encoding="utf-8"))
    assert info["algorithm"] == "logistic_regression" and info["threshold"] == pytest.approx(lr_model.threshold)
    reqs = (out / "requirements.txt").read_text(encoding="utf-8")
    assert "streamlit" in reqs and "scikit-learn" in reqs and "catboost" not in reqs


def test_streamlit_app_scores_a_request(built_demo, monkeypatch):
    """Corre la app generada de punta a punta, como en Streamlit Cloud."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("plotly")
    monkeypatch.delitem(sys.modules, "demo_core", raising=False)  # usar el demo_core de la carpeta generada
    monkeypatch.syspath_prepend(str(built_demo))
    at = testing.AppTest.from_file(str(built_demo / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Evaluar solicitud", "Por qué este modelo", "Arquitectura"]
    at.button[0].click().run()
    assert not at.exception, at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Probabilidad de default"].endswith("%")
    assert any(word in labels["Decisión"] for word in ("APROBAR", "RECHAZAR"))
    assert labels["AUC test"].startswith("0.")
