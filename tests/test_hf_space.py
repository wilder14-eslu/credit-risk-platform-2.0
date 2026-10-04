"""Demo de Hugging Face Spaces: lógica de scoring y carpeta autocontenida."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import joblib
import pytest

from credit_risk import config as C

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "apps" / "hf_space"


@pytest.fixture(scope="module")
def demo_core():
    mp = pytest.MonkeyPatch()
    mp.setenv("CREDIT_RISK_CONFIG_DIR", str(C.CONFIG_DIR))  # demo_core no debe alterar la config del proceso
    mp.syspath_prepend(str(SPACE))
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


def test_build_space_is_self_contained(tmp_path, lr_model):
    """La carpeta generada puntúa sola: sin el repo en el path ni la config del entorno."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        build_hf_space = importlib.import_module("build_hf_space")
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    model_file = tmp_path / "model.joblib"
    joblib.dump(lr_model, model_file)

    out = build_hf_space.build(tmp_path / "space", model_file, "champion", "workspace", "credit_risk", "3.12")

    for name in ("app.py", "demo_core.py", "README.md", "requirements.txt", "config/data_schema.yaml"):
        assert (out / name).exists(), name
    assert (out / "credit_risk" / "models" / "credit_model.py").exists()
    info = json.loads((out / "model" / "model_info.json").read_text(encoding="utf-8"))
    assert info["algorithm"] == "logistic_regression" and info["threshold"] == pytest.approx(lr_model.threshold)
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n") and 'python_version: "3.12"' in readme and "sdk: gradio" in readme
    reqs = (out / "requirements.txt").read_text(encoding="utf-8")
    assert "scikit-learn" in reqs and "catboost" not in reqs


def test_gradio_app_evaluates(demo_core, lr_model, monkeypatch):
    pytest.importorskip("gradio")
    pytest.importorskip("plotly")
    app = importlib.import_module("app")
    monkeypatch.setattr(demo_core, "load_model", lambda path=None: lr_model)
    html, fig = app.evaluate(*demo_core.EXAMPLES["Perfil típico"])
    assert "Probabilidad de default" in html and fig.data
    assert app.build_demo() is not None
    assert "AUC" in app.results_markdown()
