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


class FakeDatabricks:
    """Responde las consultas del dashboard como lo haría el SQL warehouse (filas como dicts)."""

    ROWS = {
        "monitoring_metrics": [
            {"run_id": f"r{i}", "run_ts": f"2026-10-0{i + 1} 10:00:00", "clock_month": f"2014-0{i + 1}-01",
             "roc_auc": str(0.70 - 0.01 * i), "prediction_psi": "0.04", "default_rate_observed": "0.15",
             "default_rate_predicted": "0.17", "auc_drop": str(0.01 * i), "severity": "estable" if i < 2 else "warning",
             "reasons": json.dumps(["AUC 0.68 < referencia"]) if i == 2 else "[]"}
            for i in range(3)
        ],
        "drift_by_feature": [
            {"feature": "int_rate", "kind": "numeric", "psi": "0.12", "ks_pvalue": "0.001", "status": "warning"},
            {"feature": "dti", "kind": "numeric", "psi": "0.03", "ks_pvalue": "0.2", "status": "estable"},
        ],
        "model_evaluation": [
            {"split": split, "model": m, "auc": str(a), "auc_ci_low": str(a - 0.005), "auc_ci_high": str(a + 0.005),
             "selected": str(m == "catboost").lower()}
            for split in ("validation", "test") for m, a in (("catboost", 0.693), ("logistic_regression", 0.686))
        ],
        "model_benchmark": [{"algorithm": "catboost", "test_roc_auc": "0.69", "selected": "true"}],
    }  # fmt: skip

    def sql(self, warehouse_id, statement):
        table = next(t for t in ("monitoring_metrics", "drift_by_feature", "ab_test_results", "retrain_events",
                                 "model_benchmark", "model_evaluation") if f".{t}" in statement.split("WHERE")[0])  # fmt: skip
        if table == "retrain_events":
            raise RuntimeError("TABLE_OR_VIEW_NOT_FOUND")  # una tabla que aún no existe no rompe la exportación
        return self.ROWS.get(table, [])


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
    out = build_demo.build(tmp / "out", model_file, "champion", "workspace", "credit_risk")
    assert not (out / "snapshot").exists()  # sin warehouse no hay foto
    info = build_demo.export_snapshot(out / "snapshot", "wh", "workspace", "credit_risk", client=FakeDatabricks())
    assert info["rows"]["monitoring_metrics"] == 3 and info["rows"]["retrain_events"] == 0
    return out


def test_build_demo_is_self_contained(built_demo, lr_model):
    """La carpeta generada puntúa sola (lo verifica build) y lleva todo lo que necesita Streamlit Cloud."""
    out = built_demo
    for name in ("app.py", "demo_core.py", "views.py", "README.md", "requirements.txt", ".streamlit/config.toml"):
        assert (out / name).exists(), name
    assert (out / "config" / "data_schema.yaml").exists()
    assert (out / "credit_risk" / "models" / "credit_model.py").exists()
    assert not list(out.rglob("__pycache__"))
    info = json.loads((out / "model" / "model_info.json").read_text(encoding="utf-8"))
    assert info["algorithm"] == "logistic_regression" and info["threshold"] == pytest.approx(lr_model.threshold)
    reqs = (out / "requirements.txt").read_text(encoding="utf-8")
    assert "streamlit" in reqs and "scikit-learn" in reqs and "catboost" not in reqs


@pytest.fixture
def app_test(built_demo, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("plotly")
    for name in ("demo_core", "views"):  # usar los módulos de la carpeta generada
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(built_demo))
    at = testing.AppTest.from_file(str(built_demo / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def test_streamlit_app_scores_a_request(app_test):
    """Corre la app generada de punta a punta, como en Streamlit Cloud."""
    at = app_test
    assert at.sidebar.radio[0].options == ["Scoring", "Monitoreo", "Modelos", "Por qué este modelo", "Arquitectura"]
    at.button[0].click().run()
    assert not at.exception, at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Probabilidad de default"].endswith("%")
    assert any(word in labels["Decisión"] for word in ("APROBAR", "RECHAZAR"))


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        ("Monitoreo", "Último chequeo (2014-03): warning"),
        ("Modelos", "Benchmark del último entrenamiento"),
        ("Por qué este modelo", "Benchmark"),
    ],
)
def test_streamlit_pages_render_the_snapshot(app_test, page, expected):
    at = app_test
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    texts = [e.value for e in at.subheader] + [e.value for e in at.header]
    assert any(expected in t for t in texts), texts


def test_databricks_dashboard_uses_the_same_views(monkeypatch):
    """El dashboard de Databricks Apps renderiza las mismas páginas leyendo en vivo (cliente simulado)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("plotly")
    import types

    fake = FakeDatabricks()
    fake.served_versions = lambda endpoint: {"champion": "7"}
    module = types.ModuleType("databricks_client")
    module.DatabricksClient = lambda: fake
    monkeypatch.setitem(sys.modules, "databricks_client", module)
    monkeypatch.delitem(sys.modules, "views", raising=False)
    monkeypatch.syspath_prepend(str(ROOT / "apps" / "dashboard"))
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "wh")
    at = testing.AppTest.from_file(str(ROOT / "apps" / "dashboard" / "app.py"), default_timeout=60).run()
    for page, expected in (("Monitoreo", "Último chequeo (2014-03): warning"), ("Modelos", "Versiones servidas")):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, at.exception
        assert any(expected in s.value for s in at.subheader), [s.value for s in at.subheader]
