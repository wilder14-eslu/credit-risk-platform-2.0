"""Tests de la API FastAPI que corre como Databricks App (con un cliente de Databricks simulado)."""

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from credit_risk.config import PROJECT_ROOT
from credit_risk.monitoring.ab_testing import assign_variant as core_assign

API_DIR = PROJECT_ROOT / "apps" / "api"

APPLICATION = {
    "loan_amnt": 12000,
    "term_months": 36,
    "int_rate": 13.5,
    "installment": 407.2,
    "grade": "C",
    "sub_grade": "C2",
    "emp_length_years": 5,
    "home_ownership": "RENT",
    "annual_inc": 65000,
    "verification_status": "Source Verified",
    "purpose": "debt_consolidation",
    "addr_state": "CA",
    "dti": 18.5,
    "delinq_2yrs": 0,
    "fico_score": 702,
    "inq_last_6mths": 1,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_bal": 14500,
    "revol_util": 55.2,
    "total_acc": 24,
    "mort_acc": 1,
    "pub_rec_bankruptcies": 0,
    "credit_history_months": 180,
    "application_type": "Individual",
    "initial_list_status": "w",
}


@pytest.fixture(scope="module")
def api():
    sys.path.insert(0, str(API_DIR))
    modules = {name: importlib.import_module(name) for name in ("main", "sinks", "ab", "databricks_client")}
    yield modules
    sys.path.remove(str(API_DIR))
    for name in modules:
        sys.modules.pop(name, None)


class FakeDatabricks:
    """Imita DatabricksClient usando el modelo real entrenado en los tests."""

    def __init__(self, models):
        self.models, self.calls, self.statements = models, [], []

    def served_versions(self, endpoint):
        return {"champion": "3", "challenger": "4"}

    def invoke(self, endpoint, variant, records, params=None):
        import pandas as pd

        self.calls.append(variant)
        return self.models[variant].predict_frame(pd.DataFrame(records)).to_dict(orient="records")

    def sql(self, warehouse_id, statement):
        self.statements.append(statement)
        return []


@pytest.fixture()
def client(api, lr_model, xgb_model):
    fake = FakeDatabricks({"champion": lr_model, "challenger": xgb_model})
    sink = api["sinks"].MemorySink()
    with TestClient(api["main"].create_app(client=fake, sink=sink)) as c:
        yield c, fake, sink


def test_predict_logs_prediction(client):
    c, fake, sink = client
    body = c.post("/api/v1/predict", json={"loan_id": "L-1", "application": APPLICATION}).json()
    assert 0 <= body["probability"] <= 1 and body["variant"] in {"champion", "challenger"}
    assert len(body["top_factors"]) == 3
    assert sink.predictions[-1]["request_id"] == body["request_id"]
    assert sink.predictions[-1]["source"] == "api"


def test_ab_is_sticky_and_matches_core(client):
    c, fake, _ = client
    variants = {}
    for i in range(40):
        lid = f"L-{i}"
        variants[lid] = c.post("/api/v1/predict", json={"loan_id": lid, "application": APPLICATION}).json()["variant"]
        assert variants[lid] == core_assign(lid, 0.2)
    assert set(variants.values()) == {"champion", "challenger"}


def test_validation(client):
    c, *_ = client
    bad = {**APPLICATION, "grade": "Z"}
    assert c.post("/api/v1/predict", json={"application": bad}).status_code == 422


def test_outcomes_and_health(client):
    c, _, sink = client
    assert c.post("/api/v1/outcomes", json={"loan_id": "L-1", "actual_default": True}).status_code == 200
    assert sink.outcomes[-1]["actual_default"] == 1
    assert c.get("/health").json()["served"] == {"champion": "3", "challenger": "4"}
    assert c.get("/api/v1/models").json()["ab_test_active"] is True


def test_delta_sink_batches_and_escapes(api):
    fake = FakeDatabricks({})
    sink = api["sinks"].DeltaSink(fake, "wh", "workspace.credit_risk", flush_every_n=3, flush_every_seconds=999)
    sink.log_prediction({"request_id": "r1", "purpose": "O'Neil", "event_ts": "2026-01-01 00:00:00"})
    sink.log_prediction({"request_id": "r2", "probability": 0.2})
    assert fake.statements == []
    sink.log_outcome({"loan_id": "L", "actual_default": 1, "observed_ts": "2026-01-01 00:00:00"})
    assert any("INSERT INTO workspace.credit_risk.inference_log" in s for s in fake.statements)
    assert any("O\\'Neil" in s for s in fake.statements)


def test_app_copies_are_identical():
    """ab.py y databricks_client.py se duplican en cada app (se despliegan solas)."""
    for name in ("ab.py", "databricks_client.py"):
        api_src = (API_DIR / name).read_text(encoding="utf-8")
        dash_src = (PROJECT_ROOT / "apps" / "dashboard" / name).read_text(encoding="utf-8")
        assert api_src == dash_src


def test_databricks_client_uses_served_model_path(api):
    class FakeApi:
        def __init__(self):
            self.paths = []

        def do(self, method, path, body=None):
            self.paths.append(path)
            return {"predictions": {"probability": [0.1], "decision": ["APROBAR"]}}

    class FakeW:
        api_client = FakeApi()

    dc = api["databricks_client"].DatabricksClient(workspace_client=FakeW())
    out = dc.invoke("ep", "challenger", [APPLICATION], {"explain": False})
    assert out == [{"probability": 0.1, "decision": "APROBAR"}]
    assert FakeW.api_client.paths == ["/serving-endpoints/ep/served-models/challenger/invocations"]
    assert Path(API_DIR / "app.yaml").exists()
