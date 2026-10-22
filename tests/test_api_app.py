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
    names = ("main", "sinks", "ab", "databricks_client", "policy", "resilience", "schemas")
    for name in names:  # otra app (dashboard, demo) pudo importar módulos con el mismo nombre
        sys.modules.pop(name, None)
    modules = {name: importlib.import_module(name) for name in names}
    yield modules
    sys.path.remove(str(API_DIR))
    for name in modules:
        sys.modules.pop(name, None)


class FakeDatabricks:
    """Imita DatabricksClient usando el modelo real entrenado en los tests."""

    def __init__(self, models):
        self.models, self.calls, self.statements, self.parameters = models, [], [], []

    def served_versions(self, endpoint):
        return {"champion": "3", "challenger": "4"}

    def invoke(self, endpoint, variant, records, params=None):
        import pandas as pd

        self.calls.append(variant)
        return self.models[variant].predict_frame(pd.DataFrame(records)).to_dict(orient="records")

    def sql(self, warehouse_id, statement, parameters=None):
        self.statements.append(statement)
        self.parameters.append(parameters or [])
        return []


@pytest.fixture()
def client(api, lr_model, xgb_model):
    fake = FakeDatabricks({"champion": lr_model, "challenger": xgb_model})
    sink = api["sinks"].MemorySink()
    with TestClient(api["main"].create_app(client=fake, sink=sink)) as c:
        yield c, fake, sink


def test_predict_logs_prediction(client):
    c, fake, sink = client
    body = c.post("/api/v1/predict", json={"loan_id": "L-1", "application": APPLICATION, "explain": True}).json()
    assert 0 <= body["probability"] <= 1 and body["variant"] in {"champion", "challenger"}
    assert len(body["top_factors"]) == 3
    assert body["decision"] in {"APROBAR", "RECHAZAR", "REVISION_MANUAL"} and body["policy_version"]
    assert len(body["input_hash"]) == 64 and body["model_latency_ms"] is not None
    logged = sink.predictions[-1]
    assert logged["request_id"] == body["request_id"] and logged["source"] == "api"
    assert logged["policy_version"] == body["policy_version"] and logged["input_hash"] == body["input_hash"]


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


def test_delta_sink_batches_with_parameters(api):
    """Los datos viajan como parámetros: un valor malicioso nunca llega al texto del SQL."""
    fake = FakeDatabricks({})
    sink = api["sinks"].DeltaSink(
        fake, "wh", "workspace.credit_risk", flush_every_n=3, flush_every_seconds=999, background=False
    )
    evil = "x'); DROP TABLE workspace.credit_risk.inference_log; --"
    sink.log_prediction({"request_id": "r1", "purpose": evil, "event_ts": "2026-01-01 00:00:00"})
    sink.log_prediction({"request_id": "r2", "probability": 0.2})
    assert fake.statements == []
    sink.log_outcome({"loan_id": "L", "actual_default": 1, "observed_ts": "2026-01-01 00:00:00"})
    inserts = [(s, p) for s, p in zip(fake.statements, fake.parameters, strict=True) if s.startswith("INSERT")]
    assert any("INSERT INTO workspace.credit_risk.inference_log" in s for s, _ in inserts)
    assert all(evil not in s and "DROP" not in s for s, _ in inserts)
    values = [p["value"] for _, params in inserts for p in params]
    assert evil in values and "0.2" in values
    types = {p["name"].split("_", 1)[1]: p["type"] for _, params in inserts for p in params}
    assert types["probability"] == "DOUBLE" and types["event_ts"] == "TIMESTAMP" and types["purpose"] == "STRING"


def test_delta_sink_splits_large_batches_and_requeues(api):
    sinks = api["sinks"]
    fake = FakeDatabricks({})
    sink = sinks.DeltaSink(fake, "wh", "w.s", flush_every_n=10_000, flush_every_seconds=999, background=False)
    for i in range(30):
        sink.log_prediction({"request_id": f"r{i}"})
    sink.flush()
    inserts = [p for s, p in zip(fake.statements, fake.parameters, strict=True) if s.startswith("INSERT")]
    assert len(inserts) > 1 and all(len(p) <= sinks.MAX_PARAMETERS for p in inserts)
    assert sum(len(p) for p in inserts) == 30 * len(sinks.LOG_COLUMNS)

    class Down(FakeDatabricks):
        def sql(self, warehouse_id, statement, parameters=None):
            raise RuntimeError("warehouse apagado")

    down = sinks.DeltaSink(Down({}), "wh", "w.s", flush_every_n=10_000, flush_every_seconds=999, background=False)
    down.log_prediction({"request_id": "r1"})
    down.flush()
    assert down._pending() == 1  # nada se pierde si el warehouse falla


def test_delta_sink_only_writes_existing_columns(api):
    """Si la tabla aún no tiene las columnas nuevas (antes de migrar), se escribe lo que existe."""

    class OldTable(FakeDatabricks):
        def sql(self, warehouse_id, statement, parameters=None):
            super().sql(warehouse_id, statement, parameters)
            if statement.startswith("DESCRIBE"):
                return [{"col_name": c} for c in ("request_id", "probability", "decision")]
            return []

    fake = OldTable({})
    sink = api["sinks"].DeltaSink(fake, "wh", "w.s", flush_every_n=1, flush_every_seconds=999, background=False)
    sink.log_prediction({"request_id": "r1", "probability": 0.3, "policy_version": "x"})
    insert = next(s for s in fake.statements if s.startswith("INSERT"))
    assert "(request_id, probability, decision)" in insert and "policy_version" not in insert


def test_delta_sink_background_thread_flushes(api):
    import time as _time

    fake = FakeDatabricks({})
    sink = api["sinks"].DeltaSink(fake, "wh", "w.s", flush_every_n=2, flush_every_seconds=999)
    sink.log_prediction({"request_id": "r1"})
    sink.log_prediction({"request_id": "r2"})  # llena el lote: despierta al hilo, no bloquea
    for _ in range(100):
        if any(s.startswith("INSERT") for s in fake.statements):
            break
        _time.sleep(0.02)
    assert any(s.startswith("INSERT") for s in fake.statements)
    sink.close()


def test_app_copies_are_identical():
    """ab.py y databricks_client.py se duplican en cada app; policy.py es copia del paquete (se despliegan solas)."""
    for name in ("ab.py", "databricks_client.py"):
        api_src = (API_DIR / name).read_text(encoding="utf-8")
        dash_src = (PROJECT_ROOT / "apps" / "dashboard" / name).read_text(encoding="utf-8")
        assert api_src == dash_src
    core = (PROJECT_ROOT / "src" / "credit_risk" / "policy.py").read_text(encoding="utf-8")
    assert (API_DIR / "policy.py").read_text(encoding="utf-8") == core, "copia apps/api/policy.py desactualizada"


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


def test_root_redirects_to_docs(client):
    c, *_ = client
    response = c.get("/", follow_redirects=False)
    assert response.status_code in (302, 307) and response.headers["location"] == "/docs"


# --- Hardening: validación estricta, política, resiliencia e idempotencia ----------------------


@pytest.mark.parametrize(
    "change",
    [
        {"purpose": "crypto"},  # fuera de la lista cerrada
        {"addr_state": "ZZ"},
        {"sub_grade": "A1"},  # no corresponde al grade C
        {"unknown_field": 1},  # campos extra prohibidos
        {"annual_inc": 1e12},
    ],
)
def test_strict_validation(client, change):
    c, *_ = client
    assert c.post("/api/v1/predict", json={"application": {**APPLICATION, **change}}).status_code == 422


def test_rejects_extra_top_level_fields_and_bad_ids(client):
    c, *_ = client
    assert c.post("/api/v1/predict", json={"application": APPLICATION, "admin": True}).status_code == 422
    assert c.post("/api/v1/predict", json={"loan_id": "x'; --", "application": APPLICATION}).status_code == 422


def test_body_size_limit_and_security_headers(client):
    c, *_ = client
    big = c.post("/api/v1/predict", content=b"{" + b" " * 20_000 + b"}", headers={"content-type": "application/json"})
    assert big.status_code == 413
    ok = c.get("/health")
    assert ok.headers["x-content-type-options"] == "nosniff" and ok.headers["cache-control"] == "no-store"


def test_rejection_always_explains_and_approval_only_on_request(client):
    c, fake, _ = client
    risky = {**APPLICATION, "int_rate": 28.0, "grade": "G", "sub_grade": "G5", "fico_score": 620, "dti": 39.0}
    body = c.post("/api/v1/predict", json={"application": risky}).json()
    assert body["decision"] in {"RECHAZAR", "REVISION_MANUAL"}
    assert body["reasons"] and body["top_factors"]  # motivos y factores sin pedirlos
    safe = {**APPLICATION, "int_rate": 6.0, "grade": "A", "sub_grade": "A1", "fico_score": 820, "dti": 5.0}
    approved = c.post("/api/v1/predict", json={"application": safe}).json()
    if approved["decision"] == "APROBAR":
        assert approved["reasons"] == [] and approved["top_factors"] == []


def test_knockout_rule_rejects_regardless_of_model(client):
    c, *_ = client
    unaffordable = {**APPLICATION, "installment": 3000.0, "annual_inc": 30000}  # cuota = 120 % del ingreso mensual
    body = c.post("/api/v1/predict", json={"application": unaffordable}).json()
    assert body["decision"] == "RECHAZAR" and body["reasons"][0] == "CUOTA_INGRESO_ALTA"


def test_idempotent_requests_hit_the_cache(client):
    c, fake, sink = client
    payload = {"loan_id": "IDEM-1", "application": APPLICATION}
    first = c.post("/api/v1/predict", json=payload).json()
    calls, logged = len(fake.calls), len(sink.predictions)
    second = c.post("/api/v1/predict", json=payload).json()
    assert second["cached"] is True and first["cached"] is False
    assert second["request_id"] == first["request_id"] and second["probability"] == first["probability"]
    assert len(fake.calls) == calls and len(sink.predictions) == logged  # ni modelo ni log otra vez
    changed = c.post("/api/v1/predict", json={**payload, "application": {**APPLICATION, "dti": 20.0}}).json()
    assert changed["cached"] is False  # otros datos: otra huella


class _Broken:
    """Model Serving caído: cada llamada falla."""

    def __init__(self):
        self.calls = 0

    def served_versions(self, endpoint):
        return {"champion": "3"}

    def invoke(self, *a, **k):
        self.calls += 1
        raise RuntimeError("503 endpoint scaling from zero")

    def sql(self, *a, **k):
        return []


def test_serving_down_goes_to_manual_review_and_opens_the_circuit(api):
    broken, sink = _Broken(), api["sinks"].MemorySink()
    breaker = api["resilience"].CircuitBreaker(failure_threshold=2, reset_seconds=60)
    with TestClient(api["main"].create_app(client=broken, sink=sink, breaker=breaker)) as c:
        bodies = [c.post("/api/v1/predict", json={"application": APPLICATION}).json() for _ in range(4)]
        health = c.get("/health").json()
    for body in bodies:  # nunca se aprueba sin modelo
        assert body["decision"] == "REVISION_MANUAL" and body["reasons"] == ["MODELO_NO_DISPONIBLE"]
        assert body["probability"] is None
    assert broken.calls == 2 and health["serving_circuit"] == "abierto"  # tras 2 fallos ya no se llama
    assert sink.predictions[-1]["decision"] == "REVISION_MANUAL" and sink.predictions[-1]["probability"] is None


def test_slow_serving_times_out(api, lr_model):
    import time as _time

    class Slow(FakeDatabricks):
        def invoke(self, *a, **k):
            _time.sleep(2.0)
            return super().invoke(*a, **k)

    slow = Slow({"champion": lr_model})
    slow.served_versions = lambda endpoint: {"champion": "3"}
    with TestClient(api["main"].create_app(client=slow, sink=api["sinks"].MemorySink(), timeout_seconds=0.2)) as c:
        started = _time.perf_counter()
        body = c.post("/api/v1/predict", json={"application": APPLICATION}).json()
        assert _time.perf_counter() - started < 1.5  # sin timeout tardaría 2 s o más
    assert body["decision"] == "REVISION_MANUAL" and body["reasons"] == ["MODELO_NO_DISPONIBLE"]


def test_incoherent_model_output_is_never_used(api, lr_model):
    class Corrupt(FakeDatabricks):
        def invoke(self, *a, **k):
            out = super().invoke(*a, **k)
            out[0]["probability"] = 1.7  # fuera de [0, 1]
            return out

    corrupt = Corrupt({"champion": lr_model})
    corrupt.served_versions = lambda endpoint: {"champion": "3"}
    with TestClient(api["main"].create_app(client=corrupt, sink=api["sinks"].MemorySink())) as c:
        body = c.post("/api/v1/predict", json={"application": APPLICATION}).json()
    assert body["decision"] == "REVISION_MANUAL" and body["reasons"] == ["SALIDA_MODELO_INCOHERENTE"]
    assert body["probability"] is None


def test_versions_are_cached_and_failures_back_off(api, lr_model):
    fake = FakeDatabricks({"champion": lr_model})
    calls = {"n": 0}

    def flaky(endpoint):
        calls["n"] += 1
        raise RuntimeError("timeout")

    fake.served_versions = flaky
    with TestClient(api["main"].create_app(client=fake, sink=api["sinks"].MemorySink())) as c:
        for _ in range(5):
            c.post("/api/v1/predict", json={"application": APPLICATION})
    assert calls["n"] == 1  # un fallo no provoca una consulta por cada solicitud


# --- Cobertura de bordes de la API -------------------------------------------------------------


def test_lifespan_builds_real_client_and_sink(api, monkeypatch):
    """Sin cliente ni sink inyectados, la app crea los suyos (MemorySink sin warehouse, DeltaSink con él)."""
    created = []

    class StubClient(FakeDatabricks):
        def __init__(self):
            super().__init__({})
            created.append(self)

    main = api["main"]
    monkeypatch.setattr(main, "DatabricksClient", StubClient)
    monkeypatch.setattr(main, "WAREHOUSE_ID", "")
    with TestClient(main.create_app()) as c:
        assert c.get("/health").status_code == 200
    monkeypatch.setattr(main, "WAREHOUSE_ID", "wh")
    with TestClient(main.create_app()) as c:
        c.post("/api/v1/outcomes", json={"loan_id": "L-9", "actual_default": False})
    assert len(created) == 2
    assert any(s.startswith("INSERT INTO workspace.credit_risk.outcomes") for s in created[1].statements)


def test_failed_challenger_falls_back_to_champion(api, lr_model):
    class ChallengerDown(FakeDatabricks):
        def invoke(self, endpoint, variant, records, params=None):
            if variant == "challenger":
                raise RuntimeError("challenger caído")
            return super().invoke(endpoint, variant, records, params)

    fake = ChallengerDown({"champion": lr_model})
    lid = next(f"L-{i}" for i in range(500) if core_assign(f"L-{i}", 0.2) == "challenger")
    with TestClient(api["main"].create_app(client=fake, sink=api["sinks"].MemorySink())) as c:
        body = c.post("/api/v1/predict", json={"loan_id": lid, "application": APPLICATION}).json()
    assert body["variant"] == "champion" and body["probability"] is not None


def test_monitoring_summary(api, monkeypatch):
    main = api["main"]

    class Partial(FakeDatabricks):
        def sql(self, warehouse_id, statement, parameters=None):
            if "retrain_events" in statement:
                raise RuntimeError("tabla no existe")
            return [{"ok": 1}]

    monkeypatch.setattr(main, "WAREHOUSE_ID", "")
    with TestClient(main.create_app(client=Partial({}), sink=api["sinks"].MemorySink())) as c:
        assert c.get("/api/v1/monitoring/summary").json()["available"] is False
    monkeypatch.setattr(main, "WAREHOUSE_ID", "wh")
    with TestClient(main.create_app(client=Partial({}), sink=api["sinks"].MemorySink())) as c:
        out = c.get("/api/v1/monitoring/summary").json()
    assert out["available"] and out["monitoring"] == [{"ok": 1}] and out["retrains"] == []
    assert "tabla no existe" in out["errors"]["retrains"]


def test_ttl_cache_expires_and_evicts(api):
    now = [0.0]
    cache = api["resilience"].TTLCache(max_items=2, ttl_seconds=10, clock=lambda: now[0])
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)  # supera el máximo: sale el menos usado ("b")
    assert cache.get("b") is None and cache.get("c") == 3
    now[0] = 11.0
    assert cache.get("a") is None and cache.get("missing") is None


def test_circuit_breaker_half_open_recovers(api):
    now = [0.0]
    breaker = api["resilience"].CircuitBreaker(failure_threshold=1, reset_seconds=5, clock=lambda: now[0])
    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert breaker.state == "abierto"
    with pytest.raises(api["resilience"].CircuitOpenError):
        breaker.call(lambda: 1)
    now[0] = 6.0
    assert breaker.state == "semiabierto" and breaker.call(lambda: 42) == 42 and breaker.state == "cerrado"


def test_parameter_values_and_memory_sink(api):
    sinks = api["sinks"]
    assert sinks.parameter_value(True) == "true" and sinks.parameter_value(False) == "false"
    assert sinks.parameter_value(None) is None and sinks.parameter_value(float("nan")) is None
    assert sinks.parameter_value(3) == "3" and sinks.sql_type("actual_default") == "INT"
    mem = sinks.MemorySink()
    mem.flush()
    mem.close()


def test_databricks_client_with_sdk_objects(api, monkeypatch):
    """El cliente real se prueba contra objetos con la forma de las respuestas del SDK de Databricks."""
    from types import SimpleNamespace as NS

    import databricks.sdk
    from databricks.sdk.service.sql import StatementState

    dc_mod = api["databricks_client"]

    class FakeStatements:
        def __init__(self):
            self.kwargs = []

        def execute_statement(self, **kwargs):
            self.kwargs.append(kwargs)
            if "falla" in kwargs["statement"]:
                return NS(status=NS(state=StatementState.FAILED, error=NS(message="sintaxis")))
            if "vacio" in kwargs["statement"]:
                return NS(status=NS(state=StatementState.SUCCEEDED, error=None), manifest=None, result=None)
            return NS(
                status=NS(state=StatementState.SUCCEEDED, error=None),
                manifest=NS(schema=NS(columns=[NS(name="a"), NS(name="b")])),
                result=NS(data_array=[["1", "x"], ["2", "y"]]),
            )

    fake_w = NS(
        statement_execution=FakeStatements(),
        serving_endpoints=NS(get=lambda name: NS(config=NS(served_entities=[NS(name="champion", entity_version=3)]))),
    )
    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", lambda: fake_w)
    dc = dc_mod.DatabricksClient()  # sin argumentos: construye el WorkspaceClient (aquí simulado)
    assert dc.served_versions("ep") == {"champion": "3"}
    rows = dc.sql("wh", "SELECT :v", [{"name": "v", "value": "1", "type": "INT"}])
    assert rows == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]
    params = fake_w.statement_execution.kwargs[-1]["parameters"]
    assert params[0].name == "v" and params[0].value == "1" and params[0].type == "INT"
    assert dc.sql("wh", "SELECT vacio") == []
    with pytest.raises(RuntimeError, match="sintaxis"):
        dc.sql("wh", "SELECT falla")
