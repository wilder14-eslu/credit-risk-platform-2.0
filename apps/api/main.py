"""API de scoring crediticio desplegada como Databricks App (FastAPI).

- Autenticación: la de Databricks Apps (OAuth del workspace), sin tokens en el código.
- Modelo: Databricks Model Serving (champion y challenger en el mismo endpoint).
- A/B testing: asignación determinista por `loan_id` y registro de la variante.
- Feedback loop: predicciones y desenlaces se escriben en Delta (`inference_log`, `outcomes`).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, status

from ab import assign_variant
from databricks_client import DatabricksClient
from schemas import OutcomeRequest, PredictionRequest, PredictionResponse
from sinks import DeltaSink, MemorySink

logger = logging.getLogger("credit-risk-api")
logging.basicConfig(level=logging.INFO)

ENDPOINT = os.getenv("SERVING_ENDPOINT", "credit-risk-lc-endpoint")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
FQ_SCHEMA = f"{os.getenv('UC_CATALOG', 'workspace')}.{os.getenv('UC_SCHEMA', 'credit_risk')}"
CHALLENGER_TRAFFIC = float(os.getenv("CHALLENGER_TRAFFIC", "0.2"))
VERSION_CACHE_SECONDS = 60.0


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def create_app(client=None, sink=None) -> FastAPI:
    state: dict = {"client": client, "sink": sink, "versions": {}, "fetched": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["client"] is None:
            state["client"] = DatabricksClient()
        if state["sink"] is None:
            state["sink"] = DeltaSink(state["client"], WAREHOUSE_ID, FQ_SCHEMA) if WAREHOUSE_ID else MemorySink()
        yield
        state["sink"].flush()

    app = FastAPI(
        title="Credit Risk API (Databricks App)",
        version="2.0.0",
        description="Scoring de riesgo crediticio Lending Club sobre Databricks Model Serving con A/B testing.",
        lifespan=lifespan,
    )

    def versions() -> dict[str, str]:
        if state["fetched"] is None or time.monotonic() - state["fetched"] > VERSION_CACHE_SECONDS:
            try:
                state["versions"] = state["client"].served_versions(ENDPOINT)
                state["fetched"] = time.monotonic()
            except Exception as exc:
                logger.warning("No se pudo leer el endpoint: %s", exc)
        return state["versions"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "endpoint": ENDPOINT, "served": versions()}

    @app.get("/api/v1/models")
    def models() -> dict:
        served = versions()
        return {
            "served": served,
            "ab_test_active": "challenger" in served,
            "challenger_traffic": CHALLENGER_TRAFFIC if "challenger" in served else 0.0,
        }

    @app.post("/api/v1/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        served = versions()
        loan_id = request.loan_id or f"api-{uuid.uuid4().hex[:12]}"
        variant = assign_variant(loan_id, CHALLENGER_TRAFFIC if "challenger" in served else 0.0)
        record = request.application.model_dump()
        start = time.perf_counter()
        try:
            result = state["client"].invoke(ENDPOINT, variant, [record], {"explain": request.explain})[0]
        except Exception as exc:
            if variant != "challenger":
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Model Serving no disponible: {exc}") from exc
            logger.warning("Challenger falló (%s); se usa champion", exc)
            variant = "champion"
            result = state["client"].invoke(ENDPOINT, variant, [record], {"explain": request.explain})[0]
        latency = (time.perf_counter() - start) * 1000
        factors = result.get("top_factors") or "[]"
        if isinstance(factors, str):
            import json

            factors = json.loads(factors)
        response = PredictionResponse(
            request_id=str(uuid.uuid4()),
            loan_id=loan_id,
            probability=float(result["probability"]),
            decision=str(result["decision"]),
            risk_band=str(result["risk_band"]),
            top_factors=factors,
            variant=variant,
            model_version=str(served.get(variant, "unknown")),
            latency_ms=round(latency, 2),
        )
        state["sink"].log_prediction(
            {
                **record,
                "request_id": response.request_id,
                "loan_id": loan_id,
                "event_ts": _now(),
                "source": "api",
                "variant": variant,
                "model_version": response.model_version,
                "probability": response.probability,
                "decision": response.decision,
                "risk_band": response.risk_band,
                "latency_ms": response.latency_ms,
            }
        )
        return response

    @app.post("/api/v1/outcomes")
    def outcomes(request: OutcomeRequest) -> dict:
        state["sink"].log_outcome(
            {
                "loan_id": request.loan_id,
                "request_id": request.request_id,
                "actual_default": int(request.actual_default),
                "observed_ts": _now(),
                "source": "api",
            }
        )
        return {"status": "registrado", "loan_id": request.loan_id}

    @app.get("/api/v1/monitoring/summary")
    def monitoring_summary() -> dict:
        if not WAREHOUSE_ID:
            return {"available": False, "detail": "Falta el recurso sql-warehouse en la app"}
        queries = {
            "monitoring": f"SELECT * FROM {FQ_SCHEMA}.monitoring_metrics ORDER BY clock_month DESC, run_ts DESC LIMIT 36",
            "ab_tests": f"SELECT * FROM {FQ_SCHEMA}.ab_test_results ORDER BY run_ts DESC LIMIT 10",
            "retrains": f"SELECT * FROM {FQ_SCHEMA}.retrain_events ORDER BY event_ts DESC LIMIT 10",
        }
        out: dict = {"available": True}
        for key, sql in queries.items():
            try:
                out[key] = state["client"].sql(WAREHOUSE_ID, sql)
            except Exception as exc:
                out[key], out.setdefault("errors", {})[key] = [], str(exc)
        return out

    return app


app = create_app()
