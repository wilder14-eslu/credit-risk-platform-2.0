"""API de scoring crediticio desplegada como Databricks App (FastAPI).

- Autenticación: la de Databricks Apps (OAuth del workspace), sin tokens en el código.
- Modelo: Databricks Model Serving (champion y challenger en el mismo endpoint). El modelo
  solo estima la PD; la decisión la toma la política versionada (`policy.py`).
- Robustez: timeout por llamada, circuit breaker y respuesta prudente (REVISION_MANUAL)
  si el modelo no responde o su salida no cumple los invariantes.
- Velocidad: caché de versiones servidas, idempotencia por solicitud y escritura en Delta
  en segundo plano.
- Seguridad: validación estricta con listas cerradas, límite de tamaño del cuerpo y SQL
  parametrizado.
- A/B testing: asignación determinista por `loan_id` y registro de la variante.
- Feedback loop: predicciones y desenlaces se escriben en Delta (`inference_log`, `outcomes`).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

import policy as P
from ab import assign_variant
from databricks_client import DatabricksClient
from resilience import CircuitBreaker, TimeoutCaller, TTLCache
from schemas import OutcomeRequest, PredictionRequest, PredictionResponse
from sinks import DeltaSink, MemorySink

logger = logging.getLogger("credit-risk-api")
logging.basicConfig(level=logging.INFO)

ENDPOINT = os.getenv("SERVING_ENDPOINT", "credit-risk-lc-endpoint")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
FQ_SCHEMA = f"{os.getenv('UC_CATALOG', 'workspace')}.{os.getenv('UC_SCHEMA', 'credit_risk')}"
CHALLENGER_TRAFFIC = float(os.getenv("CHALLENGER_TRAFFIC", "0.2"))
VERSION_CACHE_SECONDS = 60.0
VERSION_RETRY_SECONDS = 10.0  # tras un fallo no se reintenta en cada solicitud
SERVING_TIMEOUT_SECONDS = float(os.getenv("SERVING_TIMEOUT_SECONDS", "3"))
MAX_BODY_BYTES = 16 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def create_app(client=None, sink=None, breaker: CircuitBreaker | None = None, timeout_seconds: float | None = None):
    state: dict = {"client": client, "sink": sink, "versions": {}, "fetched": None, "next_try": 0.0}
    breaker = breaker or CircuitBreaker(failure_threshold=3, reset_seconds=30.0)
    timeout_seconds = SERVING_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    call_with_timeout = TimeoutCaller()
    responses = TTLCache(max_items=5000, ttl_seconds=900.0)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["client"] is None:
            state["client"] = DatabricksClient()
        if state["sink"] is None:
            state["sink"] = DeltaSink(state["client"], WAREHOUSE_ID, FQ_SCHEMA) if WAREHOUSE_ID else MemorySink()
        yield
        state["sink"].close()

    app = FastAPI(
        title="Credit Risk API (Databricks App)",
        version="2.0.0",
        description="Scoring de riesgo crediticio Lending Club sobre Databricks Model Serving con A/B testing.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def limit_body_and_harden(request: Request, call_next):
        size = request.headers.get("content-length")
        if size is not None and (not size.isdigit() or int(size) > MAX_BODY_BYTES):
            return JSONResponse({"detail": "Cuerpo de la solicitud demasiado grande"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    def versions() -> dict[str, str]:
        """Versiones servidas, en caché 60 s. Si la consulta falla, no se reintenta en cada solicitud."""
        now = time.monotonic()
        stale = state["fetched"] is None or now - state["fetched"] > VERSION_CACHE_SECONDS
        if stale and now >= state["next_try"]:
            try:
                state["versions"] = state["client"].served_versions(ENDPOINT)
                state["fetched"] = now
            except Exception as exc:
                state["next_try"] = now + VERSION_RETRY_SECONDS
                logger.warning("No se pudo leer el endpoint: %s", exc)
        return state["versions"]

    def score(variant: str, record: dict) -> dict:
        """Una llamada a Model Serving con timeout y circuit breaker (SHAP incluido: TreeSHAP es barato)."""

        def call() -> dict:
            return call_with_timeout(
                lambda: state["client"].invoke(ENDPOINT, variant, [record], {"explain": True})[0], timeout_seconds
            )

        return breaker.call(call)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """La raíz lleva a la documentación interactiva (Swagger)."""
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "endpoint": ENDPOINT,
            "served": versions(),
            "serving_circuit": breaker.state,
            "policy_version": P.POLICY_VERSION,
        }

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
        start = time.perf_counter()
        record = request.application.model_dump()
        digest = P.input_hash(record)
        cache_key = f"{request.loan_id}:{digest}:{P.POLICY_VERSION}:{request.explain}" if request.loan_id else None
        if cache_key and (hit := responses.get(cache_key)) is not None:
            # Idempotencia: la misma solicitud devuelve la misma respuesta y no se registra dos veces.
            return hit.model_copy(update={"cached": True, "latency_ms": round((time.perf_counter() - start) * 1000, 2)})

        served = versions()
        loan_id = request.loan_id or f"api-{uuid.uuid4().hex[:12]}"
        variant = assign_variant(loan_id, CHALLENGER_TRAFFIC if "challenger" in served else 0.0)
        result, model_ms = None, None
        try:
            t0 = time.perf_counter()
            try:
                result = score(variant, record)
            except Exception as exc:
                if variant != "challenger":
                    raise
                logger.warning("Challenger falló (%s); se usa champion", exc)
                variant = "champion"
                result = score(variant, record)
            model_ms = round((time.perf_counter() - t0) * 1000, 2)
        except Exception as exc:  # timeout, circuito abierto o error del endpoint
            logger.error("Model Serving no disponible (%s): la solicitud va a revisión manual", exc)

        probability, band, factors = None, None, []
        if result is None:
            decision = P.fallback(P.MODEL_UNAVAILABLE)
        else:
            threshold = float(result.get("threshold") or P.POLICY["default_threshold"])
            factors = result.get("top_factors") or "[]"
            factors = json.loads(factors) if isinstance(factors, str) else list(factors)
            try:
                probability = P.check_invariants(
                    result.get("probability"), result.get("risk_band"), threshold, result.get("decision")
                )
                band = P.risk_band(probability)
                decision = P.decide(probability, threshold, record, factors)
            except P.InvariantError as exc:
                logger.error("Salida del modelo incoherente (%s): revisión manual", exc)
                probability, band, factors = None, None, []
                decision = P.fallback(P.INVARIANT_BROKEN)

        show_factors = decision.decision != P.APPROVE or request.explain
        response = PredictionResponse(
            request_id=str(uuid.uuid4()),
            loan_id=loan_id,
            probability=probability,
            decision=decision.decision,
            reasons=decision.reasons,
            risk_band=band,
            top_factors=factors if show_factors else [],
            variant=variant,
            model_version=str(served.get(variant, "unknown")),
            policy_version=decision.policy_version,
            input_hash=digest,
            model_latency_ms=model_ms,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
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
                "model_latency_ms": response.model_latency_ms,
                "policy_version": response.policy_version,
                "reasons": json.dumps(response.reasons),
                "input_hash": digest,
            }
        )
        if cache_key and result is not None:
            responses.put(cache_key, response)
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
