"""Política de crédito: el modelo estima el riesgo, la política decide.

El modelo entrega una probabilidad de default (PD). La decisión final
(APROBAR, RECHAZAR o REVISION_MANUAL) la toma esta capa determinista y versionada,
con códigos de motivo auditables. Así se puede cambiar un criterio de negocio
(umbral, zona gris, reglas duras) por un PR revisado en CI/CD, sin reentrenar.

Principios:
- Prudencia: ante duda o falla, nunca se aprueba de forma automática (REVISION_MANUAL).
- Invariantes: la salida del modelo se verifica antes de usarse; si es incoherente,
  se rechaza con un error explícito en vez de corregir números en silencio.

IMPORTANTE: este archivo se copia tal cual en `apps/api/policy.py` (la app se
despliega sin el paquete del repo). Un test de CI verifica que ambas copias sean
idénticas. Usa solo la biblioteca estándar.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

POLICY_VERSION = "2026.10-1"

POLICY: dict[str, float | int] = {
    # Umbral de respaldo si el modelo servido no informa el suyo (el de cada versión se
    # elige en el entrenamiento minimizando el costo esperado con FN:FP = 5:1).
    "default_threshold": 0.18,
    # Zona gris: PD en [umbral - review_band, umbral) va a un analista en vez de aprobarse sola.
    "review_band": 0.02,
    # Reglas duras (knock-out), independientes del modelo:
    "max_payment_to_income": 0.40,  # cuota mensual / ingreso mensual
    "max_dti": 45.0,  # deuda / ingreso declarado (%)
    "max_reason_codes": 4,
}

APPROVE, REJECT, REVIEW = "APROBAR", "RECHAZAR", "REVISION_MANUAL"
DECISIONS = (APPROVE, REJECT, REVIEW)

# Códigos de motivo (los de riesgo del modelo se agregan como RIESGO_<VARIABLE>).
PD_HIGH = "PD_SOBRE_UMBRAL"
GREY_ZONE = "PD_ZONA_GRIS"
PAYMENT_TO_INCOME = "CUOTA_INGRESO_ALTA"
DTI_HIGH = "DTI_ALTO"
MODEL_UNAVAILABLE = "MODELO_NO_DISPONIBLE"
INVARIANT_BROKEN = "SALIDA_MODELO_INCOHERENTE"

_BANDS = ((0.05, "A"), (0.10, "B"), (0.20, "C"), (0.35, "D"))


def risk_band(probability: float) -> str:
    for limit, band in _BANDS:
        if probability < limit:
            return band
    return "E"


@dataclass(frozen=True)
class Decision:
    decision: str
    reasons: list[str] = field(default_factory=list)
    policy_version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"Decisión desconocida: {self.decision}")
        if self.decision != APPROVE and not self.reasons:
            raise ValueError("Toda decisión distinta de APROBAR necesita al menos un código de motivo")


class InvariantError(ValueError):
    """La salida del modelo no cumple el contrato: no se usa para decidir."""


def check_invariants(probability: Any, band: str | None, threshold: float, model_decision: str | None = None) -> float:
    """Verifica la salida del modelo y devuelve la PD como float. Lanza InvariantError si es incoherente."""
    try:
        p = float(probability)
    except (TypeError, ValueError) as exc:
        raise InvariantError(f"PD no numérica: {probability!r}") from exc
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise InvariantError(f"PD fuera de [0, 1]: {p}")
    if not 0.0 < threshold < 1.0:
        raise InvariantError(f"Umbral fuera de (0, 1): {threshold}")
    if band is not None and band != risk_band(p):
        raise InvariantError(f"Banda {band} incoherente con PD {p:.4f} (esperada {risk_band(p)})")
    if model_decision is not None and model_decision in (APPROVE, REJECT):
        if (model_decision == REJECT) != (p >= threshold):
            raise InvariantError(
                f"Decisión del modelo {model_decision} incoherente con PD {p:.4f} y umbral {threshold}"
            )
    return p


def _rule_reasons(application: dict[str, Any], policy: dict) -> list[str]:
    reasons = []
    income = float(application.get("annual_inc") or 0.0)
    installment = float(application.get("installment") or 0.0)
    if income > 0 and installment / (income / 12.0) > policy["max_payment_to_income"]:
        reasons.append(PAYMENT_TO_INCOME)
    dti = application.get("dti")
    if dti is not None and float(dti) > policy["max_dti"]:
        reasons.append(DTI_HIGH)
    return reasons


def _risk_reasons(factors: list[dict] | None) -> list[str]:
    """Variables que más subieron la PD (contribuciones SHAP positivas), en orden de impacto."""
    out = []
    for f in sorted(factors or [], key=lambda f: -float(f.get("impact", 0.0))):
        if float(f.get("impact", 0.0)) > 0:
            out.append("RIESGO_" + str(f.get("feature", "")).upper())
    return out


def decide(
    probability: float,
    threshold: float,
    application: dict[str, Any],
    factors: list[dict] | None = None,
    policy: dict | None = None,
) -> Decision:
    """Decisión determinista: reglas duras, luego PD contra umbral y zona gris."""
    policy = POLICY if policy is None else policy
    limit = int(policy["max_reason_codes"])
    rules = _rule_reasons(application, policy)
    if rules:
        return Decision(REJECT, (rules + _risk_reasons(factors))[:limit])
    if probability >= threshold:
        return Decision(REJECT, ([PD_HIGH] + _risk_reasons(factors))[:limit])
    if probability >= threshold - float(policy["review_band"]):
        return Decision(REVIEW, ([GREY_ZONE] + _risk_reasons(factors))[:limit])
    return Decision(APPROVE, [])


def fallback(reason: str = MODEL_UNAVAILABLE) -> Decision:
    """Lectura prudente: sin una PD confiable nunca se aprueba automáticamente."""
    return Decision(REVIEW, [reason])


def input_hash(application: dict[str, Any]) -> str:
    """Huella reproducible de la solicitud (mismas variables y valores -> mismo hash)."""
    canonical = json.dumps(application, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
