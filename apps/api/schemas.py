"""Contrato de la API: variables de la solicitud de crédito (Lending Club)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Listas cerradas: todo dato externo es hostil hasta validarse (valores fuera de lista -> 422).
Purpose = Literal[
    "debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "medical",
    "small_business", "car", "vacation", "moving", "house", "wedding", "renewable_energy", "educational",
]  # fmt: skip
State = Literal[
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS",
    "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV",
    "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]  # fmt: skip
STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LoanApplication(BaseModel):
    model_config = STRICT

    loan_amnt: float = Field(..., ge=500, le=40000, examples=[12000])
    term_months: Literal[36, 60] = Field(36, examples=[36])
    int_rate: float = Field(..., ge=5, le=31, examples=[13.5])
    installment: float = Field(..., ge=0, le=5000, examples=[407.2])
    grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(..., examples=["C"])
    sub_grade: str = Field(..., pattern=r"^[A-G][1-5]$", examples=["C2"])
    emp_length_years: float | None = Field(None, ge=0, le=10, examples=[5])
    home_ownership: Literal["MORTGAGE", "RENT", "OWN", "OTHER"] = Field(..., examples=["RENT"])
    annual_inc: float = Field(..., gt=0, le=10_000_000, examples=[65000])
    verification_status: Literal["Not Verified", "Source Verified", "Verified"] = Field(
        ..., examples=["Source Verified"]
    )
    purpose: Purpose = Field(..., examples=["debt_consolidation"])
    addr_state: State = Field(..., examples=["CA"])
    dti: float = Field(..., ge=0, le=100, examples=[18.5])
    delinq_2yrs: float = Field(0, ge=0, le=200, examples=[0])
    fico_score: float = Field(..., ge=300, le=850, examples=[702])
    inq_last_6mths: float = Field(0, ge=0, le=200, examples=[1])
    open_acc: float = Field(..., ge=0, le=200, examples=[10])
    pub_rec: float = Field(0, ge=0, le=200, examples=[0])
    revol_bal: float = Field(..., ge=0, le=10_000_000, examples=[14500])
    revol_util: float | None = Field(None, ge=0, le=200, examples=[55.2])
    total_acc: float = Field(..., ge=0, le=200, examples=[24])
    mort_acc: float | None = Field(None, ge=0, le=200, examples=[1])
    pub_rec_bankruptcies: float | None = Field(None, ge=0, le=200, examples=[0])
    credit_history_months: float = Field(..., ge=0, le=1200, examples=[180])
    application_type: Literal["Individual", "Joint App"] = Field("Individual", examples=["Individual"])
    initial_list_status: Literal["w", "f"] = Field("w", examples=["w"])

    @model_validator(mode="after")
    def _sub_grade_matches_grade(self):
        if not self.sub_grade.startswith(self.grade):
            raise ValueError(f"sub_grade {self.sub_grade} no corresponde al grade {self.grade}")
        return self


class PredictionRequest(BaseModel):
    model_config = STRICT

    loan_id: str | None = Field(
        None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.:-]+$",
        description="Id estable del solicitante: A/B sticky e idempotencia (misma solicitud, misma respuesta)",
    )
    application: LoanApplication
    explain: bool = Field(
        False,
        description="Incluir factores SHAP también en aprobaciones (en rechazos y revisiones van siempre)",
    )


class Factor(BaseModel):
    feature: str
    label: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    request_id: str
    loan_id: str
    probability: float | None = Field(description="PD estimada; null si el modelo no estuvo disponible")
    decision: Literal["APROBAR", "RECHAZAR", "REVISION_MANUAL"]
    reasons: list[str] = Field(description="Códigos de motivo de rechazos y revisiones manuales")
    risk_band: str | None
    top_factors: list[Factor]
    variant: str
    model_version: str
    policy_version: str
    input_hash: str
    cached: bool = False
    model_latency_ms: float | None = None
    latency_ms: float


class OutcomeRequest(BaseModel):
    model_config = STRICT

    loan_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    request_id: str | None = Field(None, max_length=64)
    actual_default: bool
