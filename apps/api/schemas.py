"""Contrato de la API: variables de la solicitud de crédito (Lending Club)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., ge=500, le=40000, examples=[12000])
    term_months: Literal[36, 60] = Field(36, examples=[36])
    int_rate: float = Field(..., ge=5, le=31, examples=[13.5])
    installment: float = Field(..., ge=0, examples=[407.2])
    grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(..., examples=["C"])
    sub_grade: str = Field(..., pattern=r"^[A-G][1-5]$", examples=["C2"])
    emp_length_years: float | None = Field(None, ge=0, le=10, examples=[5])
    home_ownership: Literal["MORTGAGE", "RENT", "OWN", "OTHER"] = Field(..., examples=["RENT"])
    annual_inc: float = Field(..., gt=0, examples=[65000])
    verification_status: Literal["Not Verified", "Source Verified", "Verified"] = Field(
        ..., examples=["Source Verified"]
    )
    purpose: str = Field(..., examples=["debt_consolidation"])
    addr_state: str = Field(..., min_length=2, max_length=2, examples=["CA"])
    dti: float = Field(..., ge=0, le=100, examples=[18.5])
    delinq_2yrs: float = Field(0, ge=0, examples=[0])
    fico_score: float = Field(..., ge=300, le=850, examples=[702])
    inq_last_6mths: float = Field(0, ge=0, examples=[1])
    open_acc: float = Field(..., ge=0, examples=[10])
    pub_rec: float = Field(0, ge=0, examples=[0])
    revol_bal: float = Field(..., ge=0, examples=[14500])
    revol_util: float | None = Field(None, ge=0, le=200, examples=[55.2])
    total_acc: float = Field(..., ge=0, examples=[24])
    mort_acc: float | None = Field(None, ge=0, examples=[1])
    pub_rec_bankruptcies: float | None = Field(None, ge=0, examples=[0])
    credit_history_months: float = Field(..., ge=0, examples=[180])
    application_type: Literal["Individual", "Joint App"] = Field("Individual", examples=["Individual"])
    initial_list_status: Literal["w", "f"] = Field("w", examples=["w"])


class PredictionRequest(BaseModel):
    loan_id: str | None = Field(None, description="Id estable del solicitante (A/B sticky)")
    application: LoanApplication
    explain: bool = True


class Factor(BaseModel):
    feature: str
    label: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    request_id: str
    loan_id: str
    probability: float
    decision: str
    risk_band: str
    top_factors: list[Factor]
    variant: str
    model_version: str
    latency_ms: float


class OutcomeRequest(BaseModel):
    loan_id: str
    request_id: str | None = None
    actual_default: bool
