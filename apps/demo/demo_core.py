"""Lógica de la demo pública (Streamlit Community Cloud), sin dependencias de UI.

La demo es autocontenida: lleva el modelo champion exportado de Unity Catalog
(`model/credit_model.joblib`), el paquete `credit_risk` y la configuración YAML.
No llama a Databricks: funciona sin tokens ni endpoints encendidos.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent

# Debe definirse ANTES de importar credit_risk: config.py lee la ruta al importarse.
os.environ.setdefault("CREDIT_RISK_CONFIG_DIR", str(APP_DIR / "config"))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if not (APP_DIR / "views.py").exists():  # desde el repo: las páginas viven en apps/dashboard
    sys.path.append(str(APP_DIR.parent / "dashboard"))

import pandas as pd  # noqa: E402

PURPOSES = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "small_business",
    "car",
    "medical",
    "other",
]
HOME = ["RENT", "MORTGAGE", "OWN", "OTHER"]
VERIFICATION = ["Source Verified", "Verified", "Not Verified"]
GRADES = list("ABCDEFG")
SUB_GRADES = [f"{g}{i}" for g in GRADES for i in range(1, 6)]

# Orden de los campos del formulario (lo comparten la UI, los ejemplos y los tests).
FORM_FIELDS = [
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "purpose",
    "annual_inc",
    "emp",
    "home",
    "verif",
    "dti",
    "state",
    "fico",
    "revol_bal",
    "revol_util",
    "open_acc",
    "total_acc",
    "history",
]

EXAMPLES = {
    "Perfil conservador": [10000, 36, 7.5, "A", "A3", "credit_card", 95000, 8, "MORTGAGE", "Verified", 9.0, "TX",
                           760, 6000, 22, 9, 28, 260],
    "Perfil típico": [12000, 36, 13.5, "C", "C2", "debt_consolidation", 65000, 5, "RENT", "Source Verified", 18.5,
                      "CA", 702, 14500, 55, 10, 24, 180],
    "Perfil riesgoso": [30000, 60, 24.0, "F", "F2", "small_business", 38000, 1, "RENT", "Not Verified", 34.0, "NV",
                        662, 21000, 92, 15, 18, 60],
}  # fmt: skip


def installment(loan_amnt: float, rate_pct: float, term: int) -> float:
    """Cuota francesa mensual (como la calcula Lending Club)."""
    r = rate_pct / 1200
    return loan_amnt / term if r == 0 else loan_amnt * r / (1 - (1 + r) ** (-term))


def build_record(**form: Any) -> dict[str, Any]:
    """Del formulario a las 26 variables de entrada del modelo (mismo mapeo que el dashboard)."""
    term = int(form["term"])
    loan = float(form["loan_amnt"])
    rate = float(form["int_rate"])
    return {
        "loan_amnt": loan,
        "term_months": term,
        "int_rate": rate,
        "installment": installment(loan, rate, term),
        "grade": form["grade"],
        "sub_grade": form["sub_grade"],
        "emp_length_years": float(form["emp"]),
        "home_ownership": form["home"],
        "annual_inc": float(form["annual_inc"]),
        "verification_status": form["verif"],
        "purpose": form["purpose"],
        "addr_state": str(form["state"]).strip().upper()[:2] or "CA",
        "dti": float(form["dti"]),
        "delinq_2yrs": 0.0,
        "fico_score": float(form["fico"]),
        "inq_last_6mths": 1.0,
        "open_acc": float(form["open_acc"]),
        "pub_rec": 0.0,
        "revol_bal": float(form["revol_bal"]),
        "revol_util": float(form["revol_util"]),
        "total_acc": float(form["total_acc"]),
        "mort_acc": 1.0 if form["home"] == "MORTGAGE" else 0.0,
        "pub_rec_bankruptcies": 0.0,
        "credit_history_months": float(form["history"]),
        "application_type": "Individual",
        "initial_list_status": "w",
    }


@lru_cache(maxsize=1)
def load_model(path: str | None = None):
    import joblib

    return joblib.load(path or APP_DIR / "model" / "credit_model.joblib")


@lru_cache(maxsize=1)
def model_info() -> dict[str, Any]:
    f = APP_DIR / "model" / "model_info.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def figures_dir() -> Path | None:
    """Figuras del análisis: las de la demo o, al correr desde el repo, docs/figures."""
    return _first_existing(APP_DIR / "figures", APP_DIR.parents[1] / "docs" / "figures")


@lru_cache(maxsize=1)
def results() -> dict[str, Any]:
    f = _first_existing(APP_DIR / "results" / "results.json", APP_DIR.parents[1] / "reports" / "results.json")
    return json.loads(f.read_text(encoding="utf-8")) if f else {}


def score(record: dict[str, Any], model=None) -> dict[str, Any]:
    """PD, decisión, banda de riesgo y los factores (SHAP) que más pesaron."""
    model = model or load_model()
    out = model.predict_frame(pd.DataFrame([record]), explain=True, top_k=5).iloc[0]
    return {
        "probability": float(out["probability"]),
        "decision": str(out["decision"]),
        "risk_band": str(out["risk_band"]),
        "threshold": float(model.threshold),
        "installment": float(record["installment"]),
        "factors": json.loads(out["top_factors"]),
    }


SNAPSHOT_TABLES = (
    "monitoring_metrics",
    "drift_by_feature",
    "ab_test_results",
    "retrain_events",
    "model_benchmark",
    "model_evaluation",
)


def snapshot(table: str) -> pd.DataFrame:
    """Foto de una tabla Delta de producción exportada al publicar la demo (vacía si no existe)."""
    f = APP_DIR / "snapshot" / f"{table}.json"
    return pd.DataFrame(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def snapshot_info() -> dict[str, Any]:
    f = APP_DIR / "snapshot" / "snapshot_info.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
