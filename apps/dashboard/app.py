"""Panel de riesgo crediticio (Databricks App con Streamlit).

- Scoring: formulario de solicitud -> Databricks Model Serving (champion).
- Monitoreo: métricas del replay de producción, drift por variable, A/B y reentrenamientos,
  leídos de las tablas Delta con el SQL warehouse (identidad de la app, sin tokens).
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import views
from databricks_client import DatabricksClient

ENDPOINT = os.getenv("SERVING_ENDPOINT", "credit-risk-lc-endpoint")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
FQ = f"{os.getenv('UC_CATALOG', 'workspace')}.{os.getenv('UC_SCHEMA', 'credit_risk')}"

STATUS = views.STATUS

st.set_page_config(page_title="Credit Risk Platform 2.0", page_icon="💳", layout="wide")


@st.cache_resource
def client() -> DatabricksClient:
    return DatabricksClient()


@st.cache_data(ttl=120)
def _query(sql: str) -> pd.DataFrame:
    return pd.DataFrame(client().sql(WAREHOUSE_ID, sql))


def query(sql: str) -> pd.DataFrame:
    """Una tabla que todavía no existe (p. ej. sin corridas de monitoreo) se muestra como vacía."""
    try:
        return _query(sql)
    except Exception as exc:
        st.caption(f"No se pudo leer una tabla: {exc}")
        return pd.DataFrame()


def page_scoring() -> None:
    st.header("Evaluar una solicitud de crédito")
    st.caption(f"El score lo calcula el endpoint `{ENDPOINT}` de Databricks Model Serving (champion).")
    with st.form("app"):
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = st.number_input("Monto solicitado (USD)", 500.0, 40000.0, 12000.0, step=500.0)
            term = st.selectbox("Plazo (meses)", [36, 60])
            int_rate = st.number_input("Tasa de interés (%)", 5.0, 31.0, 13.5)
            grade = st.selectbox("Grado Lending Club", list("ABCDEFG"), index=2)
            sub_grade = st.selectbox("Subgrado", [f"{grade}{i}" for i in range(1, 6)], index=1)
            purpose = st.selectbox(
                "Propósito",
                [
                    "debt_consolidation",
                    "credit_card",
                    "home_improvement",
                    "major_purchase",
                    "small_business",
                    "car",
                    "medical",
                    "other",
                ],
            )
        with c2:
            annual_inc = st.number_input("Ingreso anual (USD)", 1000.0, 2_000_000.0, 65000.0, step=1000.0)
            emp = st.slider("Años en el empleo", 0, 10, 5)
            home = st.selectbox("Vivienda", ["RENT", "MORTGAGE", "OWN", "OTHER"])
            verif = st.selectbox("Verificación de ingresos", ["Source Verified", "Verified", "Not Verified"])
            dti = st.number_input("DTI (%)", 0.0, 100.0, 18.5)
            state = st.text_input("Estado (EE. UU.)", "CA", max_chars=2)
        with c3:
            fico = st.slider("FICO", 600, 850, 702)
            revol_bal = st.number_input("Saldo revolvente (USD)", 0.0, 1_000_000.0, 14500.0)
            revol_util = st.number_input("Utilización revolvente (%)", 0.0, 200.0, 55.0)
            open_acc = st.number_input("Líneas abiertas", 0, 100, 10)
            total_acc = st.number_input("Líneas totales", 0, 200, 24)
            history = st.number_input("Antigüedad crediticia (meses)", 0, 900, 180)
        submitted = st.form_submit_button("Evaluar", type="primary")
    if not submitted:
        return
    r = int_rate / 1200
    installment = loan_amnt * r / (1 - (1 + r) ** (-term))
    record = {
        "loan_amnt": loan_amnt,
        "term_months": term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length_years": emp,
        "home_ownership": home,
        "annual_inc": annual_inc,
        "verification_status": verif,
        "purpose": purpose,
        "addr_state": state.upper(),
        "dti": dti,
        "delinq_2yrs": 0,
        "fico_score": fico,
        "inq_last_6mths": 1,
        "open_acc": open_acc,
        "pub_rec": 0,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "mort_acc": 1 if home == "MORTGAGE" else 0,
        "pub_rec_bankruptcies": 0,
        "credit_history_months": history,
        "application_type": "Individual",
        "initial_list_status": "w",
    }
    try:
        result = client().invoke(ENDPOINT, "champion", [record], {"explain": True})[0]
    except Exception as exc:
        st.error(f"No se pudo consultar Model Serving: {exc}")
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Probabilidad de default", f"{float(result['probability']):.1%}")
    k2.metric("Decisión", result["decision"])
    k3.metric("Banda de riesgo", result["risk_band"])
    k4.metric("Cuota mensual", f"${installment:,.0f}")
    factors = pd.DataFrame(
        json.loads(result["top_factors"]) if isinstance(result["top_factors"], str) else result["top_factors"]
    )
    if not factors.empty:
        color = factors["direction"].map({"aumenta_riesgo": STATUS["alerta"], "reduce_riesgo": STATUS["estable"]})
        fig = go.Figure(
            go.Bar(
                x=factors["impact"],
                y=factors["label"],
                orientation="h",
                marker_color=color,
                hovertemplate="%{y}: %{x:.3f} log-odds<extra></extra>",
            )
        )
        fig.update_layout(title="Factores que más influyeron (SHAP)", height=260, xaxis_title="impacto en log-odds")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(factors[["label", "impact", "direction"]], hide_index=True)


def page_monitoring() -> None:
    st.header("Monitoreo del replay de producción (originación 2014-2015)")
    if not WAREHOUSE_ID:
        st.warning("La app no tiene el recurso sql-warehouse configurado.")
        return
    q = views.queries(FQ)
    views.render_monitoring(
        query(q["monitoring_metrics"]),
        query(q["drift_by_feature"]),
        query(q["ab_test_results"]),
        query(q["retrain_events"]),
    )


def page_models() -> None:
    st.header("Benchmark de modelos (último entrenamiento)")
    if not WAREHOUSE_ID:
        st.warning("La app no tiene el recurso sql-warehouse configurado.")
        return
    try:
        served = client().served_versions(ENDPOINT)
    except Exception as exc:
        st.caption(f"Endpoint no disponible: {exc}")
        served = None
    q = views.queries(FQ)
    views.render_models(query(q["model_benchmark"]), query(q["model_evaluation"]), served)


PAGES = {"Scoring": page_scoring, "Monitoreo": page_monitoring, "Modelos": page_models}
choice = st.sidebar.radio("Navegación", list(PAGES))
st.sidebar.caption("Lending Club 2007-2018 · Databricks Free Edition")
PAGES[choice]()
