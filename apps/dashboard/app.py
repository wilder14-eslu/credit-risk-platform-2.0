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

from databricks_client import DatabricksClient

ENDPOINT = os.getenv("SERVING_ENDPOINT", "credit-risk-lc-endpoint")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
FQ = f"{os.getenv('UC_CATALOG', 'workspace')}.{os.getenv('UC_SCHEMA', 'credit_risk')}"

# Paleta validada: identidad por entidad y colores de estado reservados (siempre con texto/ícono).
SERIES = {"observed": "#2a78d6", "predicted": "#eb6834"}
STATUS = {"estable": "#0ca30c", "warning": "#fab219", "alerta": "#d03b3b"}
ICON = {"estable": "✅", "warning": "⚠️", "alerta": "🛑", "critico": "🛑", "sin_datos": "ℹ️"}

st.set_page_config(page_title="Credit Risk Platform 2.0", page_icon="💳", layout="wide")


@st.cache_resource
def client() -> DatabricksClient:
    return DatabricksClient()


@st.cache_data(ttl=120)
def query(sql: str) -> pd.DataFrame:
    return pd.DataFrame(client().sql(WAREHOUSE_ID, sql))


def num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


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
    st.header("Monitoreo del replay de producción (2015 en adelante)")
    if not WAREHOUSE_ID:
        st.warning("La app no tiene el recurso sql-warehouse configurado.")
        return
    mon = query(f"SELECT * FROM {FQ}.monitoring_metrics ORDER BY clock_month")
    if mon.empty:
        st.info("Aún no hay corridas de monitoreo. Ejecuta el job credit-risk-production-monitoring.")
        return
    mon = num(mon, ["roc_auc", "prediction_psi", "default_rate_observed", "default_rate_predicted", "auc_drop"])
    mon["clock_month"] = pd.to_datetime(mon["clock_month"])
    last = mon.iloc[-1]
    sev = str(last["severity"])
    st.subheader(f"{ICON.get(sev, 'ℹ️')} Último chequeo ({last['clock_month']:%Y-%m}): {sev}")
    for reason in json.loads(last["reasons"] or "[]"):
        st.write(f"- {reason}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mon["clock_month"],
            y=mon["roc_auc"],
            mode="lines+markers",
            name="AUC en producción",
            line=dict(color=SERIES["observed"], width=2),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title="AUC con desenlaces reales (concept drift)",
        hovermode="x unified",
        yaxis_title="ROC-AUC",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mon["clock_month"],
            y=mon["default_rate_observed"],
            name="Default observado",
            line=dict(color=SERIES["observed"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mon["clock_month"],
            y=mon["default_rate_predicted"],
            name="Default predicho",
            line=dict(color=SERIES["predicted"], width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Calibración: tasa de default observada vs predicha", hovermode="x unified", yaxis_tickformat=".0%"
    )
    st.plotly_chart(fig, use_container_width=True)

    drift = query(f"""SELECT * FROM {FQ}.drift_by_feature
                      WHERE run_id = (SELECT run_id FROM {FQ}.monitoring_metrics ORDER BY run_ts DESC LIMIT 1)""")
    if not drift.empty:
        drift = num(drift, ["psi", "ks_pvalue"]).sort_values("psi")
        fig = go.Figure(
            go.Bar(
                x=drift["psi"],
                y=drift["feature"],
                orientation="h",
                marker_color=drift["status"].map(STATUS).fillna(STATUS["estable"]),
                customdata=drift["status"],
                hovertemplate="%{y}: PSI %{x:.3f} (%{customdata})<extra></extra>",
            )
        )
        fig.add_vline(x=0.10, line_dash="dot", annotation_text="warning 0.10")
        fig.add_vline(x=0.25, line_dash="dot", annotation_text="alerta 0.25")
        fig.update_layout(title="Data drift por variable (PSI vs entrenamiento)", height=620)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(drift[["feature", "kind", "psi", "ks_pvalue", "status"]], hide_index=True)

    st.subheader("A/B testing champion vs challenger")
    ab = query(f"SELECT * FROM {FQ}.ab_test_results ORDER BY run_ts DESC LIMIT 10")
    st.dataframe(ab, hide_index=True) if not ab.empty else st.caption("Sin evaluaciones A/B todavía.")
    st.subheader("Reentrenamientos disparados por el monitoreo")
    rt = query(f"SELECT * FROM {FQ}.retrain_events ORDER BY event_ts DESC LIMIT 10")
    st.dataframe(rt, hide_index=True) if not rt.empty else st.caption("Ninguno todavía.")


def page_models() -> None:
    st.header("Benchmark de modelos (último entrenamiento)")
    if not WAREHOUSE_ID:
        return
    bench = query(f"""SELECT * FROM {FQ}.model_benchmark
                      WHERE run_ts = (SELECT max(run_ts) FROM {FQ}.model_benchmark)""")
    st.dataframe(bench, hide_index=True) if not bench.empty else st.caption("Sin entrenamientos registrados.")
    try:
        st.json(client().served_versions(ENDPOINT))
    except Exception as exc:
        st.caption(f"Endpoint no disponible: {exc}")


PAGES = {"Scoring": page_scoring, "Monitoreo": page_monitoring, "Modelos": page_models}
choice = st.sidebar.radio("Navegación", list(PAGES))
st.sidebar.caption("Lending Club 2007-2018 · Databricks Free Edition")
PAGES[choice]()
