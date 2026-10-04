"""Páginas de Monitoreo y Modelos a partir de DataFrames (sin acceso a datos).

Las usan dos apps con la misma presentación:
- el dashboard de Databricks Apps, con las tablas Delta leídas en vivo por el SQL warehouse;
- la demo pública de Streamlit Cloud, con una foto de esas tablas exportada al publicarla.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Paleta validada: identidad por entidad y colores de estado reservados (siempre con texto/ícono).
SERIES = {"observed": "#2a78d6", "predicted": "#eb6834"}
STATUS = {"estable": "#0ca30c", "warning": "#fab219", "alerta": "#d03b3b"}
ICON = {"estable": "✅", "warning": "⚠️", "alerta": "🛑", "critico": "🛑", "sin_datos": "ℹ️"}

MONITORING_NUMERIC = ["roc_auc", "prediction_psi", "default_rate_observed", "default_rate_predicted", "auc_drop"]


def queries(fq: str) -> dict[str, str]:
    """SQL de cada tabla que muestran las páginas (`fq` = catalog.schema)."""
    return {
        "monitoring_metrics": f"SELECT * FROM {fq}.monitoring_metrics ORDER BY clock_month",
        "drift_by_feature": f"""SELECT * FROM {fq}.drift_by_feature WHERE run_id =
            (SELECT run_id FROM {fq}.monitoring_metrics ORDER BY run_ts DESC LIMIT 1)""",
        "ab_test_results": f"SELECT * FROM {fq}.ab_test_results ORDER BY run_ts DESC LIMIT 10",
        "retrain_events": f"SELECT * FROM {fq}.retrain_events ORDER BY event_ts DESC LIMIT 10",
        "model_benchmark": f"""SELECT * FROM {fq}.model_benchmark
            WHERE run_ts = (SELECT max(run_ts) FROM {fq}.model_benchmark)""",
        "model_evaluation": f"""SELECT * FROM {fq}.model_evaluation
            WHERE run_ts = (SELECT max(run_ts) FROM {fq}.model_evaluation)""",
    }


def num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _reasons(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        return [str(v) for v in json.loads(value or "[]")]
    except (TypeError, ValueError):
        return [str(value)] if value else []


def render_monitoring(mon: pd.DataFrame, drift: pd.DataFrame, ab: pd.DataFrame, rt: pd.DataFrame) -> None:
    """mon: monitoring_metrics; drift: drift_by_feature de la última corrida; ab y rt: últimos eventos."""
    if mon.empty:
        st.info("Aún no hay corridas de monitoreo. Ejecuta el job credit-risk-production-monitoring.")
        return
    mon = num(mon.copy(), MONITORING_NUMERIC)
    mon["clock_month"] = pd.to_datetime(mon["clock_month"])
    mon = mon.sort_values("clock_month")
    last = mon.iloc[-1]
    sev = str(last["severity"])
    st.subheader(f"{ICON.get(sev, 'ℹ️')} Último chequeo ({last['clock_month']:%Y-%m}): {sev}")
    for reason in _reasons(last.get("reasons")):
        st.write(f"- {reason}")

    fig = go.Figure(
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
    fig.update_xaxes(tickformat="%Y-%m", dtick="M1")
    st.plotly_chart(fig)

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
    fig.update_xaxes(tickformat="%Y-%m", dtick="M1")
    st.plotly_chart(fig)

    if not drift.empty:
        drift = num(drift.copy(), ["psi", "ks_pvalue"]).sort_values("psi")
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
        st.plotly_chart(fig)
        cols = [c for c in ("feature", "kind", "psi", "ks_pvalue", "status") if c in drift]
        st.dataframe(drift[cols], hide_index=True)

    st.subheader("A/B testing champion vs challenger")
    if ab.empty:
        st.caption("Sin evaluaciones A/B todavía.")
    else:
        st.dataframe(ab, hide_index=True)
    st.subheader("Reentrenamientos disparados por el monitoreo")
    if rt.empty:
        st.caption("Ninguno todavía.")
    else:
        st.dataframe(rt, hide_index=True)


def render_models(bench: pd.DataFrame, evaluation: pd.DataFrame, served: dict | None = None) -> None:
    """bench: model_benchmark del último entrenamiento; evaluation: model_evaluation (IC de DeLong)."""
    if bench.empty and evaluation.empty:
        st.caption("Sin entrenamientos registrados.")
    if not evaluation.empty:
        ev = num(evaluation.copy(), ["auc", "auc_ci_low", "auc_ci_high", "p_holm"])
        for split, title in (("validation", "Validación (aquí se decide)"), ("test", "Test fuera de tiempo")):
            part = ev[ev["split"] == split].sort_values("auc")
            if part.empty:
                continue
            chosen = part["selected"].astype(str).str.lower().eq("true")
            fig = go.Figure(
                go.Scatter(
                    x=part["auc"],
                    y=part["model"],
                    mode="markers",
                    marker=dict(size=11, color=[SERIES["observed"] if s else "#8a8a8a" for s in chosen]),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=part["auc_ci_high"] - part["auc"],
                        arrayminus=part["auc"] - part["auc_ci_low"],
                        color="#8a8a8a",
                    ),
                    hovertemplate="%{y}: AUC %{x:.4f}<extra></extra>",
                )
            )
            fig.update_layout(title=f"AUC con IC 95 % (DeLong): {title}", xaxis_title="ROC-AUC", height=300)
            st.plotly_chart(fig)
        st.caption("En azul el modelo elegido por la regla DeLong + Holm + parsimonia.")
    if not bench.empty:
        st.subheader("Benchmark del último entrenamiento")
        st.dataframe(bench, hide_index=True)
    if served:
        st.subheader("Versiones servidas")
        st.json(served)
