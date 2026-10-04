"""Demo pública de Credit Risk Platform 2.0 (Streamlit Community Cloud).

Mismas páginas que el dashboard de Databricks (Scoring, Monitoreo, Modelos, con `views.py`
compartido) más el análisis estadístico y la arquitectura. Sin credenciales: el scoring usa el
champion exportado y Monitoreo/Modelos una foto de las tablas Delta tomada al publicar.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import demo_core as D
import views

REPO = "https://github.com/wilder14-eslu/credit-risk-platform-2.0"
RISK, SAFE = "#d03b3b", "#0ca30c"
FIGURES = [
    ("auc_ci.png", "AUC por modelo en el test fuera de tiempo, con IC 95 % (DeLong)"),
    ("pairwise.png", "Comparaciones por pares: diferencia de AUC, p ajustado por Holm y margen práctico"),
    ("roc_pr.png", "Curvas ROC y Precision-Recall en el test fuera de tiempo"),
    ("calibration.png", "Calibración: PD predicha vs tasa observada (IC de Wilson)"),
    ("deciles.png", "Tasa de default y lift por decil de riesgo"),
    ("threshold.png", "Elección del umbral por costo esperado (FN:FP = 5:1)"),
    ("vintages.png", "Estabilidad por cosecha trimestral, incluido el periodo de producción 2014-2015"),
    ("shap.png", "Importancia SHAP y dirección del efecto"),
    ("shap_dependence.png", "Dependencia SHAP de las variables numéricas más importantes"),
    ("iv.png", "Information Value del scorecard WoE (benchmark interpretable)"),
]

ARCHITECTURE = f"""
### Qué hay detrás de esta demo

Esta página usa una copia del modelo **champion** registrado en Unity Catalog. La plataforma completa corre en
**Databricks Free Edition** y está definida como código con Databricks Asset Bundles:

1. **Datos:** 2.26 millones de préstamos de Lending Club (2007-2018) en una arquitectura medallion
   (Bronze, Silver, Gold) sobre Delta Lake, con validaciones de calidad y solo variables conocidas al solicitar
   (sin data leakage).
2. **Entrenamiento:** benchmark de regresión logística, XGBoost, LightGBM y CatBoost más un scorecard WoE;
   selección estadística en validación (DeLong + Holm + parsimonia), tuning con Optuna y quality gates.
   Partición temporal: entrenamiento 2007-2012, validación 2013-S1, test 2013-S2.
3. **Registro y despliegue:** MLflow con Unity Catalog (aliases champion/challenger), lineage código-datos-modelo
   (git SHA y versión Delta), Model Serving con A/B testing, API FastAPI y dashboard Streamlit como Databricks Apps.
4. **Producción simulada:** replay mes a mes de la originación 2014-2015, monitoreo de data, prediction y concept
   drift, y **reentrenamiento continuo** disparado por drift con validación champion vs challenger y rollback.
5. **CI/CD:** GitHub Actions con entornos dev, staging y prod (aprobación manual), 90+ tests y CodeQL. Esta demo
   se publica con un workflow que exporta el champion, prueba una predicción y actualiza la rama `demo`.

Código, reporte técnico y resultados completos: [{REPO}]({REPO})

> Demo educativa con datos históricos públicos. No es una herramienta de decisión crediticia real.
"""


@st.cache_resource
def model():
    return D.load_model()


def factors_figure(factors: list[dict]) -> go.Figure:
    factors = sorted(factors, key=lambda f: abs(f["impact"]))
    fig = go.Figure(
        go.Bar(
            x=[f["impact"] for f in factors],
            y=[f["label"] for f in factors],
            orientation="h",
            marker_color=[RISK if f["impact"] > 0 else SAFE for f in factors],
            text=[f"{f['impact']:+.3f}" for f in factors],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=40, t=10, b=30),
        xaxis_title="Contribución al log-odds de default",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def page_scoring() -> None:
    profile = st.radio("Perfil de ejemplo", list(D.EXAMPLES), index=1, horizontal=True)
    v = dict(zip(D.FORM_FIELDS, D.EXAMPLES[profile], strict=True))
    k = profile  # las claves cambian con el perfil para recargar los valores del formulario
    with st.form("solicitud"):
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = st.number_input("Monto solicitado (USD)", 500, 40000, v["loan_amnt"], 500, key=f"a{k}")
            term = st.selectbox("Plazo (meses)", [36, 60], [36, 60].index(v["term"]), key=f"b{k}")
            int_rate = st.slider("Tasa de interés (%)", 5.0, 31.0, float(v["int_rate"]), 0.1, key=f"c{k}")
            grade = st.selectbox("Grado Lending Club", D.GRADES, D.GRADES.index(v["grade"]), key=f"d{k}")
            sub_grade = st.selectbox("Subgrado", D.SUB_GRADES, D.SUB_GRADES.index(v["sub_grade"]), key=f"e{k}")
            purpose = st.selectbox("Propósito", D.PURPOSES, D.PURPOSES.index(v["purpose"]), key=f"f{k}")
        with c2:
            annual_inc = st.number_input("Ingreso anual (USD)", 1000, 2_000_000, v["annual_inc"], 1000, key=f"g{k}")
            emp = st.slider("Años en el empleo", 0, 10, v["emp"], key=f"h{k}")
            home = st.selectbox("Vivienda", D.HOME, D.HOME.index(v["home"]), key=f"i{k}")
            verif = st.selectbox(
                "Verificación de ingresos", D.VERIFICATION, D.VERIFICATION.index(v["verif"]), key=f"j{k}"
            )
            dti = st.slider("Deuda / ingreso, DTI (%)", 0.0, 60.0, float(v["dti"]), 0.1, key=f"l{k}")
            state = st.text_input("Estado (EE. UU.)", v["state"], max_chars=2, key=f"m{k}")
        with c3:
            fico = st.slider("FICO", 600, 850, v["fico"], key=f"n{k}")
            revol_bal = st.number_input("Saldo revolvente (USD)", 0, 1_000_000, v["revol_bal"], 500, key=f"o{k}")
            revol_util = st.slider("Utilización revolvente (%)", 0, 150, v["revol_util"], key=f"p{k}")
            open_acc = st.number_input("Líneas abiertas", 0, 100, v["open_acc"], key=f"q{k}")
            total_acc = st.number_input("Líneas totales", 0, 200, v["total_acc"], key=f"r{k}")
            history = st.number_input("Antigüedad crediticia (meses)", 0, 900, v["history"], key=f"s{k}")
        submitted = st.form_submit_button("Evaluar", type="primary")
    if not submitted:
        st.caption("Elige un perfil o ajusta los valores y presiona **Evaluar**.")
        return

    if not sub_grade.startswith(grade):
        st.warning(f"El subgrado {sub_grade} no corresponde al grado {grade}: se usa {grade}3.")
        sub_grade = f"{grade}3"
    form = dict(
        loan_amnt=loan_amnt, term=term, int_rate=int_rate, grade=grade, sub_grade=sub_grade, purpose=purpose,
        annual_inc=annual_inc, emp=emp, home=home, verif=verif, dti=dti, state=state, fico=fico,
        revol_bal=revol_bal, revol_util=revol_util, open_acc=open_acc, total_acc=total_acc, history=history,
    )  # fmt: skip
    result = D.score(D.build_record(**form), model=model())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Probabilidad de default", f"{result['probability']:.1%}")
    icon = "✅" if result["decision"] == "APROBAR" else "🛑"
    k2.metric("Decisión", f"{icon} {result['decision']}")
    k3.metric("Banda de riesgo", result["risk_band"])
    k4.metric("Cuota mensual", f"${result['installment']:,.0f}")
    st.caption(
        f"Se rechaza si la PD es mayor o igual al umbral **{result['threshold']:.2f}**, elegido minimizando el costo "
        "esperado con un falso negativo 5 veces más caro que un falso positivo."
    )
    st.subheader("Factores que más pesaron (SHAP)")
    st.caption("Contribución al log-odds de default: rojo aumenta el riesgo, verde lo reduce.")
    st.plotly_chart(factors_figure(result["factors"]), width="stretch")


def page_results() -> None:
    r = D.results()
    if not r:
        st.info("No se encontró `results.json`.")
        return
    f = r["final"]
    d, c = f["discrimination"], f["calibration"]
    th = min(f["thresholds"], key=lambda t: abs(t["threshold"] - f["threshold"]))
    tuned = " (ajustado con Optuna)" if f.get("origin") == "optuna" else ""
    st.markdown(f"**Modelo elegido:** `{f['algorithm']}`{tuned}. {r['selection']['reason']}.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC test", f"{d['auc']:.4f}", help=f"IC 95 % [{d['auc_ci'][0]:.4f}, {d['auc_ci'][1]:.4f}] (DeLong)")
    m2.metric("Gini / KS", f"{d['gini']:.3f} / {d['ks']:.3f}")
    m3.metric("Calibración (pendiente)", f"{c['slope']:.2f}", help=f"Intercepto {c['intercept']:.2f}")
    m4.metric("Morosidad de aprobados", f"{th['bad_rate_approved']:.1%}", f"{th['bad_rate_approved'] - d['prevalence']:+.1%}",
              delta_color="inverse", help=f"Umbral {f['threshold']:.2f}: aprueba {th['approval_rate']:.1%}. "
              f"Sin modelo la morosidad sería {d['prevalence']:.1%}.")  # fmt: skip
    st.caption(
        f"Test fuera de tiempo: {r['periods']['test']}. PD media {c['mean_pd']:.1%} vs tasa observada "
        f"{c['observed_rate']:.1%}."
    )

    st.subheader("Benchmark")
    st.caption("La selección se hace en validación; el test se usa una sola vez para confirmar.")
    st.dataframe(
        [
            {
                "Algoritmo": b["algorithm"],
                "AUC validación": round(b["val_auc"], 4),
                "AUC test": round(b["test_auc"], 4),
                "KS test": round(b["test_ks"], 3),
            }
            for b in sorted(r["benchmark"], key=lambda b: -b["val_auc"])
        ],  # fmt: skip
        hide_index=True,
    )
    ab = r["ablation"]
    st.markdown(
        "**Cómo se eligió:** prueba de DeLong para AUCs correlacionadas, corrección de Holm por comparaciones "
        "múltiples y un margen práctico de 0.005 de AUC; entre modelos empatados se prefiere el más simple.\n\n"
        f"**Ablación:** sin `grade`, `sub_grade` ni `int_rate` (el scoring propio de Lending Club) el AUC baja de "
        f"{ab['auc_full']:.4f} a {ab['auc_without']:.4f}: el modelo aporta información más allá del grado asignado."
    )
    fig_dir = D.figures_dir()
    if fig_dir is not None:
        st.subheader("Análisis estadístico")
        for name, caption in FIGURES:
            if (fig_dir / name).exists():
                st.markdown(f"**{caption}**")
                st.image(str(fig_dir / name), width="stretch")
    st.caption(f"Datos: {r.get('data_file', 'Lending Club')}. Generado el {r.get('generated_at', '')}.")


def page_monitoring() -> None:
    st.header("Monitoreo del replay de producción (originación 2014-2015)")
    snap = D.snapshot_info()
    if not snap:
        st.info("Esta publicación de la demo no incluye la foto de monitoreo.")
        return
    st.caption(
        f"Foto de las tablas Delta de producción ({snap.get('source', '')}) del {snap.get('exported_at', '?')}. "
        "En Databricks este mismo panel se lee en vivo con el SQL warehouse."
    )
    views.render_monitoring(
        D.snapshot("monitoring_metrics"),
        D.snapshot("drift_by_feature"),
        D.snapshot("ab_test_results"),
        D.snapshot("retrain_events"),
    )


def page_models() -> None:
    st.header("Benchmark de modelos (último entrenamiento)")
    info, snap = D.model_info(), D.snapshot_info()
    if snap:
        st.caption(f"Foto de Unity Catalog del {snap.get('exported_at', '?')}.")
    served = {"champion": info["version"]} if info.get("version") else None
    views.render_models(D.snapshot("model_benchmark"), D.snapshot("model_evaluation"), served)


def page_scoring_header() -> None:
    st.header("Evaluar una solicitud de crédito")
    info = D.model_info()
    version = f" v{info['version']}" if info.get("version") else ""
    st.caption(
        f"El score lo calcula el modelo champion `{info.get('algorithm', '?')}`{version} exportado de Unity Catalog "
        "(en Databricks lo calcula el endpoint de Model Serving)."
    )
    page_scoring()


def page_results_header() -> None:
    st.header("Por qué este modelo")
    page_results()


def page_architecture() -> None:
    st.markdown(ARCHITECTURE)


PAGES = {
    "Scoring": page_scoring_header,
    "Monitoreo": page_monitoring,
    "Modelos": page_models,
    "Por qué este modelo": page_results_header,
    "Arquitectura": page_architecture,
}


def main() -> None:
    st.set_page_config(page_title="Credit Risk Platform 2.0", page_icon="💳", layout="wide")
    choice = st.sidebar.radio("Navegación", list(PAGES))
    st.sidebar.caption("Lending Club 2007-2018 · Databricks Free Edition")
    st.sidebar.markdown(f"[Código en GitHub]({REPO})")
    info = D.model_info()
    if info:
        st.sidebar.caption(f"Modelo exportado el {info.get('exported_at', '?')}.")
    PAGES[choice]()


main()
