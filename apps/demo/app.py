"""Demo pública de Credit Risk Platform 2.0 (Streamlit Community Cloud).

Mismas páginas que el dashboard de Databricks (Scoring, Monitoreo, Modelos, con `views.py`
compartido) más el análisis estadístico y la arquitectura. Sin credenciales: el scoring usa el
champion exportado y Monitoreo/Modelos una foto de las tablas Delta tomada al publicar.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import demo_core as D
import ui
import views

REPO = "https://github.com/wilder14-eslu/credit-risk-platform-2.0"
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


ARCHITECTURE = [
    ("Datos", "2.26 millones de préstamos de Lending Club (2007-2018) en Bronze, Silver y Gold sobre Delta Lake, "
     "con validaciones de calidad y solo variables conocidas al solicitar (sin data leakage)."),
    ("Entrenamiento", "Regresión logística, XGBoost, LightGBM y CatBoost más un scorecard WoE. Selección en "
     "validación con DeLong + Holm + parsimonia, tuning con Optuna y quality gates. Partición temporal."),
    ("Registro y despliegue", "MLflow con Unity Catalog (champion y challenger), lineage código-datos-modelo, "
     "Model Serving con A/B testing, API FastAPI y dashboard Streamlit como Databricks Apps."),
    ("Política de crédito", "El modelo estima la PD y una política versionada decide: reglas duras, umbral por "
     "costo esperado y zona gris a revisión manual, con códigos de motivo auditables."),
    ("Producción simulada", "Replay mes a mes de la originación 2014-2015, monitoreo de data, prediction y "
     "concept drift, y reentrenamiento continuo con validación champion vs challenger y rollback."),
    ("CI/CD", "GitHub Actions con entornos dev, staging y prod (aprobación manual), 169 tests con cobertura "
     "100 % y CodeQL. Esta demo se publica con un workflow que exporta el champion y prueba una predicción."),
]  # fmt: skip

DISCLAIMER = (
    "Demo educativa con datos históricos públicos de Lending Club. No es una herramienta de decisión crediticia real."
)


@st.cache_resource
def model():
    return D.load_model()


def _champion_badge() -> str:
    info = D.model_info()
    version = f" v{info['version']}" if info.get("version") else ""
    return f"Champion {info.get('algorithm', '?')}{version}"


def page_scoring() -> None:
    from credit_risk import policy as P

    info = D.model_info()
    ui.hero(
        "Evaluar una solicitud de crédito",
        "El modelo champion estima la probabilidad de default y la política de crédito toma la decisión.",
        [_champion_badge(), f"Umbral {float(info.get('threshold', 0.18)):.2f}", f"Política {P.POLICY_VERSION}",
         "En Databricks lo sirve Model Serving"],
    )  # fmt: skip

    profile = st.radio("Perfil de ejemplo", list(D.EXAMPLES), index=1, horizontal=True)
    v = dict(zip(D.FORM_FIELDS, D.EXAMPLES[profile], strict=True))
    k = profile  # las claves cambian con el perfil para recargar los valores del formulario
    with st.form("solicitud", border=False):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            ui.section("Préstamo")
            loan_amnt = st.number_input("Monto solicitado (USD)", 500, 40000, v["loan_amnt"], 500, key=f"a{k}")
            term = st.selectbox("Plazo (meses)", [36, 60], [36, 60].index(v["term"]), key=f"b{k}")
            int_rate = st.slider("Tasa de interés (%)", 5.0, 31.0, float(v["int_rate"]), 0.1, key=f"c{k}")
            g1, g2 = st.columns(2)
            grade = g1.selectbox("Grado", D.GRADES, D.GRADES.index(v["grade"]), key=f"d{k}")
            sub_grade = g2.selectbox("Subgrado", D.SUB_GRADES, D.SUB_GRADES.index(v["sub_grade"]), key=f"e{k}")
            purpose = st.selectbox("Propósito", D.PURPOSES, D.PURPOSES.index(v["purpose"]), key=f"f{k}")
        with c2:
            ui.section("Solicitante y capacidad de pago")
            annual_inc = st.number_input("Ingreso anual (USD)", 1000, 2_000_000, v["annual_inc"], 1000, key=f"g{k}")
            dti = st.slider("Deuda / ingreso, DTI (%)", 0.0, 60.0, float(v["dti"]), 0.1, key=f"l{k}")
            emp = st.slider("Años en el empleo", 0, 10, v["emp"], key=f"h{k}")
            h1, h2 = st.columns(2)
            home = h1.selectbox("Vivienda", D.HOME, D.HOME.index(v["home"]), key=f"i{k}")
            state = h2.text_input("Estado (EE. UU.)", v["state"], max_chars=2, key=f"m{k}")
            verif = st.selectbox(
                "Verificación de ingresos", D.VERIFICATION, D.VERIFICATION.index(v["verif"]), key=f"j{k}"
            )
        with c3:
            ui.section("Historial crediticio")
            fico = st.slider("FICO", 600, 850, v["fico"], key=f"n{k}")
            revol_bal = st.number_input("Saldo revolvente (USD)", 0, 1_000_000, v["revol_bal"], 500, key=f"o{k}")
            revol_util = st.slider("Utilización revolvente (%)", 0, 150, v["revol_util"], key=f"p{k}")
            l1, l2 = st.columns(2)
            open_acc = l1.number_input("Líneas abiertas", 0, 100, v["open_acc"], key=f"q{k}")
            total_acc = l2.number_input("Líneas totales", 0, 200, v["total_acc"], key=f"r{k}")
            history = st.number_input("Antigüedad crediticia (meses)", 0, 900, v["history"], key=f"s{k}")
        submitted = st.form_submit_button("Evaluar solicitud", type="primary")
    if not submitted:
        st.caption("Elige un perfil o ajusta los valores y presiona **Evaluar solicitud**.")
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
    show_result(result, annual_inc, float(P.POLICY["review_band"]))


def show_result(result: dict, annual_inc: float, review_band: float) -> None:
    st.subheader("Resultado")
    decision = result["decision"]
    left, right = st.columns([5, 6], gap="large")
    with left:
        gauge = ui.pd_gauge(result["probability"], result["threshold"], review_band, decision)
        st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})
        st.caption(
            f"Verde: aprobación automática · ámbar: zona gris (revisión manual) · rojo: rechazo. "
            f"La línea marca el umbral {result['threshold']:.2f}."
        )
    with right:
        labels = {f["feature"]: f["label"] for f in result["factors"]}
        ui.decision_card(
            decision,
            result["reasons"],
            labels,
            f"Política {result['policy_version']}: rechaza si la PD es mayor o igual al umbral "
            f"{result['threshold']:.2f} (costo esperado con FN:FP = 5:1), envía la zona gris a un analista y "
            "aplica reglas duras de capacidad de pago.",
        )
        icon = {"APROBAR": "✅", "REVISION_MANUAL": "⚠️"}.get(decision, "🛑")
        payment_ratio = result["installment"] / (annual_inc / 12) if annual_inc else 0.0
        m1, m2 = st.columns(2)
        m1.metric("Probabilidad de default", f"{result['probability']:.1%}")
        m2.metric("Decisión", f"{icon} {decision.replace('_', ' ')}")
        m3, m4 = st.columns(2)
        m3.metric("Banda de riesgo", result["risk_band"], help="A (menor riesgo) a E (mayor riesgo)")
        m4.metric("Cuota mensual", f"${result['installment']:,.0f}", f"{payment_ratio:.0%} del ingreso",
                  delta_color="off", delta_arrow="off")  # fmt: skip
    st.subheader("Factores que más pesaron (SHAP)")
    st.caption("Contribución al log-odds de default de esta solicitud: rojo aumenta el riesgo, verde lo reduce.")
    st.plotly_chart(ui.factors_figure(result["factors"]), width="stretch", config={"displayModeBar": False})


def page_results() -> None:
    ui.hero(
        "Por qué este modelo",
        "Selección estadística en validación y confirmación única en el test fuera de tiempo.",
        ["DeLong + Holm", "Margen práctico 0.005 AUC", "Parsimonia", "Optuna"],
    )
    r = D.results()
    if not r:
        st.info("No se encontró `results.json`.")
        return
    f = r["final"]
    d, c = f["discrimination"], f["calibration"]
    th = min(f["thresholds"], key=lambda t: abs(t["threshold"] - f["threshold"]))
    tuned = " (ajustado con Optuna)" if f.get("origin") == "optuna" else ""
    st.markdown(
        f"**Modelo elegido en el reporte reproducible:** `{f['algorithm']}`{tuned}. {r['selection']['reason']}."
    )
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
        width="stretch",
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
        available = [(n, cap) for n, cap in FIGURES if (fig_dir / n).exists()]
        for i in range(0, len(available), 2):
            cols = st.columns(2, gap="large")
            for col, (name, caption) in zip(cols, available[i : i + 2], strict=False):
                with col.container(border=True):
                    st.markdown(f"**{caption}**")
                    st.image(str(fig_dir / name), width="stretch")
    ui.footer(f"Datos: {r.get('data_file', 'Lending Club')}. Generado el {r.get('generated_at', '')}.")


def _num(row: pd.Series, col: str) -> float | None:
    value = pd.to_numeric(row.get(col), errors="coerce")
    return None if pd.isna(value) else float(value)


def monitoring_kpis(mon: pd.DataFrame, ab: pd.DataFrame) -> None:
    last = mon.sort_values("clock_month").iloc[-1]
    auc, psi = _num(last, "roc_auc"), _num(last, "prediction_psi")
    obs, pred = _num(last, "default_rate_observed"), _num(last, "default_rate_predicted")
    drifted = _num(last, "features_drifted")
    month = str(last["clock_month"])[:7]
    st.markdown(
        f"**Último mes evaluado:** {month} &nbsp; {ui.status_pill(str(last['severity']))}", unsafe_allow_html=True
    )
    items = [
        ("AUC", f"{auc:.3f}" if auc is not None else "s/d", "Con desenlaces reales del mes"),
        ("PSI predicción", f"{psi:.3f}" if psi is not None else "s/d", "< 0.10 estable, > 0.25 alerta"),
        ("Default observado", f"{obs:.1%}" if obs is not None else "s/d", "Tasa real de default del mes",
         f"Predicho {pred:.1%}" if pred is not None else None),
        ("Variables con drift", f"{drifted:.0f}" if drifted is not None else "s/d", "PSI o KS sobre el umbral"),
    ]  # fmt: skip
    if not ab.empty:
        items.append(
            ("Último A/B", str(ab.iloc[0].get("decision", "s/d")).capitalize(), str(ab.iloc[0].get("reason", "")))
        )
    ui.kpis(items)


def page_monitoring() -> None:
    snap = D.snapshot_info()
    ui.hero(
        "Monitoreo de producción",
        "Replay mes a mes de la originación 2014-2015: data drift, prediction drift, concept drift y A/B testing.",
        [f"Foto del {snap.get('exported_at', '?')}" if snap else "Sin foto", _champion_badge(),
         "En Databricks se lee en vivo"],
    )  # fmt: skip
    if not snap:
        st.info("Esta publicación de la demo no incluye la foto de monitoreo.")
        return
    mon, ab = D.snapshot("monitoring_metrics"), D.snapshot("ab_test_results")
    if not mon.empty:
        monitoring_kpis(mon, ab)
    st.caption(
        f"Foto de las tablas Delta de producción ({snap.get('source', '')}). "
        "En Databricks este mismo panel se lee en vivo con el SQL warehouse."
    )
    drift = D.snapshot("drift_by_feature")
    for col, digits in (("psi", 3), ("ks_pvalue", 4)):
        if col in drift:
            drift[col] = pd.to_numeric(drift[col], errors="coerce").round(digits)
    views.render_monitoring(mon, drift, ab, D.snapshot("retrain_events"), plot=ui.plot)


def page_models() -> None:
    info, snap = D.model_info(), D.snapshot_info()
    ui.hero(
        "Modelos",
        "Benchmark del último entrenamiento en Databricks con intervalos de confianza de DeLong.",
        [_champion_badge()] + ([f"Foto del {snap.get('exported_at', '?')}"] if snap else []),
    )
    bench = D.snapshot("model_benchmark")
    auc = pd.to_numeric(pd.Series([info.get("test_roc_auc")]), errors="coerce").iloc[0]
    ui.kpis(
        [
            ("Modelo en producción", f"{info.get('algorithm', '?')}", "Alias champion en Unity Catalog"),
            ("Versión", f"v{info['version']}" if info.get("version") else "local"),
            ("AUC test (champion)", f"{auc:.4f}" if pd.notna(auc) else "s/d"),
            ("Candidatos evaluados", f"{len(bench)}" if not bench.empty else "s/d"),
        ]
    )
    table = bench.drop(columns=["run_ts"], errors="ignore")
    for col in table.columns:
        values = pd.to_numeric(table[col], errors="coerce")
        if values.notna().all() and col not in ("registered_version",):
            table[col] = values.round(4)
    # La versión servida ya está en las tarjetas: no se repite como JSON.
    views.render_models(table, D.snapshot("model_evaluation"), None, plot=ui.plot)


def page_architecture() -> None:
    ui.hero(
        "Arquitectura",
        "La plataforma completa corre en Databricks Free Edition y está definida como código con Asset Bundles.",
        ["Delta Lake", "MLflow + Unity Catalog", "Model Serving", "Databricks Apps", "GitHub Actions"],
    )
    for i in range(0, len(ARCHITECTURE), 3):
        cols = st.columns(3, gap="medium")
        for j, (col, (title, text)) in enumerate(zip(cols, ARCHITECTURE[i : i + 3], strict=False), start=i + 1):
            col.markdown(
                f'<div class="arch-card"><div class="num">{j:02d}</div><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True,
            )
        st.write("")
    st.markdown(f"Código, reporte técnico y resultados completos: [{REPO}]({REPO})")


PAGES = {
    "Scoring": page_scoring,
    "Monitoreo": page_monitoring,
    "Modelos": page_models,
    "Por qué este modelo": page_results,
    "Arquitectura": page_architecture,
}


def main() -> None:
    st.set_page_config(page_title="Credit Risk Platform 2.0", page_icon="💳", layout="wide")
    ui.inject_css()
    ui.sidebar_brand()
    choice = st.sidebar.radio("Navegación", list(PAGES), label_visibility="collapsed")
    info = D.model_info()
    lines = [("Modelo", _champion_badge().removeprefix("Champion "))]
    if info.get("threshold") is not None:
        lines.append(("Umbral", f"{float(info['threshold']):.2f}"))
    if info.get("exported_at"):
        lines.append(("Exportado", str(info["exported_at"])))
    lines.append(("Datos", "Lending Club 2007-2018"))
    ui.sidebar_card(lines)
    st.sidebar.markdown(f"<br>[Código en GitHub]({REPO})", unsafe_allow_html=True)
    PAGES[choice]()
    ui.footer(DISCLAIMER)


main()
