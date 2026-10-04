"""Demo pública de Credit Risk Platform 2.0 en Hugging Face Spaces (Gradio).

- Evaluar solicitud: PD, decisión y factores SHAP con el modelo champion exportado de Unity Catalog.
- Por qué este modelo: resultados del análisis estadístico (reports/results.json) y sus gráficas.
- Arquitectura: cómo funciona la plataforma completa en Databricks.
"""

from __future__ import annotations

import gradio as gr
import plotly.graph_objects as go

import demo_core as D

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


def _card(title: str, value: str, color: str | None = None) -> str:
    style = f"color:{color};" if color else ""
    return (
        '<div style="flex:1;min-width:140px;padding:14px 16px;border-radius:10px;'
        'border:1px solid var(--border-color-primary);background:var(--block-background-fill)">'
        f'<div style="font-size:13px;opacity:.75">{title}</div>'
        f'<div style="font-size:24px;font-weight:600;{style}">{value}</div></div>'
    )


def _factors_plot(factors: list[dict]) -> go.Figure:
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
    fig.update_layout(
        title="Factores que más pesaron (SHAP, log-odds): rojo aumenta el riesgo, verde lo reduce",
        height=330,
        margin=dict(l=10, r=40, t=50, b=30),
        xaxis_title="Contribución al log-odds de default",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    return fig


def evaluate(*values):
    form = dict(zip(D.FORM_FIELDS, values, strict=True))
    if not str(form["sub_grade"]).startswith(str(form["grade"])):
        form["sub_grade"] = f"{form['grade']}3"
    result = D.score(D.build_record(**form))
    approve = result["decision"] == "APROBAR"
    cards = "".join(
        [
            _card("Probabilidad de default", f"{result['probability']:.1%}"),
            _card("Decisión", result["decision"], SAFE if approve else RISK),
            _card("Banda de riesgo", result["risk_band"]),
            _card("Cuota mensual", f"${result['installment']:,.0f}"),
        ]
    )
    note = (
        f"<p style='opacity:.8;font-size:13px;margin-top:8px'>Se rechaza si la PD es mayor o igual al umbral "
        f"<b>{result['threshold']:.2f}</b>, elegido minimizando el costo esperado con un falso negativo 5 veces "
        "más caro que un falso positivo.</p>"
    )
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap">{cards}</div>{note}', _factors_plot(result["factors"])


def sync_subgrades(grade: str, current: str):
    return current if str(current).startswith(grade) else f"{grade}3"


def results_markdown() -> str:
    r = D.results()
    if not r:
        return "_No se encontró `results.json` en el Space._"
    f = r["final"]
    d, c = f["discrimination"], f["calibration"]
    th = min(f["thresholds"], key=lambda t: abs(t["threshold"] - f["threshold"]))
    bench = "\n".join(
        f"| `{b['algorithm']}` | {b['val_auc']:.4f} | {b['test_auc']:.4f} | {b['test_ks']:.3f} |"
        for b in sorted(r["benchmark"], key=lambda b: -b["val_auc"])
    )
    ab = r["ablation"]
    tuned = " (ajustado con Optuna)" if f.get("origin") == "optuna" else ""
    return f"""
### Decisión

**Modelo elegido:** `{f["algorithm"]}`{tuned}. {r["selection"]["reason"]}.

| Métrica (test fuera de tiempo, {r["periods"]["test"]}) | Valor |
|---|---|
| AUC | **{d["auc"]:.4f}** IC 95 % [{d["auc_ci"][0]:.4f}, {d["auc_ci"][1]:.4f}] |
| Gini / KS | {d["gini"]:.3f} / {d["ks"]:.3f} |
| Calibración (pendiente, intercepto) | {c["slope"]:.2f}, {c["intercept"]:.2f} |
| PD media vs tasa observada | {c["mean_pd"]:.1%} vs {c["observed_rate"]:.1%} |
| Umbral {f["threshold"]:.2f} | aprueba {th["approval_rate"]:.1%}, morosidad de aprobados {th["bad_rate_approved"]:.1%} (vs {d["prevalence"]:.1%} sin modelo) |

### Benchmark (selección en validación, test usado una sola vez)

| Algoritmo | AUC validación | AUC test | KS test |
|---|---|---|---|
{bench}

**Cómo se eligió:** prueba de DeLong para AUCs correlacionadas, corrección de Holm por comparaciones múltiples
y un margen práctico de 0.005 de AUC; entre modelos empatados se prefiere el más simple (parsimonia).

**Ablación:** sin `grade`, `sub_grade` ni `int_rate` (el scoring propio de Lending Club) el AUC baja de
{ab["auc_full"]:.4f} a {ab["auc_without"]:.4f}: el modelo aporta información más allá del grado asignado.

_Datos: {r.get("data_file", "Lending Club")}. Generado el {r.get("generated_at", "")}._
"""


def gallery_items() -> list[tuple[str, str]]:
    fig_dir = D.figures_dir()
    if fig_dir is None:
        return []
    return [(str(fig_dir / name), caption) for name, caption in FIGURES if (fig_dir / name).exists()]


def model_badge() -> str:
    info = D.model_info()
    if not info:
        return ""
    version = f" v{info['version']}" if info.get("version") else ""
    return (
        f"Modelo: **`{info.get('algorithm', '?')}`{version}** ({info.get('source', '')}), "
        f"exportado el {info.get('exported_at', '?')}."
    )


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
5. **CI/CD:** GitHub Actions con entornos dev, staging y prod (aprobación manual), 90+ tests y CodeQL.

Código, reporte técnico y resultados completos: [{REPO}]({REPO})

> Demo educativa con datos históricos públicos. No es una herramienta de decisión crediticia real.
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Credit Risk Platform 2.0") as demo:
        gr.Markdown(
            "# Credit Risk Platform 2.0\n"
            "Probabilidad de default de un préstamo de Lending Club con el modelo en producción de una plataforma "
            f"MLOps en Databricks. [Código en GitHub]({REPO})\n\n{model_badge()}"
        )
        with gr.Tab("Evaluar solicitud"):
            with gr.Row():
                with gr.Column():
                    loan_amnt = gr.Number(12000, label="Monto solicitado (USD)", minimum=500, maximum=40000)
                    term = gr.Radio([36, 60], value=36, label="Plazo (meses)")
                    int_rate = gr.Slider(5, 31, 13.5, step=0.1, label="Tasa de interés (%)")
                    grade = gr.Dropdown(D.GRADES, value="C", label="Grado Lending Club")
                    sub_grade = gr.Dropdown(D.SUB_GRADES, value="C2", label="Subgrado")
                    purpose = gr.Dropdown(D.PURPOSES, value="debt_consolidation", label="Propósito")
                with gr.Column():
                    annual_inc = gr.Number(65000, label="Ingreso anual (USD)", minimum=1000)
                    emp = gr.Slider(0, 10, 5, step=1, label="Años en el empleo")
                    home = gr.Dropdown(D.HOME, value="RENT", label="Vivienda")
                    verif = gr.Dropdown(D.VERIFICATION, value="Source Verified", label="Verificación de ingresos")
                    dti = gr.Slider(0, 60, 18.5, step=0.1, label="Deuda / ingreso, DTI (%)")
                    state = gr.Textbox("CA", label="Estado (EE. UU.)", max_length=2)
                with gr.Column():
                    fico = gr.Slider(600, 850, 702, step=1, label="FICO")
                    revol_bal = gr.Number(14500, label="Saldo revolvente (USD)", minimum=0)
                    revol_util = gr.Slider(0, 150, 55, step=1, label="Utilización revolvente (%)")
                    open_acc = gr.Number(10, label="Líneas abiertas", minimum=0, precision=0)
                    total_acc = gr.Number(24, label="Líneas totales", minimum=0, precision=0)
                    history = gr.Number(180, label="Antigüedad crediticia (meses)", minimum=0, precision=0)
            inputs = [loan_amnt, term, int_rate, grade, sub_grade, purpose, annual_inc, emp, home, verif, dti,
                      state, fico, revol_bal, revol_util, open_acc, total_acc, history]  # fmt: skip
            button = gr.Button("Evaluar", variant="primary")
            summary = gr.HTML()
            factors = gr.Plot(show_label=False)
            gr.Examples(list(D.EXAMPLES.values()), inputs=inputs, example_labels=list(D.EXAMPLES), label="Ejemplos")
            grade.change(sync_subgrades, [grade, sub_grade], sub_grade)
            button.click(evaluate, inputs, [summary, factors])
        with gr.Tab("Por qué este modelo"):
            gr.Markdown(results_markdown())
            for path, caption in gallery_items():
                gr.Markdown(f"#### {caption}")
                gr.Image(path, show_label=False, interactive=False, container=False)
        with gr.Tab("Arquitectura"):
            gr.Markdown(ARCHITECTURE)
    return demo


if __name__ == "__main__":
    build_demo().launch(theme=gr.themes.Soft())
