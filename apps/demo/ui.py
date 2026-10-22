"""Identidad visual de la demo pública: estilos, encabezado, tarjetas y gráficos.

Solo presentación (sin lógica de negocio). Las páginas compartidas con el dashboard de
Databricks (`views.py`) no se tocan: aquí se agregan el encabezado, las tarjetas KPI,
el medidor de PD y los estilos que envuelven esas páginas en la demo.
"""

from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

# Paleta sobria de banca: un azul institucional como acento y colores de estado reservados.
NAVY, BLUE, INK, MUTED, LINE, SURFACE = "#0b2545", "#1f5fbf", "#1c2430", "#5b6675", "#dfe4ec", "#f5f7fb"
GOOD, WARN, BAD = "#15803d", "#b7791f", "#c0392b"
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

DECISION_STYLE = {
    "APROBAR": (GOOD, "#e8f5ec", "Aprobada"),
    "REVISION_MANUAL": (WARN, "#fdf4e3", "Revisión manual"),
    "RECHAZAR": (BAD, "#fbeaea", "Rechazada"),
}
SEVERITY_STYLE = {"estable": GOOD, "warning": WARN, "alerta": BAD, "critico": BAD}

REASON_TEXT = {
    "PD_SOBRE_UMBRAL": "PD mayor o igual al umbral",
    "PD_ZONA_GRIS": "PD en la zona gris, justo bajo el umbral",
    "CUOTA_INGRESO_ALTA": "Cuota mayor al 40 % del ingreso mensual",
    "DTI_ALTO": "Deuda / ingreso (DTI) mayor a 45 %",
    "MODELO_NO_DISPONIBLE": "Modelo no disponible",
    "SALIDA_MODELO_INCOHERENTE": "Salida del modelo incoherente",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, div, p, label, li, h1, h2, h3, h4, button, input, textarea {{ font-family: {FONT}; }}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
.block-container {{ padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1280px; }}
h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.45rem !important; }}
h3 {{ font-size: 1.12rem !important; }}

/* Barra lateral */
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] b {{ color: #e8eef7; }}
[data-testid="stSidebarCollapseButton"] * {{ color: #e8eef7 !important; }}
[data-testid="stSidebar"] a {{ color: #9cc3ff !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: .45rem .6rem; border-radius: 8px; margin-bottom: 2px; width: 100%;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,.08); }}
.side-brand {{ font-weight: 700; font-size: 1.05rem; line-height: 1.25; margin-bottom: .2rem; }}
.side-sub {{ font-size: .78rem; opacity: .75; margin-bottom: 1rem; }}
.side-card {{ border: 1px solid rgba(255,255,255,.18); border-radius: 10px; padding: .7rem .8rem;
  font-size: .8rem; line-height: 1.55; margin-top: .6rem; }}
.side-card b {{ font-weight: 600; }}

/* Encabezado */
.hero {{ background: linear-gradient(120deg, {NAVY} 0%, #13315c 55%, {BLUE} 100%); color: #fff;
  border-radius: 14px; padding: 1.35rem 1.6rem; margin-bottom: 1.2rem; }}
.hero h1 {{ color: #fff; font-size: 1.6rem; margin: 0 0 .25rem 0; padding: 0; }}
.hero p {{ color: #d6e2f3; margin: 0; font-size: .93rem; }}
.badges {{ margin-top: .8rem; display: flex; flex-wrap: wrap; gap: .4rem; }}
.badge {{ background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); color: #fff;
  border-radius: 999px; padding: .18rem .65rem; font-size: .76rem; white-space: nowrap; }}

/* Tarjetas KPI (st.metric) */
[data-testid="stMetric"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px;
  padding: .85rem 1rem; box-shadow: 0 1px 2px rgba(11,37,69,.05); }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .8rem; font-weight: 500; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-size: 1.4rem; font-weight: 650; }}

/* Formulario y contenedores */
[data-testid="stForm"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 1.2rem 1.3rem; }}
.section-title {{ font-size: .78rem; font-weight: 650; color: {BLUE}; text-transform: uppercase;
  letter-spacing: .06em; border-bottom: 1px solid {LINE}; padding-bottom: .35rem; margin-bottom: .6rem; }}
.stButton button[kind="primaryFormSubmit"], .stFormSubmitButton button {{ border-radius: 8px; padding: .5rem 1.6rem;
  font-weight: 600; }}

/* Decisión y motivos */
.decision {{ border-radius: 12px; padding: 1rem 1.2rem; border: 1px solid; margin-bottom: .8rem; }}
.decision .label {{ font-size: .76rem; text-transform: uppercase; letter-spacing: .06em; opacity: .85; }}
.decision .value {{ font-size: 1.55rem; font-weight: 700; margin: .1rem 0 .3rem 0; }}
.decision .note {{ font-size: .85rem; color: {INK}; }}
.chips {{ display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .5rem; }}
.chip {{ background: #fff; border: 1px solid {LINE}; border-radius: 999px; padding: .15rem .6rem;
  font-size: .78rem; color: {INK}; }}
.chip code {{ background: none; color: {MUTED}; font-size: .72rem; padding: 0; }}
.status-pill {{ display: inline-block; border-radius: 999px; padding: .12rem .6rem; color: #fff;
  font-size: .78rem; font-weight: 600; }}
.foot {{ color: {MUTED}; font-size: .78rem; border-top: 1px solid {LINE}; padding-top: .8rem; margin-top: 2rem; }}
.arch-card {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 1rem 1.1rem; min-height: 215px; }}
.arch-card .num {{ color: {BLUE}; font-weight: 700; font-size: .8rem; letter-spacing: .06em; }}
.arch-card h4 {{ color: {NAVY}; margin: .2rem 0 .4rem 0; font-size: 1rem; }}
.arch-card p {{ color: {INK}; font-size: .86rem; line-height: 1.5; margin: 0; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    chips = "".join(f'<span class="badge">{escape(b)}</span>' for b in badges or [])
    st.markdown(
        f'<div class="hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p><div class="badges">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="side-brand">Credit Risk Platform 2.0</div>'
        '<div class="side-sub">Riesgo crediticio end-to-end · MLOps en Databricks</div>',
        unsafe_allow_html=True,
    )


def sidebar_card(lines: list[tuple[str, str]]) -> None:
    body = "<br>".join(f"<b>{escape(k)}:</b> {escape(v)}" for k, v in lines)
    st.sidebar.markdown(f'<div class="side-card">{body}</div>', unsafe_allow_html=True)


def kpis(items: list[tuple]) -> None:
    """Fila de tarjetas: (etiqueta, valor[, ayuda[, nota]])."""
    for col, item in zip(st.columns(len(items)), items, strict=True):
        label, value, *rest = item
        help_text = rest[0] if rest else None
        note = rest[1] if len(rest) > 1 else None
        col.metric(label, value, note, delta_color="off", delta_arrow="off", help=help_text)


def reason_text(code: str, labels: dict[str, str]) -> str:
    if code.startswith("RIESGO_"):
        feature = code.removeprefix("RIESGO_").lower()
        return f"Aumenta el riesgo: {labels.get(feature, feature)}"
    return REASON_TEXT.get(code, code)


def decision_card(decision: str, reasons: list[str], labels: dict[str, str], note: str) -> None:
    color, background, text = DECISION_STYLE.get(decision, (BLUE, SURFACE, decision))
    chips = "".join(
        f'<span class="chip">{escape(reason_text(r, labels))} <code>{escape(r)}</code></span>' for r in reasons
    )
    st.markdown(
        f'<div class="decision" style="border-color:{color}; background:{background};">'
        f'<div class="label" style="color:{color};">Decisión de la política</div>'
        f'<div class="value" style="color:{color};">{escape(text)}</div>'
        f'<div class="note">{escape(note)}</div>'
        f'<div class="chips">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def status_pill(severity: str) -> str:
    color = SEVERITY_STYLE.get(severity, MUTED)
    return f'<span class="status-pill" style="background:{color};">{escape(severity.upper())}</span>'


def footer(text: str) -> None:
    st.markdown(f'<div class="foot">{escape(text)}</div>', unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Mismo estilo para todos los gráficos propios de la demo."""
    fig.update_layout(
        font=dict(family=FONT, color=INK, size=13),
        title_font=dict(color=NAVY, size=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_family=FONT),
        margin=dict(l=10, r=30, t=30, b=30),
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    if height:
        fig.update_layout(height=height)
    return fig


def pd_gauge(probability: float, threshold: float, review_band: float, decision: str) -> go.Figure:
    """Medidor de PD con zonas de la política: aprobación, zona gris y rechazo."""
    top = max(0.5, min(1.0, round(probability * 1.25 + 0.05, 1)))
    grey = max(0.0, threshold - review_band)
    color = DECISION_STYLE.get(decision, (BLUE,))[0]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number=dict(suffix=" %", valueformat=".1f", font=dict(size=40, color=NAVY, family=FONT)),
            title=dict(text="Probabilidad de default (PD)", font=dict(size=14, color=MUTED, family=FONT)),
            gauge=dict(
                axis=dict(range=[0, top * 100], ticksuffix=" %", tickcolor=MUTED, tickfont=dict(color=MUTED)),
                bar=dict(color=color, thickness=0.28),
                bgcolor="#fff",
                borderwidth=0,
                steps=[
                    dict(range=[0, grey * 100], color="#e8f5ec"),
                    dict(range=[grey * 100, threshold * 100], color="#fdf4e3"),
                    dict(range=[threshold * 100, top * 100], color="#fbeaea"),
                ],
                threshold=dict(line=dict(color=NAVY, width=3), thickness=0.85, value=threshold * 100),
            ),
        )
    )
    fig = style_figure(fig, height=280)
    fig.update_layout(margin=dict(l=40, r=50, t=50, b=10))
    return fig


def factors_figure(factors: list[dict]) -> go.Figure:
    factors = sorted(factors, key=lambda f: abs(f["impact"]))
    fig = go.Figure(
        go.Bar(
            x=[f["impact"] for f in factors],
            y=[f["label"] for f in factors],
            orientation="h",
            marker_color=[BAD if f["impact"] > 0 else GOOD for f in factors],
            text=[f"{f['impact']:+.3f}" for f in factors],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=MUTED)
    fig.update_layout(xaxis_title="Contribución al log-odds de default")
    return style_figure(fig, height=300)


def plot(fig: go.Figure) -> None:
    """Dibuja un gráfico de las páginas compartidas (`views.py`) con el estilo de la demo."""
    style_figure(fig)
    if fig.layout.xaxis.tickformat == "%Y-%m":  # series mensuales: una marca por trimestre
        fig.update_xaxes(dtick="M3", tickangle=0)
    anns = fig.layout.annotations or ()
    if len(anns) == 2:  # líneas de referencia cercanas: etiquetas a cada lado para que no se encimen
        fig.layout.annotations[0].xanchor, fig.layout.annotations[1].xanchor = "right", "left"
        fig.update_annotations(yshift=14, font=dict(size=11, color=MUTED))
    if fig.layout.yaxis.tickformat == ".0%":  # tasas cercanas: un decimal para que no se repitan marcas
        fig.update_yaxes(tickformat=".1%")
    if fig.layout.title.text:
        fig.update_layout(margin=dict(t=50))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
