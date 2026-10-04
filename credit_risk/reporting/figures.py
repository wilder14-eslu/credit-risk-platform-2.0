"""Figuras estáticas (PNG) para el README y MLflow, generadas desde el dict de resultados.

Estilo: superficie clara, marcas finas (líneas de 2 px, marcadores >= 8 px), grilla
sólida y recesiva, texto en tintas neutras (nunca en el color de la serie), un hue
de énfasis para el modelo final y los tres primeros slots de la paleta categórica
(azul, naranja, aqua; validados para distinguirse con daltonismo) cuando hay varias
series, siempre con leyenda.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # azul, naranja, aqua
WASH = 0.12

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "font.family": "DejaVu Sans",
    }
)


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def auc_forest(results: dict[str, Any], path: Path) -> str:
    rows = list(reversed(results["comparison"]["auc"]))
    final = results["final"]["algorithm"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, r in enumerate(rows):
        color = SERIES[0] if r["model"] == final else MUTED
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=color, lw=2)
        ax.plot(r["auc"], i, "o", ms=8, color=color, mec=SURFACE, mew=2)
        ax.text(r["ci_high"] + 0.0006, i, f"{r['auc']:.4f}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(rows)), [r["model"] + (" (elegido)" if r["model"] == final else "") for r in rows])
    ax.set_xlabel("AUC en test fuera de tiempo, parámetros por defecto (IC 95 % DeLong)")
    ax.set_title("AUC por modelo con intervalo de confianza")
    ax.grid(axis="y", visible=False)
    lo = min(r["ci_low"] for r in rows)
    hi = max(r["ci_high"] for r in rows)
    ax.set_xlim(lo - 0.002, hi + 0.006)
    return _save(fig, path)


def pairwise_vs_final(results: dict[str, Any], path: Path) -> str:
    final = results["final"]["algorithm"]
    margin = results["selection_config"]["practical_margin"]
    rows = []
    for r in results["comparison"]["pairwise"]:
        if final not in (r["model_a"], r["model_b"]):
            continue
        sign = 1 if r["model_a"] == final else -1
        other = r["model_b"] if sign == 1 else r["model_a"]
        lo, hi = sorted((sign * r["ci_low"], sign * r["ci_high"]))
        rows.append((other, sign * r["delta"], lo, hi, r["p_holm"]))
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axvspan(-margin, margin, color=MUTED, alpha=WASH, lw=0)
    ax.axvline(0, color=MUTED, lw=1)
    for i, (_name, d, lo, hi, p) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=SERIES[0], lw=2)
        ax.plot(d, i, "o", ms=8, color=SERIES[0], mec=SURFACE, mew=2)
        ax.text(
            hi + 0.0004,
            i,
            f"Δ {d:+.4f} · p Holm {p:.1e}" if p < 1e-3 else f"Δ {d:+.4f} · p Holm {p:.3f}",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel(f"AUC({final}) − AUC(otro), IC 95 %   ·   banda gris = equivalencia práctica (±{margin})")
    ax.set_title(f"¿Cuánto mejor es {final}? Diferencias pareadas en test")
    ax.grid(axis="y", visible=False)
    right = max(r[3] for r in rows)
    ax.set_xlim(min(-margin - 0.002, min(r[2] for r in rows) - 0.002), right + 0.012)
    return _save(fig, path)


def roc_pr(results: dict[str, Any], path: Path) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    a.plot([0, 1], [0, 1], color=MUTED, lw=1)
    prevalence = results["final"]["discrimination"]["prevalence"]
    b.axhline(prevalence, color=MUTED, lw=1)
    for color, (name, c) in zip(SERIES, results["curves"].items(), strict=False):
        a.plot(c["fpr"], c["tpr"], color=color, label=name)
        b.plot(c["recall"], c["precision"], color=color, label=name)
    a.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="Curva ROC (test)")
    b.set(xlabel="Recall (defaults detectados)", ylabel="Precisión", title="Curva Precision-Recall (test)")
    b.text(0.98, prevalence + 0.01, f"línea base = prevalencia {_pct(prevalence)}", ha="right", fontsize=8, color=INK_2)
    a.legend(loc="lower right")
    return _save(fig, path)


def calibration_plot(results: dict[str, Any], path: Path) -> str:
    cal = results["final"]["calibration"]
    t = cal["table"]
    x = np.array([r["mean_pd"] for r in t])
    y = np.array([r["observed"] for r in t])
    lo = y - np.array([r["ci"][0] for r in t])
    hi = np.array([r["ci"][1] for r in t]) - y
    top = max(x.max(), y.max()) * 1.1
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, top], [0, top], color=MUTED, lw=1)
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o", ms=8, color=SERIES[0], mec=SURFACE, mew=2, elinewidth=2, capsize=0)
    ax.plot(x, y, color=SERIES[0], lw=2, alpha=0.6)
    ax.set(
        xlim=(0, top), ylim=(0, top), xlabel="PD media predicha (decil)", ylabel="Tasa de default observada (IC Wilson)"
    )
    ax.set_title("Diagrama de fiabilidad (test)")
    ax.text(
        0.03 * top,
        0.93 * top,
        f"pendiente {cal['slope']:.2f} · intercepto {cal['intercept']:+.2f}\n"
        f"PD media {_pct(cal['mean_pd'])} vs observada {_pct(cal['observed_rate'])}",
        fontsize=9,
        color=INK_2,
        va="top",
    )
    return _save(fig, path)


def deciles_plot(results: dict[str, Any], path: Path) -> str:
    d = results["final"]["deciles"]
    base = results["final"]["discrimination"]["prevalence"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    xs = [r["decile"] for r in d]
    rates = [r["default_rate"] for r in d]
    ax.bar(xs, rates, width=0.6, color=SERIES[0])
    ax.axhline(base, color=MUTED, lw=1)
    ax.text(10.4, base, f"promedio {_pct(base)}", va="bottom", ha="right", fontsize=8, color=INK_2)
    for xv, r in zip(xs, rates, strict=True):
        if xv in (1, 10):
            ax.text(xv, r + 0.006, _pct(r), ha="center", fontsize=9, color=INK)
    ax.set(xticks=xs, xlabel="Decil de riesgo (1 = 10 % más riesgoso)", ylabel="Tasa de default observada")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Ordenamiento de cartera: default por decil (test)")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def threshold_plot(results: dict[str, Any], path: Path) -> str:
    rows = results["final"]["thresholds"]
    chosen = results["final"]["threshold"]
    t = np.array([r["threshold"] for r in rows])
    cost = np.array([r["expected_cost"] for r in rows])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.plot(t, cost, color=SERIES[0])
    i = int(np.argmin(np.abs(t - chosen)))
    a.plot(t[i], cost[i], "o", ms=8, color=SERIES[0], mec=SURFACE, mew=2)
    a.annotate(
        f"umbral {chosen:.2f}", (t[i], cost[i]), xytext=(10, 14), textcoords="offset points", fontsize=9, color=INK
    )
    c = results["costs"]
    a.set(
        xlabel="Umbral de rechazo (PD)",
        ylabel="Costo esperado por solicitud",
        title=f"Costo esperado (FN:FP = {c['fn']:g}:{c['fp']:g})",
    )
    b.plot(t, [r["approval_rate"] for r in rows], color=SERIES[0], label="Tasa de aprobación")
    b.plot(t, [r["bad_rate_approved"] for r in rows], color=SERIES[1], label="Default entre aprobados")
    b.axvline(chosen, color=MUTED, lw=1)
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    b.set(xlabel="Umbral de rechazo (PD)", title="Aprobación vs morosidad de la cartera")
    b.legend(loc="center right")
    return _save(fig, path)


def _phase_of(quarter: str, periods: dict[str, str]) -> str:
    """Fase de un trimestre ('2013Q3') según los periodos del split ('2013-01 a 2013-06 (...)')."""
    import pandas as pd

    q = pd.Period(quarter, freq="Q")
    for key, label in (("val", "validación"), ("test", "test")):
        text = periods.get(key, "")
        try:
            start, end = (pd.Period(part.strip()[:7], freq="M") for part in text.split("(")[0].split(" a "))
        except ValueError:
            continue
        if start.asfreq("Q") <= q <= end.asfreq("Q"):
            return label
    return "producción"


def vintage_plot(results: dict[str, Any], path: Path) -> str:
    v = results["vintages"]
    q = [r["quarter"] for r in v]
    x = np.arange(len(q))
    phases = [_phase_of(quarter, results["periods"]) for quarter in q]
    auc = np.array([r["auc"] for r in v])
    lo = np.array([r["auc_ci"][0] for r in v])
    hi = np.array([r["auc_ci"][1] for r in v])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9))
    shades = {"validación": (MUTED, WASH), "test": (SERIES[0], 0.07)}
    for ax in (a, b):
        for i, ph in enumerate(phases):
            if ph in shades:
                color, alpha = shades[ph]
                ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=alpha, lw=0)
        ax.set_xticks(x, q, rotation=45, ha="right")
        ax.grid(axis="x", visible=False)
    a.fill_between(x, lo, hi, color=SERIES[0], alpha=WASH, lw=0)
    a.plot(x, auc, color=SERIES[0], marker="o", ms=6, mec=SURFACE, mew=1.5)
    a.set(ylabel="AUC (IC 95 % DeLong)", title="Estabilidad del AUC por cosecha")
    ymin = a.get_ylim()[0]
    for label in ("validación", "test", "producción"):
        idx = [i for i, ph in enumerate(phases) if ph == label]
        if idx:
            text = "producción (nunca vista)" if label == "producción" else label
            a.text(float(np.mean(idx)), ymin + 0.001, text, ha="center", fontsize=8, color=INK_2)
    b.plot(
        x,
        [r["observed"] for r in v],
        color=SERIES[0],
        marker="o",
        ms=6,
        mec=SURFACE,
        mew=1.5,
        label="Default observado",
    )
    b.plot(
        x, [r["mean_pd"] for r in v], color=SERIES[1], marker="o", ms=6, mec=SURFACE, mew=1.5, label="PD media predicha"
    )
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    b.set(title="Calibración en el tiempo")
    b.legend(loc="lower left")
    return _save(fig, path)


def shap_bar(results: dict[str, Any], path: Path) -> str:
    rows = list(reversed(results["shap"]))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh([r["feature"] for r in rows], [r["mean_abs_shap"] for r in rows], height=0.6, color=SERIES[0])
    ax.set(
        xlabel="|SHAP| medio (contribución al log-odds)",
        title=f"Importancia global SHAP ({results['final']['algorithm']})",
    )
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def shap_dependence(results: dict[str, Any], path: Path) -> str:
    dep = results["shap_dependence"]
    fig, axes = plt.subplots(1, len(dep), figsize=(4 * len(dep), 3.6), squeeze=False)
    for ax, (name, d) in zip(axes[0], dep.items(), strict=True):
        xv, yv = np.array(d["values"]), np.array(d["shap"])
        hi = np.quantile(xv, 0.99)
        keep = xv <= hi
        ax.scatter(xv[keep], yv[keep], s=10, color=SERIES[0], alpha=0.35, lw=0)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set(xlabel=name, title=name)
    axes[0][0].set_ylabel("Contribución SHAP (log-odds)")
    fig.suptitle("Dependencia SHAP: cómo cambia el riesgo con cada variable", x=0.01, ha="left", fontsize=12, color=INK)
    return _save(fig, path)


def iv_bar(results: dict[str, Any], path: Path) -> str:
    rows = list(reversed(results["scorecard"]["iv"][:15]))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh([r["feature"] for r in rows], [r["iv"] for r in rows], height=0.6, color=SERIES[0])
    for ref, label in ((0.1, "medio"), (0.3, "fuerte")):
        ax.axvline(ref, color=MUTED, lw=1)
        ax.text(ref, len(rows) - 0.4, label, fontsize=8, color=INK_2, ha="left")
    ax.set(xlabel="Information Value (scorecard WoE, train)", title="Poder predictivo por variable (IV)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


FIGURES = {
    "auc_ci": auc_forest,
    "pairwise": pairwise_vs_final,
    "roc_pr": roc_pr,
    "calibration": calibration_plot,
    "deciles": deciles_plot,
    "threshold": threshold_plot,
    "vintages": vintage_plot,
    "shap": shap_bar,
    "shap_dependence": shap_dependence,
    "iv": iv_bar,
}


def render_all(results: dict[str, Any], out_dir: Path) -> dict[str, str]:
    """Genera todas las figuras; devuelve {clave: nombre de archivo}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, fn in FIGURES.items():
        if key in ("shap_dependence", "vintages") and not results.get(key):
            continue
        files[key] = fn(results, out_dir / f"{key}.png")
    return files
