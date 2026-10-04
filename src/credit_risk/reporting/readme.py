"""Sección de resultados del README, generada desde el dict de resultados.

Todo número y toda conclusión salen de `results`: el texto se arma con reglas
explícitas (significancia y magnitud), así que al reentrenar el README se regenera
sin editar a mano y nunca contradice a los datos.
"""

from __future__ import annotations

from typing import Any

START = "<!-- RESULTADOS:INICIO (generado por scripts/build_report.py; no editar a mano) -->"
END = "<!-- RESULTADOS:FIN -->"


def _p(x: float) -> str:
    return f"{100 * x:.1f}%"


def _period(text: str) -> str:
    """'2013-01 a 2013-06 (53374 préstamos)' -> '2013-01 a 2013-06'."""
    return text.split("(")[0].strip()


def _pv(p: float | None) -> str:
    if p is None or p != p:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _ci(ci: list[float], digits: int = 4) -> str:
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _pair(results: dict[str, Any], a: str, b: str) -> dict[str, Any] | None:
    """Fila pareada orientada como AUC(a) - AUC(b)."""
    for r in results["comparison"]["pairwise"]:
        if (r["model_a"], r["model_b"]) == (a, b):
            return r
        if (r["model_a"], r["model_b"]) == (b, a):
            return {
                **r,
                "model_a": a,
                "model_b": b,
                "delta": -r["delta"],
                "ci_low": -r["ci_high"],
                "ci_high": -r["ci_low"],
            }
    return None


def _img(fig_dir: str, files: dict[str, str], key: str, alt: str) -> str:
    return f"![{alt}]({fig_dir}/{files[key]})" if key in files else ""


def _reading(delta: float, p_holm: float, margin: float, alpha: float = 0.05) -> str:
    """Lectura en lenguaje natural de una diferencia (significancia estadística y práctica)."""
    if p_holm < alpha and abs(delta) >= margin:
        return "diferencia significativa y material"
    if p_holm < alpha:
        return "significativa pero prácticamente irrelevante"
    if abs(delta) >= margin:
        return "de tamaño relevante pero no concluyente"
    return "sin evidencia de diferencia"


def _why(results: dict[str, Any]) -> str:
    sel = results["selection"]
    best = results["final"]["algorithm"]
    tied = [r["model"] for r in sel["rows"] if r["retained"] and r["model"] != best]
    worse = [r["model"] for r in sel["rows"] if r["eligible"] and not r["retained"]]
    parts = [sel["reason"]]
    if tied:
        parts.append(f"empata en la práctica con {', '.join(tied)} (se prefiere por AUC a igual complejidad)")
    if worse:
        parts.append(f"{', '.join(worse)} es peor de forma significativa y material")
    return "; ".join(parts) + " (regla de parsimonia, decidida en validación)"


def _decision_summary(results: dict[str, Any]) -> list[str]:
    f = results["final"]
    d = f["discrimination"]
    thr = min(f["thresholds"], key=lambda r: abs(r["threshold"] - f["threshold"]))
    origin = {"optuna": "ajustado con Optuna", "default": "parámetros por defecto"}.get(f["origin"], f["origin"])
    return [
        "| | |",
        "|---|---|",
        f"| **Modelo elegido** | `{f['algorithm']}` ({origin}) |",
        f"| **Por qué** | {_why(results)} |",
        f"| **AUC en test fuera de tiempo** | **{d['auc']:.4f}** IC 95 % {_ci(d['auc_ci'])} · Gini {d['gini']:.3f} |",
        f"| **KS / PR-AUC** | {d['ks']:.3f} {_ci(d['ks_ci'], 3)} · {d['pr_auc']:.3f} (prevalencia {_p(d['prevalence'])}) |",
        f"| **Decisión** | Umbral {f['threshold']:.2f}: aprueba {_p(thr['approval_rate'])}, morosidad de aprobados "
        f"{_p(thr['bad_rate_approved'])} (vs {_p(d['prevalence'])} sin modelo), detecta {_p(thr['recall'])} de los defaults |",
        f"| **Quality gates** | {'aprobados' if f['gates']['passed'] else 'fallidos: ' + '; '.join(f['gates']['reasons'])} |",
    ]


def _conclusions(results: dict[str, Any]) -> list[str]:
    f = results["final"]
    best = f["algorithm"]
    margin = results["selection_config"]["practical_margin"]
    out = []
    retained = [r["model"] for r in results["selection"]["rows"] if r["retained"] and r["model"] != best]
    rejected = [r["model"] for r in results["selection"]["rows"] if r["eligible"] and not r["retained"]]
    if retained:
        rows = [r for r in results["selection"]["rows"] if r["model"] in retained]
        detectable = [r["model"] for r in rows if r["p_holm"] is not None and r["p_holm"] < 0.05]
        note = (
            f" (la de `{'`, `'.join(detectable)}` es estadísticamente detectable con este tamaño de muestra, pero no relevante)"
            if detectable
            else " y ninguna es significativa tras Holm"
        )
        out.append(
            f"**Empate práctico entre los más complejos.** `{best}` no supera de forma material a "
            f"{', '.join(f'`{m}`' for m in retained)}: todas las diferencias son menores al margen práctico de "
            f"{margin} de AUC{note}. La elección entre ellos es indiferente para el negocio y se resuelve por "
            "complejidad y luego por AUC."
        )
    for m in rejected:
        row = next(r for r in results["selection"]["rows"] if r["model"] == m)
        out.append(
            f"**`{m}` queda descartado en validación**: es peor por {row['delta_vs_best']:.4f} de AUC "
            f"(p Holm {_pv(row['p_holm'])}), una diferencia significativa y mayor al margen práctico."
        )
    for other in ("logistic_regression", "scorecard_woe"):
        r = _pair(results, best, other)
        if other == best or r is None:
            continue
        out.append(
            f"**Frente a `{other}`** el modelo gana {r['delta']:+.4f} de AUC en test, IC 95 % "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], p Holm {_pv(r['p_holm'])}: "
            f"{_reading(r['delta'], r['p_holm'], margin)}."
        )
    if "tuning" in results:
        tv, tt = results["tuning"]["validation"], results["tuning"]["test"]
        out.append(
            f"**El tuning con Optuna aporta poco.** En validación (donde se decide) mejora {tv['delta']:+.4f} de AUC "
            f"(p {_pv(tv['p_value'])}); en test {tt['delta']:+.4f} (p {_pv(tt['p_value'])}): "
            f"{_reading(tt['delta'], tt['p_value'], margin)}. El techo lo pone la información disponible, no los "
            "hiperparámetros."
        )
    ab = results["ablation"]
    kept = (ab["auc_without"] - 0.5) / (ab["auc_full"] - 0.5) if ab["auc_full"] > 0.5 else float("nan")
    out.append(
        f"**Aporte propio vs el scoring de Lending Club.** Sin `grade`, `sub_grade` ni `int_rate` el AUC baja de "
        f"{ab['auc_full']:.4f} a {ab['auc_without']:.4f} (pérdida {ab['delta']:.4f}, IC {_ci(ab['delta_ci'])}, p "
        f"{_pv(ab['p_value'])}): {_reading(ab['delta'], ab['p_value'], margin)}. Aun así, el modelo conserva "
        f"{_p(kept)} de su poder discriminante (Gini) solo con variables del solicitante."
    )
    cal = f["calibration"]
    if cal["spiegelhalter_p"] < 0.05 or abs(cal["intercept"]) > 0.05:
        direction = "sobreestima" if cal["mean_pd"] > cal["observed_rate"] else "subestima"
        out.append(
            f"**Calibración: ordena bien, pero {direction} el nivel.** Pendiente {cal['slope']:.2f} (ideal 1): la "
            f"escala relativa es correcta. Intercepto {cal['intercept']:+.2f} y Spiegelhalter p {_pv(cal['spiegelhalter_p'])}: "
            f"la PD media ({_p(cal['mean_pd'])}) difiere de la tasa observada ({_p(cal['observed_rate'])}). Para usar la PD "
            "como probabilidad (pricing, pérdida esperada) conviene recalibrar el intercepto con la validación."
        )
    else:
        out.append(
            f"**Calibración adecuada**: pendiente {cal['slope']:.2f}, intercepto {cal['intercept']:+.2f}, "
            f"Spiegelhalter p {_pv(cal['spiegelhalter_p'])}."
        )
    if results["vintages"]:
        aucs = [v["auc"] for v in results["vintages"]]
        last = results["vintages"][-1]
        out.append(
            f"**Estabilidad temporal.** El AUC por trimestre se mueve entre {min(aucs):.3f} y {max(aucs):.3f} en "
            f"{len(aucs)} cosechas, incluida la producción nunca vista. En {last['quarter']} la PD media es "
            f"{_p(last['mean_pd'])} vs {_p(last['observed'])} observado: la deriva de nivel es la señal que el monitoreo "
            "usa para disparar el reentrenamiento."
        )
    return [f"{i}. {text}" for i, text in enumerate(out, start=1)]


def render(results: dict[str, Any], files: dict[str, str], fig_dir: str = "docs/figures") -> str:
    f = results["final"]
    s = results["sizes"]
    stamp = ""
    if results.get("generated_at"):
        stamp = f"_Generado el {results['generated_at']} desde `{results.get('data_file', 'datos')}`._"
    lines = [START, "", stamp, "", "### Resumen de la decisión", "", *_decision_summary(results), ""]

    lines += ["### Por qué este modelo", "", *_conclusions(results), ""]

    lines += [
        "### 1. Datos y diseño de validación",
        "",
        "| Periodo | Préstamos | Rol |",
        "|---|---|---|",
        f"| Entrenamiento: {_period(results['periods']['train'])} | {s['train']:,} | Ajuste de modelos (default {_p(s['default_rate_train'])}) |",
        f"| Validación: {_period(results['periods']['val'])} | {s['validation']:,} | **Decide**: selección, tuning y umbral |",
        f"| Test: {_period(results['periods']['test'])} | {s['test']:,} | **Solo reporta** (default {_p(s['default_rate_test'])}) |",
        "",
        "### 2. Benchmark de candidatos",
        "",
        "| Modelo | AUC train | AUC validación | AUC test | Brecha train-val | KS test | Brier test | Latencia (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results["benchmark"]:
        lines.append(
            f"| {r['algorithm']} | {r['train_auc']:.4f} | {r['val_auc']:.4f} | {r['test_auc']:.4f} | "
            f"{r['val_gap']:+.4f} | {r['test_ks']:.4f} | {r['test_brier']:.4f} | {r['latency_ms']:.3f} |"
        )

    cfg = results["selection_config"]
    lines += [
        "",
        "### 3. Selección estadística en la validación",
        "",
        f"Regla: cada candidato se compara con el de mayor AUC de validación mediante el **test de DeLong** (mismos "
        f"préstamos), con **ajuste de Holm**. Se descarta solo si es peor de forma significativa (p Holm < {cfg['alpha']}) "
        f"**y** material (ΔAUC ≥ {cfg['practical_margin']}). Entre los que quedan gana el más simple; a igual complejidad, "
        "el de mayor AUC.",
        "",
        "| Modelo | AUC validación | Δ vs mejor | p Holm | Complejidad | Retenido | Elegido |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results["selection"]["rows"]:
        lines.append(
            f"| {r['model']} | {r['auc']:.4f} | {r['delta_vs_best']:+.4f} | {_pv(r['p_holm'])} | {r['complexity']} | "
            f"{'sí' if r['retained'] else 'no'} | {'**sí**' if r['selected'] else ''} |"
        )

    lines += [
        "",
        "### 4. Confirmación en el test fuera de tiempo",
        "",
        "Los candidatos se comparan en igualdad de condiciones (parámetros por defecto, mismos préstamos). El modelo "
        "final es el elegido, después ajustado con Optuna; su efecto se mide al final de esta sección.",
        "",
        _img(fig_dir, files, "auc_ci", "AUC por modelo con IC 95 % de DeLong"),
        "",
        _img(fig_dir, files, "pairwise", "Diferencias pareadas de AUC contra el modelo final"),
        "",
        "<details><summary>Todas las comparaciones pareadas (DeLong + Holm)</summary>",
        "",
        "| A | B | Δ AUC | IC 95 % | p | p Holm | Lectura |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results["comparison"]["pairwise"]:
        lines.append(
            f"| {r['model_a']} | {r['model_b']} | {r['delta']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
            f"{_pv(r['p_value'])} | {_pv(r['p_holm'])} | {r['interpretation']} |"
        )
    lines += ["", "</details>", ""]

    if "tuning" in results:
        t = results["tuning"]
        lines += [
            "**Efecto del tuning con Optuna** (mismo algoritmo, DeLong):",
            "",
            "| Periodo | AUC ajustado | AUC por defecto | Δ | p |",
            "|---|---|---|---|---|",
        ]
        for split, r in t.items():
            lines.append(
                f"| {'validación' if split == 'validation' else split} | {r['auc_tuned']:.4f} | {r['auc_default']:.4f} | {r['delta']:+.4f} | {_pv(r['p_value'])} |"
            )
        lines.append("")

    cal = f["calibration"]
    lines += [
        "### 5. Calibración",
        "",
        _img(fig_dir, files, "calibration", "Diagrama de fiabilidad"),
        "",
        "| Brier | Brier de referencia (sin modelo) | ECE | Pendiente | Intercepto | Spiegelhalter z | p |",
        "|---|---|---|---|---|---|---|",
        f"| {cal['brier']:.4f} | {cal['brier_reference']:.4f} | {cal['ece']:.4f} | {cal['slope']:.3f} | "
        f"{cal['intercept']:+.3f} | {cal['spiegelhalter_z']:.2f} | {_pv(cal['spiegelhalter_p'])} |",
        "",
        "### 6. Discriminación y ordenamiento de cartera",
        "",
        _img(fig_dir, files, "roc_pr", "Curvas ROC y Precision-Recall"),
        "",
        _img(fig_dir, files, "deciles", "Tasa de default por decil de riesgo"),
        "",
        "<details><summary>Tabla de deciles (ganancias y lift)</summary>",
        "",
        "| Decil | Préstamos | Default observado | IC 95 % (Wilson) | PD media | Captura acumulada | Lift |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in f["deciles"]:
        lines.append(
            f"| {r['decile']} | {r['n']:,} | {_p(r['default_rate'])} | [{_p(r['default_rate_ci'][0])}, "
            f"{_p(r['default_rate_ci'][1])}] | {_p(r['mean_pd'])} | {_p(r['cum_capture'])} | {r['lift']:.2f} |"
        )
    lines += ["", "</details>", ""]

    c = results["costs"]
    lines += [
        "### 7. Umbral de decisión por costos",
        "",
        f"El umbral minimiza el costo esperado en la **validación** con FN:FP = {c['fn']:g}:{c['fp']:g} (otorgar a "
        "quien incumple cuesta más que rechazar a un buen pagador) y se mide aquí en test.",
        "",
        _img(fig_dir, files, "threshold", "Costo esperado y trade-off aprobación vs morosidad"),
        "",
        "| Umbral | Aprobación | Default entre aprobados | Recall | Precisión | Costo esperado |",
        "|---|---|---|---|---|---|",
    ]
    picks = {round(f["threshold"], 2), 0.10, 0.15, 0.20, 0.25, 0.30}
    for r in f["thresholds"]:
        if round(r["threshold"], 2) in picks:
            mark = " **(elegido)**" if abs(r["threshold"] - f["threshold"]) < 0.005 else ""
            lines.append(
                f"| {r['threshold']:.2f}{mark} | {_p(r['approval_rate'])} | {_p(r['bad_rate_approved'])} | "
                f"{_p(r['recall'])} | {_p(r['precision'])} | {r['expected_cost']:.3f} |"
            )
    lines.append("")

    if results["vintages"]:
        lines += [
            "### 8. Estabilidad por cosecha (incluye producción nunca vista)",
            "",
            _img(fig_dir, files, "vintages", "AUC y calibración por trimestre de emisión"),
            "",
            "<details><summary>Tabla por trimestre</summary>",
            "",
            "| Trimestre | Préstamos | AUC | IC 95 % | Default observado | PD media |",
            "|---|---|---|---|---|---|",
        ]
        for v in results["vintages"]:
            lines.append(
                f"| {v['quarter']} | {v['n']:,} | {v['auc']:.4f} | {_ci(v['auc_ci'])} | {_p(v['observed'])} | {_p(v['mean_pd'])} |"
            )
        lines += ["", "</details>", ""]

    ab = results["ablation"]
    lines += [
        "### 9. Explicabilidad (SHAP)",
        "",
        _img(fig_dir, files, "shap", "Importancia global SHAP"),
        "",
        _img(fig_dir, files, "shap_dependence", "Dependencia SHAP de las variables principales"),
        "",
        "| Variable | Media de abs(SHAP) | Dirección (Spearman valor-SHAP) |",
        "|---|---|---|",
    ]
    for r in results["shap"]:
        sp = r["spearman"]
        direction = (
            "categórica"
            if sp is None or sp != sp
            else f"{sp:+.2f} ({'más valor, más riesgo' if sp > 0 else 'más valor, menos riesgo'})"
        )
        lines.append(f"| {r['feature']} | {r['mean_abs_shap']:.4f} | {direction} |")
    lines += [
        "",
        "SHAP describe asociaciones que el modelo aprendió, no efectos causales.",
        "",
        "### 10. Ablación: ¿cuánto depende del scoring de Lending Club?",
        "",
        "| Variables | AUC test | Δ vs completo | IC 95 % | p (DeLong) |",
        "|---|---|---|---|---|",
        f"| Todas | {ab['auc_full']:.4f} | | | |",
        f"| Sin {', '.join(ab['dropped'])} | {ab['auc_without']:.4f} | {-ab['delta']:+.4f} | "
        f"[{-ab['delta_ci'][1]:+.4f}, {-ab['delta_ci'][0]:+.4f}] | {_pv(ab['p_value'])} |",
        "",
        "### 11. Scorecard WoE e Information Value",
        "",
        _img(fig_dir, files, "iv", "Information Value por variable"),
        "",
        f"El scorecard (benchmark regulatorio) obtiene AUC {results['scorecard']['discrimination']['auc']:.4f} "
        f"IC {_ci(results['scorecard']['discrimination']['auc_ci'])} en test.",
        "",
        "| Variable | IV | Poder predictivo |",
        "|---|---|---|",
        *[f"| {r['feature']} | {r['iv']:.3f} | {r['strength']} |" for r in results["scorecard"]["iv"][:10]],
        "",
        END,
    ]
    return "\n".join(line for line in lines if line is not None)


def update_readme(readme_text: str, section: str) -> str:
    """Reemplaza el bloque entre marcadores; si no existen, falla (no adivina dónde insertar)."""
    if START not in readme_text or END not in readme_text:
        raise ValueError("El README no tiene los marcadores de resultados.")
    head = readme_text.split(START)[0]
    tail = readme_text.split(END, 1)[1]
    return head + section + tail
