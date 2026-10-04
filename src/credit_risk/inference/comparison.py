"""Comparación de modelos: Holm, significancia práctica y regla de parsimonia.

Significancia estadística responde "¿la diferencia es distinguible del ruido?";
significancia práctica responde "¿la diferencia importa?". Con decenas de miles de
préstamos, 0.002 de AUC puede ser "significativo" y operativamente irrelevante, así
que ambas se leen juntas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

from credit_risk.inference.delong import delong_auc_ci, delong_roc_test

# Margen de equivalencia práctica para ROC-AUC. Convención de trabajo, no un
# estándar regulatorio: diferencias menores no cambian de forma apreciable el
# ordenamiento de riesgo. Es un parámetro (config/platform.yaml -> selection).
DEFAULT_PRACTICAL_MARGIN = 0.005


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """P-valores ajustados por el método step-down de Holm (1979).

    Controla la tasa de error familiar (FWER) y es uniformemente más potente que
    Bonferroni. Conserva el orden original de la entrada.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return []
    if np.any(np.isnan(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Los p-valores deben estar en [0, 1].")
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = (m - np.arange(m)) * p[order]
    adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return [float(v) for v in adjusted]


def interpret_difference(
    delta: float, adjusted_p_value: float, practical_margin: float = DEFAULT_PRACTICAL_MARGIN, alpha: float = 0.05
) -> str:
    """Lectura conjunta de significancia estadística y práctica (delta = A - B)."""
    significant = adjusted_p_value < alpha
    material = abs(delta) >= practical_margin
    if significant and material:
        return f"Diferencia significativa y material a favor de {'A' if delta > 0 else 'B'}"
    if significant:
        return "Estadísticamente significativa pero prácticamente irrelevante"
    if material:
        return "Diferencia de tamaño relevante pero no concluyente (incertidumbre alta)"
    return "Sin evidencia de diferencia: modelos prácticamente equivalentes"


def auc_table(y_true: Any, scores: Mapping[str, Any], alpha: float = 0.05) -> list[dict[str, Any]]:
    """AUC de cada modelo con su IC de DeLong, ordenado de mayor a menor."""
    rows = [{"model": name, **delong_auc_ci(y_true, s, alpha)} for name, s in scores.items()]
    return sorted(rows, key=lambda r: r["auc"], reverse=True)


def pairwise_delong_comparison(
    y_true: Any,
    scores: Mapping[str, Any],
    practical_margin: float = DEFAULT_PRACTICAL_MARGIN,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Todas las parejas con DeLong, IC de la diferencia y ajuste de Holm.

    El orden de cada pareja sigue el de ``scores``; delta = AUC(A) - AUC(B).
    """
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    rows: list[dict[str, Any]] = []
    for a, b in combinations(list(scores), 2):
        test = delong_roc_test(y_true, scores[a], scores[b])
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "auc_a": test["auc_a"],
                "auc_b": test["auc_b"],
                "delta": test["diff"],
                "ci_low": test["diff"] - z_crit * test["se"],
                "ci_high": test["diff"] + z_crit * test["se"],
                "p_value": test["p_value"],
            }
        )
    for row, p_adj in zip(rows, holm_adjust([r["p_value"] for r in rows]), strict=True):
        row["p_holm"] = p_adj
        row["interpretation"] = interpret_difference(row["delta"], p_adj, practical_margin, alpha)
    return rows


def select_parsimonious(
    y_true: Any,
    scores: Mapping[str, Any],
    complexity: Mapping[str, int],
    eligible: Sequence[str] | None = None,
    practical_margin: float = DEFAULT_PRACTICAL_MARGIN,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Regla de parsimonia: el modelo más simple que no sea peor de forma material.

    1. El de mayor AUC entre los elegibles es la referencia.
    2. Cada candidato se compara con la referencia por DeLong (Holm sobre las
       comparaciones). Se descarta solo si es **peor de forma significativa y
       material** (p Holm < alpha y diferencia >= margen práctico).
    3. Entre los que sobreviven gana el de menor complejidad; a igual complejidad,
       el de mayor AUC. Los no elegibles (latencia o sobreajuste) se reportan
       pero no compiten.

    Debe aplicarse sobre el periodo de **validación**: el test queda reservado
    para medir una sola vez al modelo elegido.
    """
    eligible_set = set(eligible) if eligible is not None else set(scores)
    names = [n for n in scores if n in eligible_set]
    if not names:
        raise ValueError("No hay candidatos elegibles para seleccionar.")
    aucs = {n: delong_auc_ci(y_true, s)["auc"] for n, s in scores.items()}
    reference = max(names, key=lambda n: aucs[n])
    tests = {n: delong_roc_test(y_true, scores[reference], scores[n]) for n in scores if n != reference}
    # Holm solo sobre la familia de comparaciones que deciden (candidatos elegibles).
    decisive = [n for n in names if n != reference]
    p_holm = dict(zip(decisive, holm_adjust([tests[n]["p_value"] for n in decisive]), strict=True))

    rows = [
        {
            "model": reference,
            "auc": aucs[reference],
            "delta_vs_best": 0.0,
            "p_holm": 1.0,
            "eligible": True,
            "retained": True,
        }
    ]
    for n, test in tests.items():
        delta = test["diff"]  # AUC(referencia) - AUC(n)
        is_eligible = n in eligible_set
        p_adj = p_holm.get(n, float("nan"))
        materially_worse = is_eligible and p_adj < alpha and delta >= practical_margin
        rows.append(
            {
                "model": n,
                "auc": aucs[n],
                "delta_vs_best": delta,
                "p_holm": p_adj,
                "eligible": is_eligible,
                "retained": is_eligible and not materially_worse,
            }
        )
    retained = [r for r in rows if r["retained"]]
    chosen = min(retained, key=lambda r: (complexity.get(r["model"], 99), -r["auc"]))
    for r in rows:
        r["complexity"] = complexity.get(r["model"], 99)
        r["selected"] = r["model"] == chosen["model"]
    rows.sort(key=lambda r: -r["auc"])
    reason = (
        f"{chosen['model']} es el de mayor AUC de validación"
        if chosen["model"] == reference
        else f"{chosen['model']} no es peor de forma material que {reference} "
        f"(ΔAUC {aucs[reference] - chosen['auc']:+.4f}) y es más simple"
    )
    return {"selected": chosen["model"], "reference": reference, "reason": reason, "rows": rows}
