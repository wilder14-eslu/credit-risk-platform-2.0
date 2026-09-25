"""Generador de datos sintéticos con el MISMO formato crudo que Lending Club.

Sirve para tres cosas: tests en CI (sin descargar 1.6 GB), el test end-to-end
local y un modo demo en Databricks antes de subir el dataset real. Incluye
drift temporal deliberado desde 2014 (más grados riesgosos y cambio en
P(default|x)) y préstamos recientes sin desenlace, como en los datos reales.
NO reemplaza al dataset real: los resultados reportados salen de Lending Club.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GRADES = list("ABCDEFG")
PURPOSES = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "other",
    "major_purchase",
    "small_business",
    "car",
    "medical",
    "moving",
    "vacation",
    "house",
    "wedding",
]
STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "WA", "AZ", "MA"]
EMP = ["< 1 year", "1 year", *[f"{i} years" for i in range(2, 10)], "10+ years", None]


def _months(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="MS")


def generate_raw(
    n_rows: int = 20_000, seed: int = 42, start: str = "2008-01-01", end: str = "2018-12-01"
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = _months(start, end)
    weights = np.linspace(1, 12, len(months)) ** 1.5  # el volumen crece con los años
    issue = rng.choice(months, size=n_rows, p=weights / weights.sum())
    issue = pd.DatetimeIndex(issue)
    year = issue.year.to_numpy()
    late = (year >= 2014).astype(float)

    # Mezcla de grados: desde 2014 crece la proporción de grados riesgosos (covariate shift)
    g_probs_early = np.array([0.20, 0.30, 0.25, 0.14, 0.07, 0.03, 0.01])
    g_probs_late = np.array([0.15, 0.25, 0.27, 0.17, 0.10, 0.04, 0.02])
    grade_idx = np.where(
        late == 1,
        rng.choice(7, size=n_rows, p=g_probs_late),
        rng.choice(7, size=n_rows, p=g_probs_early),
    )
    grade = np.array(GRADES)[grade_idx]
    sub = rng.integers(1, 6, n_rows)
    base_rate = np.array([7.0, 10.8, 14.0, 17.5, 20.8, 24.5, 27.8])[grade_idx]
    int_rate = np.round(base_rate + (sub - 3) * 0.6 + rng.normal(0, 0.4, n_rows), 2)
    term = np.where(rng.random(n_rows) < 0.25 + 0.05 * grade_idx, 60, 36)
    loan = np.round(rng.lognormal(9.4, 0.6, n_rows) / 25) * 25
    loan = np.clip(loan, 1000, 40000)
    r = int_rate / 1200
    installment = np.round(loan * r / (1 - (1 + r) ** (-term)), 2)
    income = np.round(rng.lognormal(11.0 - 0.1 * late, 0.5, n_rows), -2)
    dti = np.clip(rng.normal(17 + 1.5 * grade_idx + 2 * late, 7, n_rows), 0, 60).round(2)
    fico_low = np.clip(np.round(rng.normal(730 - 12 * grade_idx, 25, n_rows) / 5) * 5, 660, 845)
    inq = rng.poisson(0.5 + 0.2 * grade_idx)
    delinq = rng.poisson(0.25, n_rows)
    open_acc = rng.poisson(11, n_rows) + 1
    total_acc = open_acc + rng.poisson(12, n_rows)
    revol_bal = np.round(rng.lognormal(9.3, 0.9, n_rows))
    revol_util = np.clip(rng.normal(50 + 4 * grade_idx, 22, n_rows), 0, 130).round(1)
    history_years = np.clip(rng.normal(16, 7, n_rows), 3, 50)
    earliest = issue - pd.to_timedelta((history_years * 365.25).astype(int), unit="D")
    emp = np.array(EMP, dtype=object)[rng.integers(0, len(EMP), n_rows)]
    home = rng.choice(["MORTGAGE", "RENT", "OWN", "ANY"], size=n_rows, p=[0.49, 0.40, 0.109, 0.001])
    verif = rng.choice(["Not Verified", "Source Verified", "Verified"], size=n_rows)
    purpose = rng.choice(PURPOSES, size=n_rows, p=np.array([55, 22, 6, 5, 2, 1.5, 1.2, 1.2, 0.8, 0.7, 0.9, 0.3]) / 96.6)
    state = rng.choice(STATES, size=n_rows)
    app_type = np.where((year >= 2016) & (rng.random(n_rows) < 0.05), "Joint App", "Individual")
    list_status = np.where(rng.random(n_rows) < np.where(year >= 2013, 0.7, 0.2), "w", "f")
    mort_acc = rng.poisson(np.where(home == "MORTGAGE", 2.5, 0.4))
    pub_rec = rng.poisson(0.2, n_rows)
    bankrupt = np.minimum(pub_rec, rng.poisson(0.1, n_rows))

    # Probabilidad de default. Desde 2014 cambia la relación (concept drift):
    # el DTI pesa más y aparece un shock en préstamos para pequeñas empresas.
    logit = (
        -2.9
        + 0.30 * grade_idx
        + 0.06 * (int_rate - 13)
        + 0.35 * (term == 60)
        + (0.020 + 0.025 * late) * (dti - 18)
        - 0.008 * (fico_low - 700)
        + 0.12 * inq
        + 0.10 * (revol_util / 50)
        - 0.25 * (np.log(income) - 11)
        + 0.15 * (home == "RENT")
        + (0.4 + 0.6 * late) * (purpose == "small_business")
        + 0.15 * late
    )
    default = rng.random(n_rows) < 1 / (1 + np.exp(-logit))

    # Préstamos recientes aún sin desenlace (como en el archivo real de 2018Q4)
    months_to_end = (pd.Timestamp(end).year - year) * 12 + (pd.Timestamp(end).month - issue.month)
    unresolved = rng.random(n_rows) < np.clip(1 - months_to_end / term, 0, 1) * 0.9
    status = np.where(default, "Charged Off", "Fully Paid").astype(object)
    status[unresolved] = rng.choice(
        ["Current", "Late (31-120 days)", "In Grace Period"], size=int(unresolved.sum()), p=[0.93, 0.05, 0.02]
    )
    early_policy = (year <= 2010) & (rng.random(n_rows) < 0.02)
    status[early_policy & ~unresolved] = [
        f"Does not meet the credit policy. Status:{s}" for s in status[early_policy & ~unresolved]
    ]

    ids = rng.choice(np.arange(1_000_000, 200_000_000), size=n_rows, replace=False)
    raw = pd.DataFrame(
        {
            "id": ids.astype(str),
            "loan_amnt": loan,
            "funded_amnt": loan,
            "term": [f" {t} months" for t in term],
            "int_rate": int_rate,
            "installment": installment,
            "grade": grade,
            "sub_grade": [f"{g}{s}" for g, s in zip(grade, sub, strict=True)],
            "emp_length": emp,
            "home_ownership": home,
            "annual_inc": income,
            "verification_status": verif,
            "issue_d": issue.strftime("%b-%Y"),
            "loan_status": status,
            "purpose": purpose,
            "addr_state": state,
            "dti": dti,
            "delinq_2yrs": delinq.astype(float),
            "earliest_cr_line": earliest.strftime("%b-%Y"),
            "fico_range_low": fico_low,
            "fico_range_high": fico_low + 4,
            "inq_last_6mths": inq.astype(float),
            "open_acc": open_acc.astype(float),
            "pub_rec": pub_rec.astype(float),
            "revol_bal": revol_bal,
            "revol_util": revol_util,
            "total_acc": total_acc.astype(float),
            "initial_list_status": list_status,
            "application_type": app_type,
            "mort_acc": mort_acc.astype(float),
            "pub_rec_bankruptcies": bankrupt.astype(float),
            # columnas de leakage (deben descartarse)
            "total_pymnt": np.where(default, loan * 0.4, loan * 1.2),
            "recoveries": np.where(default, loan * 0.1, 0.0),
            "last_fico_range_high": np.where(default, 550, 750),
        }
    )
    # Filas de pie de página como en los CSV originales de Lending Club
    footer = pd.DataFrame([{"id": "Total amount funded in policy code 1: 1000000"}])
    return pd.concat(
        [raw.sort_values("issue_d", key=lambda s: pd.to_datetime(s, format="%b-%Y")), footer], ignore_index=True
    )
