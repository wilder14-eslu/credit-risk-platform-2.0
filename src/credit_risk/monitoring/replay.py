"""Replay temporal: "producción" con préstamos reales emitidos después del entrenamiento.

En lugar de inventar drift, se re-juega la originación real de Lending Club mes a
mes (2015 en adelante). El reloj simulado es el último mes procesado y las
etiquetas reales solo se "conocen" `label_delay_months` después, como en un banco.
"""

from __future__ import annotations

import pandas as pd


def month_start(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def next_months(last_processed, production_start, n_months: int, last_available=None) -> list[pd.Timestamp]:
    """Meses a procesar en esta corrida (continúa donde quedó la anterior)."""
    start = (
        month_start(production_start)
        if last_processed is None or pd.isna(last_processed)
        else month_start(last_processed) + pd.DateOffset(months=1)
    )
    months = [start + pd.DateOffset(months=i) for i in range(n_months)]
    if last_available is not None:
        months = [m for m in months if m <= month_start(last_available)]
    return months


def monitoring_windows(clock, window_months: int, label_delay_months: int) -> dict[str, pd.Timestamp]:
    """Ventanas relativas al reloj simulado.

    - data drift: originaciones de los últimos `window_months` (datos sin etiqueta).
    - concept drift: originaciones cuya etiqueta ya se conoce (reloj - retraso).
    """
    clock = month_start(clock)
    label_end = clock - pd.DateOffset(months=label_delay_months)
    return {
        "clock": clock,
        "drift_start": clock - pd.DateOffset(months=window_months - 1),
        "drift_end": clock,
        "label_start": label_end - pd.DateOffset(months=window_months - 1),
        "label_end": label_end,
    }


def months_between(start, end) -> int:
    a, b = month_start(start), month_start(end)
    return (b.year - a.year) * 12 + (b.month - a.month)
