"""Registro de predicciones y desenlaces en Delta Lake (feedback loop del monitoreo)."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

NUMERIC = [
    "loan_amnt",
    "term_months",
    "int_rate",
    "installment",
    "emp_length_years",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_score",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "credit_history_months",
]
CATEGORICAL = [
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
    "initial_list_status",
]
LOG_COLUMNS = [
    "request_id",
    "loan_id",
    "event_ts",
    "source",
    "variant",
    "model_version",
    "probability",
    "decision",
    "risk_band",
    "latency_ms",
    *NUMERIC,
    *CATEGORICAL,
]
OUTCOME_COLUMNS = ["loan_id", "request_id", "actual_default", "observed_ts", "source"]
TIMESTAMPS = {"event_ts", "observed_ts"}


def sql_literal(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        return repr(float(value)) if isinstance(value, float) else str(value)
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def insert_statement(fq_table: str, columns: list[str], rows: list[dict]) -> str:
    values = []
    for row in rows:
        cells = []
        for col in columns:
            lit = sql_literal(row.get(col))
            cells.append(f"CAST({lit} AS TIMESTAMP)" if col in TIMESTAMPS and lit != "NULL" else lit)
        values.append("(" + ", ".join(cells) + ")")
    return f"INSERT INTO {fq_table} ({', '.join(columns)}) VALUES " + ", ".join(values)


class MemorySink:
    def __init__(self) -> None:
        self.predictions: list[dict] = []
        self.outcomes: list[dict] = []

    def log_prediction(self, row: dict) -> None:
        self.predictions.append(row)

    def log_outcome(self, row: dict) -> None:
        self.outcomes.append(row)

    def flush(self) -> None:
        return None


class DeltaSink:
    """Buffer que se vacía en lotes vía SQL Statement Execution (warehouse 2X-Small de Free Edition)."""

    def __init__(
        self, client, warehouse_id: str, fq_schema: str, flush_every_n: int = 20, flush_every_seconds: float = 10.0
    ) -> None:
        self.client, self.warehouse_id, self.fq = client, warehouse_id, fq_schema
        self.flush_every_n, self.flush_every_seconds = flush_every_n, flush_every_seconds
        self._buffer: dict[str, list[dict]] = {"inference_log": [], "outcomes": []}
        self._lock = threading.Lock()
        self._last = time.monotonic()

    def _maybe_flush(self) -> None:
        pending = sum(len(v) for v in self._buffer.values())
        if pending >= self.flush_every_n or time.monotonic() - self._last > self.flush_every_seconds:
            self.flush()

    def log_prediction(self, row: dict) -> None:
        with self._lock:
            self._buffer["inference_log"].append(row)
        self._maybe_flush()

    def log_outcome(self, row: dict) -> None:
        with self._lock:
            self._buffer["outcomes"].append(row)
        self._maybe_flush()

    def flush(self) -> None:
        with self._lock:
            batches = {k: v for k, v in self._buffer.items() if v}
            self._buffer = {"inference_log": [], "outcomes": []}
            self._last = time.monotonic()
        for table, rows in batches.items():
            columns = LOG_COLUMNS if table == "inference_log" else OUTCOME_COLUMNS
            try:
                self.client.sql(self.warehouse_id, insert_statement(f"{self.fq}.{table}", columns, rows))
            except Exception as exc:  # nunca perder datos si el warehouse falla
                logger.error("Fallo al escribir en Delta (%s); se reencolan %d filas", exc, len(rows))
                with self._lock:
                    self._buffer[table].extend(rows)
