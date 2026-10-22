"""Registro de predicciones y desenlaces en Delta Lake (feedback loop del monitoreo)."""

from __future__ import annotations

import logging
import threading

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
    "model_latency_ms",
    "policy_version",
    "reasons",
    "input_hash",
    *NUMERIC,
    *CATEGORICAL,
]
OUTCOME_COLUMNS = ["loan_id", "request_id", "actual_default", "observed_ts", "source"]
TIMESTAMPS = {"event_ts", "observed_ts"}
DOUBLES = {"probability", "latency_ms", "model_latency_ms", *NUMERIC}
INTEGERS = {"actual_default"}
# Límite prudente de parámetros por sentencia (el resto de filas va en otra sentencia).
MAX_PARAMETERS = 250


def sql_type(column: str) -> str:
    if column in TIMESTAMPS:
        return "TIMESTAMP"
    if column in DOUBLES:
        return "DOUBLE"
    if column in INTEGERS:
        return "INT"
    return "STRING"


def parameter_value(value) -> str | None:
    """Valor de un parámetro de Statement Execution (siempre texto; None = NULL)."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def insert_statements(fq_table: str, columns: list[str], rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """INSERT parametrizados: los datos nunca se concatenan al SQL (sin inyección posible).

    Solo los nombres de tabla y columnas, que vienen del código y no del usuario, forman parte del texto.
    """
    per_statement = max(1, MAX_PARAMETERS // len(columns))
    out = []
    for start in range(0, len(rows), per_statement):
        chunk = rows[start : start + per_statement]
        values, params = [], []
        for i, row in enumerate(chunk):
            cells = []
            for col in columns:
                name = f"p{i}_{col}"
                cells.append(f":{name}")
                params.append({"name": name, "value": parameter_value(row.get(col)), "type": sql_type(col)})
            values.append("(" + ", ".join(cells) + ")")
        out.append((f"INSERT INTO {fq_table} ({', '.join(columns)}) VALUES " + ", ".join(values), params))
    return out


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

    def close(self) -> None:
        return None


class DeltaSink:
    """Buffer que se vacía en lotes vía SQL Statement Execution (warehouse 2X-Small de Free Edition).

    La escritura corre en un hilo de fondo: una solicitud a /predict nunca espera al warehouse.
    """

    def __init__(
        self,
        client,
        warehouse_id: str,
        fq_schema: str,
        flush_every_n: int = 20,
        flush_every_seconds: float = 10.0,
        background: bool = True,
    ) -> None:
        self.client, self.warehouse_id, self.fq = client, warehouse_id, fq_schema
        self.flush_every_n, self.flush_every_seconds = flush_every_n, flush_every_seconds
        self._buffer: dict[str, list[dict]] = {"inference_log": [], "outcomes": []}
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._columns: dict[str, list[str]] = {}
        self._thread = None
        if background:
            self._thread = threading.Thread(target=self._run, name="delta-sink", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.flush_every_seconds)
            self._wake.clear()
            self.flush()

    def _pending(self) -> int:
        return sum(len(v) for v in self._buffer.values())

    def _enqueue(self, table: str, row: dict) -> None:
        with self._lock:
            self._buffer[table].append(row)
            full = self._pending() >= self.flush_every_n
        if full:
            if self._thread is not None:
                self._wake.set()  # el hilo de fondo escribe; la solicitud no espera
            else:
                self.flush()

    def log_prediction(self, row: dict) -> None:
        self._enqueue("inference_log", row)

    def log_outcome(self, row: dict) -> None:
        self._enqueue("outcomes", row)

    def _target_columns(self, table: str, wanted: list[str]) -> list[str]:
        """Solo columnas que existen en la tabla (tolera una tabla aún sin migrar)."""
        if table not in self._columns:
            try:
                rows = self.client.sql(self.warehouse_id, f"DESCRIBE TABLE {self.fq}.{table}")
                existing = {str(r.get("col_name", "")).lower() for r in rows}
                self._columns[table] = [c for c in wanted if c.lower() in existing] if existing else wanted
            except Exception as exc:
                logger.warning("No se pudo leer el esquema de %s (%s); se usan todas las columnas", table, exc)
                return wanted
        return self._columns[table]

    def flush(self) -> None:
        with self._flush_lock:
            with self._lock:
                batches = {k: v for k, v in self._buffer.items() if v}
                self._buffer = {"inference_log": [], "outcomes": []}
            for table, rows in batches.items():
                columns = self._target_columns(table, LOG_COLUMNS if table == "inference_log" else OUTCOME_COLUMNS)
                for i, (statement, params) in enumerate(insert_statements(f"{self.fq}.{table}", columns, rows)):
                    try:
                        self.client.sql(self.warehouse_id, statement, params)
                    except Exception as exc:  # nunca perder datos si el warehouse falla
                        per = max(1, MAX_PARAMETERS // len(columns))
                        pending = rows[i * per :]
                        logger.error("Fallo al escribir en Delta (%s); se reencolan %d filas", exc, len(pending))
                        with self._lock:
                            self._buffer[table].extend(pending)
                        break

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.flush()
