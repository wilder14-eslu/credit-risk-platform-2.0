"""Job 5 - Replay de producción con originaciones reales posteriores al entrenamiento.

Cada corrida "juega" los siguientes `months_per_run` meses de Lending Club (desde
2015-01): asigna variante A/B con el mismo hash que la API, puntúa con
champion/challenger y escribe en `inference_log`. El desenlace real se registra
en `outcomes` con fecha de observación = originación + `label_delay_months`.
El drift que detecta el monitoreo es el de los datos reales, no uno inventado.
"""

# Bootstrap: agrega <raíz del bundle>/src al path (el bundle pasa --project-root).
import os
import sys

_root = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--project-root"), None)
if _root is None:
    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        _root = os.getcwd()
sys.path.insert(0, os.path.join(_root, "src"))
os.environ.setdefault("CREDIT_RISK_CONFIG_DIR", os.path.join(_root, "config"))

import uuid  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import (  # noqa: E402
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    date_column,
    id_column,
    input_features,
    platform_config,
    target_name,
)
from credit_risk.monitoring.ab_testing import assign_variant  # noqa: E402
from credit_risk.monitoring.replay import next_months  # noqa: E402
from credit_risk.registry import mlflow_registry as R  # noqa: E402


def main() -> None:
    lh.configure_logging()
    rcfg = platform_config()["replay"]
    args = lh.job_args({"months": str(rcfg["months_per_run"])})
    names = lh.names_from(args)
    R.setup_mlflow()

    last = lh.read_pandas(f"SELECT max(event_ts) AS m FROM {names.inference_log} WHERE source = 'replay'")["m"].iloc[0]
    available = lh.read_pandas(f"SELECT max({date_column()}) AS m FROM {names.feature_table}")["m"].iloc[0]
    dcfg = platform_config()["data"]
    end = min(pd.Timestamp(available), pd.Timestamp(dcfg["production_end"]))
    months = next_months(last, dcfg["production_start"], int(args.months), end)
    if not months:
        lh.logger.info("El replay ya llegó al final del periodo de producción (%s)", end.date())
        lh.set_task_value("clock", str(pd.Timestamp(last).date()))
        return

    champion_v = R.get_alias_version(names, CHAMPION_ALIAS)
    if champion_v is None:
        raise RuntimeError("No hay champion desplegado")
    models = {"champion": (champion_v, R.load_credit_model(names, CHAMPION_ALIAS))}
    challenger_v = R.get_alias_version(names, CHALLENGER_ALIAS)
    traffic = 0.0
    if challenger_v:
        models["challenger"] = (challenger_v, R.load_credit_model(names, CHALLENGER_ALIAS))
        traffic = platform_config()["ab_testing"]["challenger_traffic"]

    cols = ", ".join([id_column(), date_column(), *input_features(), target_name()])
    delay = pd.DateOffset(months=rcfg["label_delay_months"])
    total = 0
    for month in months:
        batch = lh.read_pandas(f"SELECT {cols} FROM {names.feature_table} WHERE {date_column()} = '{month.date()}'")
        if len(batch) > rcfg["max_rows_per_month"]:
            batch = batch.sample(rcfg["max_rows_per_month"], random_state=month.month)
        if batch.empty:
            continue
        batch["variant"] = batch[id_column()].map(lambda a: assign_variant(str(a), traffic))
        logs = []
        for variant, group in batch.groupby("variant"):
            version, model = models[variant]
            scored = model.predict_frame(group[list(input_features())], explain=False)
            frame = group[[id_column(), *input_features(), target_name()]].copy()
            frame["request_id"] = [str(uuid.uuid4()) for _ in range(len(frame))]
            frame["event_ts"] = month
            frame["source"], frame["variant"], frame["model_version"] = "replay", variant, version
            for col in ("probability", "decision", "risk_band"):
                frame[col] = scored[col].to_numpy()
            frame["latency_ms"] = np.nan
            logs.append(frame)
        log = pd.concat(logs, ignore_index=True)
        lh.write_pandas(log.drop(columns=[target_name()]), names.inference_log)
        labeled = log.dropna(subset=[target_name()])
        lh.write_pandas(
            pd.DataFrame(
                {
                    id_column(): labeled[id_column()],
                    "request_id": labeled["request_id"],
                    "actual_default": labeled[target_name()].astype(int),
                    "observed_ts": month + delay,
                    "source": "replay",
                }
            ),
            names.outcomes,
        )
        total += len(log)
        lh.logger.info("Replay %s: %d solicitudes (%d con desenlace)", month.date(), len(log), len(labeled))

    lh.set_task_value("clock", str(months[-1].date()))
    lh.set_task_value("replayed_rows", total)


if __name__ == "__main__":
    main()
