"""Job 6 - Monitoreo: data drift, prediction drift, concept drift y prior shift.

Todo se calcula respecto al reloj simulado del replay (último mes procesado):
- data/prediction drift: originaciones de los últimos `window_months` vs el perfil de entrenamiento.
- concept drift: originaciones cuyo desenlace ya se observó (reloj - retraso de etiquetas).
Publica `retrain=true|false` y `clock`; una condition_task decide si disparar Continuous Training.
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

import json  # noqa: E402
import uuid  # noqa: E402

import pandas as pd  # noqa: E402

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import CHAMPION_ALIAS, input_features, platform_config  # noqa: E402
from credit_risk.monitoring.concept_drift import concept_drift_report  # noqa: E402
from credit_risk.monitoring.data_drift import feature_drift  # noqa: E402
from credit_risk.monitoring.decision import retrain_decision  # noqa: E402
from credit_risk.monitoring.replay import monitoring_windows  # noqa: E402
from credit_risk.registry import mlflow_registry as R  # noqa: E402


def main() -> None:
    lh.configure_logging()
    args = lh.job_args()
    names = lh.names_from(args)
    cfg = platform_config()["monitoring"]
    R.setup_mlflow()

    version = R.get_alias_version(names, CHAMPION_ALIAS)
    clock = lh.read_pandas(f"SELECT max(event_ts) AS m FROM {names.inference_log} WHERE source = 'replay'")["m"].iloc[0]
    if version is None or pd.isna(clock):
        lh.logger.warning("Sin champion o sin tráfico todavía: nada que monitorear")
        lh.set_task_value("retrain", "false")
        lh.set_task_value("clock", "")
        return
    win = monitoring_windows(clock, cfg["window_months"], platform_config()["replay"]["label_delay_months"])

    ref = lh.read_pandas(
        f"SELECT * FROM {names.reference_profile} WHERE model_version = '{version}' ORDER BY created_ts DESC LIMIT 1"
    )
    profile = json.loads(ref.iloc[0]["profile_json"])
    reference = json.loads(ref.iloc[0]["reference_metrics_json"])

    recent = lh.read_pandas(f"""
        SELECT * FROM {names.inference_log}
        WHERE source = 'replay' AND variant = 'champion' AND model_version = '{version}'
          AND event_ts BETWEEN '{win["drift_start"].date()}' AND '{win["drift_end"].date()}'
    """)
    current = recent[list(input_features())].copy()
    current["__score__"] = recent["probability"]
    drift = feature_drift(profile, current, cfg["psi_warning"], cfg["psi_alert"], cfg["ks_pvalue_alert"])

    labeled = lh.read_pandas(f"""
        SELECT l.probability, l.event_ts, o.actual_default
        FROM {names.inference_log} l JOIN {names.outcomes} o ON l.request_id = o.request_id
        WHERE l.source = 'replay' AND l.variant = 'champion' AND l.model_version = '{version}'
          AND l.event_ts BETWEEN '{win["label_start"].date()}' AND '{win["label_end"].date()}'
          AND o.observed_ts <= '{win["clock"].date()}'
        ORDER BY l.event_ts
    """)
    concept = None
    if len(labeled) >= cfg["min_rows"] and labeled["actual_default"].nunique() == 2:
        threshold = float(R.version_tags(names, version).get("threshold", 0.5))
        concept = concept_drift_report(
            labeled["actual_default"].to_numpy(), labeled["probability"].to_numpy(), reference, cfg, threshold=threshold
        )

    last = lh.read_pandas(f"SELECT max(clock_month) AS m FROM {names.retrain_events}")["m"].iloc[0]
    last = None if pd.isna(last) else pd.Timestamp(last).to_pydatetime()
    decision = retrain_decision(drift, concept, cfg, len(recent), last, win["clock"].to_pydatetime())

    run_id, now = str(uuid.uuid4()), lh.utcnow()
    if not drift.empty:
        lh.write_pandas(
            drift.assign(run_id=run_id, run_ts=now, clock_month=win["clock"], model_version=version),
            names.drift_features,
        )
    cur = concept["current"] if concept else {}
    pred_psi = drift.loc[drift["feature"] == "prediction", "psi"]
    lh.write_pandas(
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "run_ts": now,
                    "clock_month": win["clock"],
                    "model_version": version,
                    "n_rows": int(len(recent)),
                    "n_labeled": int(len(labeled)),
                    "features_drifted": len(decision.get("drifted_features", [])),
                    "prediction_psi": float(pred_psi.iloc[0]) if len(pred_psi) else None,
                    "roc_auc": cur.get("roc_auc"),
                    "ks": cur.get("ks"),
                    "brier": cur.get("brier"),
                    "auc_drop": concept["auc_drop"] if concept else None,
                    "default_rate_observed": cur.get("default_rate_observed"),
                    "default_rate_predicted": cur.get("default_rate_predicted"),
                    "ddm_state": concept["ddm_state"] if concept else None,
                    "page_hinkley_statistic": concept["page_hinkley_statistic"] if concept else None,
                    "concept_drift": bool(concept["concept_drift"]) if concept else False,
                    "severity": decision["severity"],
                    "retrain": decision["retrain"],
                    "reasons": json.dumps(decision["reasons"], ensure_ascii=False),
                }
            ]
        ),
        names.monitoring_metrics,
    )

    if decision["retrain"]:
        lh.write_pandas(
            pd.DataFrame(
                [
                    {
                        "event_ts": now,
                        "clock_month": win["clock"],
                        "trigger": "monitoring",
                        "reasons": json.dumps(decision["reasons"], ensure_ascii=False),
                        "model_version": version,
                    }
                ]
            ),
            names.retrain_events,
        )
    lh.logger.info("Reloj %s | decisión: %s", win["clock"].date(), decision)
    lh.set_task_value("retrain", "true" if decision["retrain"] else "false")
    lh.set_task_value("clock", str(win["clock"].date()))


if __name__ == "__main__":
    main()
