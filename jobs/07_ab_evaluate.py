"""Job 7 - Evaluación del A/B test online (con desenlaces reales ya observados) y promoción.

Decide `promover | detener | continuar`:
- promover: challenger -> @champion (el anterior queda en @previous_champion) y el endpoint pasa a 100%.
- detener: se elimina @challenger y el endpoint vuelve a 100% champion.
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

import pandas as pd  # noqa: E402

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import CHALLENGER_ALIAS, CHAMPION_ALIAS, platform_config  # noqa: E402
from credit_risk.monitoring.ab_testing import evaluate_ab_test  # noqa: E402
from credit_risk.monitoring.replay import months_between  # noqa: E402
from credit_risk.registry import mlflow_registry as R  # noqa: E402
from credit_risk.registry import serving_endpoint as S  # noqa: E402

RESULT_COLS = (
    "decision",
    "reason",
    "n_champion",
    "n_challenger",
    "auc_champion",
    "auc_challenger",
    "auc_diff",
    "ci_low",
    "ci_high",
    "p_value",
    "bad_rate_approved_champion",
    "bad_rate_approved_challenger",
    "bad_rate_pvalue",
    "months_running",
)


def main() -> None:
    lh.configure_logging()
    args = lh.job_args({"deploy-serving": "true"})
    names = lh.names_from(args)
    cfg = platform_config()["ab_testing"]
    R.setup_mlflow()

    champion_v = R.get_alias_version(names, CHAMPION_ALIAS)
    challenger_v = R.get_alias_version(names, CHALLENGER_ALIAS)
    if not champion_v or not challenger_v:
        lh.logger.info("No hay A/B test activo")
        lh.set_task_value("ab_decision", "sin_test")
        return

    clock = lh.read_pandas(f"SELECT max(event_ts) AS m FROM {names.inference_log} WHERE source = 'replay'")["m"].iloc[0]
    # Inicio del test = primera solicitud servida por el challenger (con o sin desenlace aún).
    # Ambos brazos se comparan solo desde esa fecha: mismo periodo, misma población.
    started = lh.read_pandas(
        f"SELECT min(event_ts) AS m FROM {names.inference_log} "
        f"WHERE variant = 'challenger' AND model_version = '{challenger_v}'"
    )["m"].iloc[0]
    since = f"AND l.event_ts >= '{pd.Timestamp(started)}'" if pd.notna(started) else ""
    data = lh.read_pandas(f"""
        SELECT l.variant, l.probability, l.decision, l.event_ts, o.actual_default
        FROM {names.inference_log} l JOIN {names.outcomes} o ON l.request_id = o.request_id
        WHERE o.observed_ts <= '{pd.Timestamp(clock).date()}'
          AND ((l.variant = 'champion' AND l.model_version = '{champion_v}')
            OR (l.variant = 'challenger' AND l.model_version = '{challenger_v}'))
          {since}
    """)
    arms = {
        v: {
            "y": g["actual_default"].astype(int).to_numpy(),
            "p": g["probability"].to_numpy(),
            "approved": (g["decision"] == "APROBAR").to_numpy(),
        }
        for v, g in ((v, data[data["variant"] == v]) for v in ("champion", "challenger"))
    }
    running = months_between(started, clock) if pd.notna(started) else 0
    result = evaluate_ab_test(arms["champion"], arms["challenger"], cfg, months_running=running)
    lh.logger.info("Resultado A/B: %s", result)

    row = {
        "run_ts": lh.utcnow(),
        "clock_month": pd.Timestamp(clock),
        "champion_version": champion_v,
        "challenger_version": challenger_v,
        **{k: result.get(k) for k in RESULT_COLS},
    }
    lh.write_pandas(pd.DataFrame([row]), names.ab_results)

    deploy = args.deploy_serving.lower() == "true"
    if result["decision"] == "promover":
        R.promote_challenger(names)
        if deploy:
            S.deploy(names, challenger_v, None)
    elif result["decision"] == "detener":
        R.delete_alias(names, CHALLENGER_ALIAS)
        if deploy:
            S.deploy(names, champion_v, None)
    lh.set_task_value("ab_decision", result["decision"])


if __name__ == "__main__":
    main()
