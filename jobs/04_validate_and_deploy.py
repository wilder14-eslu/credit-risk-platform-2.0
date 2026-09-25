"""Job 4 - Validación offline champion vs challenger + despliegue en Model Serving + permisos de las apps.

- Sin champion: el challenger se promueve directo (primer despliegue).
- Con champion: ambos se evalúan en el periodo de test OOT del challenger (el más reciente).
  Si el challenger mejora el AUC en `min_auc_improvement`, pasa al A/B test online.
- Actualiza el endpoint de Databricks Model Serving (champion + challenger con split de tráfico)
  y da acceso a los service principals de las Databricks Apps (API y dashboard).
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

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import (  # noqa: E402
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    date_column,
    input_features,
    target_name,
)
from credit_risk.models import metrics as M  # noqa: E402
from credit_risk.models import training as T  # noqa: E402
from credit_risk.registry import mlflow_registry as R  # noqa: E402
from credit_risk.registry import serving_endpoint as S  # noqa: E402


def main() -> None:
    lh.configure_logging()
    args = lh.job_args({"deploy-serving": "true"})
    names = lh.names_from(args)
    R.setup_mlflow()

    challenger_v = R.get_alias_version(names, CHALLENGER_ALIAS)
    champion_v = R.get_alias_version(names, CHAMPION_ALIAS)
    if challenger_v is None and champion_v is None:
        raise RuntimeError("No hay modelos registrados: corre primero el pipeline de entrenamiento")

    if challenger_v and champion_v is None:
        R.promote_challenger(names)
        champion_v, challenger_v = challenger_v, None
        lh.logger.info("Primer despliegue: v%s es champion", champion_v)
    elif challenger_v and champion_v:
        tags = R.version_tags(names, challenger_v)
        missing = [k for k in ("validation_end", "test_end") if k not in tags]
        if missing:  # sin cortes no hay evaluación justa: evaluar sobre datos de entrenamiento sería leakage
            raise RuntimeError(f"La versión v{challenger_v} no tiene los tags {missing}")
        cfg = {k: tags[k] for k in ("validation_end", "test_end")}
        cols = ", ".join([*input_features(), date_column(), target_name()])
        data = lh.read_pandas(
            f"SELECT {cols} FROM {names.feature_table} WHERE {target_name()} IS NOT NULL "
            f"AND {date_column()} >= '{cfg['validation_end']}' AND {date_column()} < '{cfg['test_end']}'"
        )
        x, y = data[list(input_features())], data[target_name()].astype(int)
        auc = {
            a: M.classification_metrics(y, R.load_credit_model(names, a).predict_proba(x))["roc_auc"]
            for a in (CHAMPION_ALIAS, CHALLENGER_ALIAS)
        }
        ok, reason = T.champion_vs_challenger(auc[CHALLENGER_ALIAS], auc[CHAMPION_ALIAS])
        lh.logger.info("Validación offline: %s (%s) %s", ok, reason, auc)
        if not ok:
            R.delete_alias(names, CHALLENGER_ALIAS)
            challenger_v = None
        lh.set_task_value("offline_validation", reason)

    if args.deploy_serving.lower() == "true":
        lh.logger.info("Endpoint: %s", S.deploy(names, champion_v, challenger_v))
        principals = S.grant_query_to_apps(S.endpoint_for(names))
        lh.grant_app_access(names, principals)
    lh.set_task_value("champion_version", champion_v)
    lh.set_task_value("challenger_version", challenger_v or "")


if __name__ == "__main__":
    main()
