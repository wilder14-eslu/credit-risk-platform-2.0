"""Job 3 - Benchmark + tuning (Optuna) + registro en Unity Catalog como @challenger.

1. Split temporal out-of-time (o ventanas desplazadas con `--as-of` si lo disparó el monitoreo).
2. Benchmark de 4 algoritmos sobre el mismo split (corridas anidadas en MLflow).
3. Tuning con Optuna del ganador (penaliza sobreajuste); si no pasa los gates, fallback
   al mejor candidato del benchmark con parámetros por defecto que sí los pase.
4. Gate absoluto de calidad; si falla, no se registra nada.
5. Registro del modelo pyfunc en UC con alias @challenger + perfil de referencia para drift.
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

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import CHALLENGER_ALIAS, date_column, input_features, platform_config, target_name  # noqa: E402
from credit_risk.models import training as T  # noqa: E402
from credit_risk.monitoring.data_drift import build_reference_profile  # noqa: E402
from credit_risk.registry import mlflow_registry as R  # noqa: E402


def write_selection_tables(names, selection: dict, comparison: dict, run_ts, version: str | None) -> None:
    """Persiste la evidencia estadística: selección (validación) y reporte (test)."""
    tests = {r["model"]: r for r in comparison["auc"]}
    rows = [
        {
            "split": "validation",
            "model": r["model"],
            "auc": r["auc"],
            "delta_vs_best": r["delta_vs_best"],
            "p_holm": r["p_holm"],
            "complexity": r["complexity"],
            "eligible": r["eligible"],
            "retained": r["retained"],
            "selected": r["selected"],
        }
        for r in selection["rows"]
    ]
    rows += [
        {
            "split": "test",
            "model": name,
            "auc": r["auc"],
            "auc_ci_low": r["ci_low"],
            "auc_ci_high": r["ci_high"],
            "selected": name == selection["selected"],
        }
        for name, r in tests.items()
    ]
    evaluation = pd.DataFrame(rows).assign(run_ts=run_ts, registered_version=version)
    lh.write_pandas(evaluation, names.model_evaluation)
    pairs = pd.DataFrame(comparison["pairwise"]).assign(run_ts=run_ts, split="test", registered_version=version)
    lh.write_pandas(pairs, names.model_comparison)


def log_statistical_report(
    run_id, splits, table, models, selection, comparison, scorecard, best, model, result, origin, data, train_end
):
    """Registra en MLflow el reporte estadístico completo (cifras + figuras) de este entrenamiento.

    Es el mismo análisis que `scripts/build_report.py` vuelca al README. Nunca hace
    fallar el entrenamiento: si algo falla, se registra la advertencia y se sigue.
    """
    if scorecard is None:
        return
    try:
        import tempfile
        from pathlib import Path

        from credit_risk.reporting import figures
        from credit_risk.reporting.pipeline import assemble_results, json_safe

        dates = pd.to_datetime(data[date_column()])
        later = data[dates >= pd.Timestamp(train_end)]
        res = json_safe(
            assemble_results(
                splits, table, models, selection, comparison, scorecard, best, model, result, origin, later
            )
        )
        with mlflow.start_run(run_id=run_id), tempfile.TemporaryDirectory() as tmp:
            figures.render_all(res, Path(tmp))
            mlflow.log_artifacts(tmp, "report/figures")
            mlflow.log_dict(res, "report/results.json")
    except Exception as exc:  # el reporte es informativo: no bloquea el registro del modelo
        lh.logger.warning("No se pudo generar el reporte estadístico: %s", exc)


def main() -> None:
    lh.configure_logging()
    args = lh.job_args(
        {"optuna-trials": None, "trigger": "manual", "git-sha": "local", "as-of": "", "data-source": "real"}
    )
    names = lh.names_from(args)
    tcfg = platform_config()["training"]
    trials = int(args.optuna_trials) if args.optuna_trials not in (None, "") else tcfg["optuna_trials"]
    split_cfg = T.split_config(args.as_of or None)
    # Lineage de datos: versión Delta del feature table con la que se entrena (reproducible con time travel).
    gold_version = str(lh.table_version(names.feature_table) or "unknown")
    lh.logger.info("Feature table %s en versión Delta %s", names.feature_table, gold_version)

    R.setup_mlflow()
    cols = ", ".join([*input_features(), date_column(), target_name()])
    data = lh.read_pandas(
        f"SELECT {cols} FROM {names.feature_table} "
        f"WHERE {target_name()} IS NOT NULL AND {date_column()} < '{split_cfg['test_end']}'"
    )
    splits = T.make_splits(data, split_cfg)
    lh.logger.info("Periodos: %s", splits.periods)

    with mlflow.start_run(run_name=f"retraining-{args.trigger}") as parent:
        mlflow.set_tags(
            {
                "trigger": args.trigger,
                "git_sha": args.git_sha,
                "stage": "pipeline",
                "as_of": args.as_of or "config",
                "gold_table": names.feature_table,
                "gold_delta_version": gold_version,
            }
        )
        mlflow.log_params({f"period_{k}": v for k, v in splits.periods.items()})
        table, models = T.run_benchmark(splits)
        for _, row in table.iterrows():
            R.log_candidate(row["algorithm"], row.to_dict(), parent_run_id=parent.info.run_id)
        # Selección en VALIDACIÓN: DeLong + Holm + margen práctico + parsimonia.
        selection = T.statistical_selection(table, models, splits)
        # Reporte en TEST (una sola vez): IC de DeLong y comparaciones pareadas, con el scorecard WoE.
        scorecard = T.fit_scorecard(splits) if tcfg.get("scorecard_benchmark", True) else None
        comparison = T.compare_on_test(models, splits, scorecard)
        iv = scorecard.iv_table() if scorecard is not None else None
        mlflow.log_dict(selection, "selection/validation_selection.json")
        mlflow.log_dict(comparison, "selection/test_comparison.json")
        if iv:
            mlflow.log_dict({"iv": iv}, "selection/scorecard_iv.json")
        mlflow.set_tags({"selection_rule": "delong_holm_parsimony", "selected_by_rule": selection["selected"]})
        best, model, result, origin = T.choose_final_model(
            table, models, splits, trials, tcfg["optuna_timeout_seconds"], best=selection["selected"]
        )
        mlflow.set_tags({"final_algorithm": best, "final_model_origin": origin})
        lh.logger.info("Modelo final: %s (%s)", best, origin)
        mlflow.log_metrics({f"best_{k}": v for k, v in result.items() if isinstance(v, float)})

    run_ts = lh.utcnow()
    bench = table.drop(columns=["params"]).copy()
    bench["run_ts"], bench["selected"] = run_ts, bench["algorithm"] == best
    passed, reasons = T.check_quality_gates(result)
    if not passed:
        bench["registered_version"] = None
        lh.write_pandas(bench, names.model_benchmark)
        write_selection_tables(names, selection, comparison, run_ts, None)
        lh.set_task_value("gate_passed", "false")
        raise RuntimeError(f"Quality gate fallido, no se registra el modelo: {reasons}")

    reference = {k.replace("test_", ""): v for k, v in result.items() if k.startswith("test_")}
    model.metadata.update(reference_metrics=reference, split_config=split_cfg)
    importance = model.global_importance(splits.x_test.sample(min(2000, len(splits.x_test)), random_state=0))
    report = T.training_report(
        table, best, result, importance, splits.periods, selection=selection, comparison=comparison, iv=iv
    )

    version = R.log_and_register(
        model,
        result,
        report,
        table,
        splits.x_test.head(5).reset_index(drop=True),
        names,
        extra_tags={
            "trigger": args.trigger,
            "git_sha": args.git_sha,
            "data_source": args.data_source,
            "gold_table": names.feature_table,
            "gold_delta_version": gold_version,
            "train_end": split_cfg["train_end"],
            "validation_end": split_cfg["validation_end"],
            "test_end": split_cfg["test_end"],
        },
    )
    R.set_alias(names, CHALLENGER_ALIAS, version)
    bench["registered_version"] = bench["selected"].map(lambda s: version if s else None)
    lh.write_pandas(bench, names.model_benchmark)
    write_selection_tables(names, selection, comparison, run_ts, version)

    profile = build_reference_profile(splits.x_train, model.predict_proba(splits.x_train))
    lh.write_pandas(
        pd.DataFrame(
            [
                {
                    "model_version": version,
                    "created_ts": lh.utcnow(),
                    "profile_json": json.dumps(profile),
                    "reference_metrics_json": json.dumps(reference),
                }
            ]
        ),
        names.reference_profile,
    )
    os.makedirs(f"{names.volume_path}/artifacts/reports", exist_ok=True)
    with open(f"{names.volume_path}/artifacts/reports/training_report_v{version}.md", "w", encoding="utf-8") as fh:
        fh.write(report)

    log_statistical_report(
        parent.info.run_id,
        splits,
        table,
        models,
        selection,
        comparison,
        scorecard,
        best,
        model,
        result,
        origin,
        data,
        split_cfg["train_end"],
    )
    lh.set_task_value("gate_passed", "true")
    lh.set_task_value("challenger_version", version)


if __name__ == "__main__":
    main()
