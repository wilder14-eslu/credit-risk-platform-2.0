"""Contratos de la infraestructura como código (Databricks Asset Bundle, Apps y Model Serving)."""

import pytest
import yaml

from credit_risk.config import PROJECT_ROOT, UCNames
from credit_risk.registry.serving_endpoint import build_config, endpoint_for

BUNDLE = yaml.safe_load((PROJECT_ROOT / "databricks.yml").read_text(encoding="utf-8"))
JOBS = yaml.safe_load((PROJECT_ROOT / "resources" / "jobs.yml").read_text(encoding="utf-8"))["resources"]["jobs"]


def test_targets():
    assert BUNDLE["targets"]["dev"]["mode"] == "development"
    assert BUNDLE["targets"]["prod"]["mode"] == "production"
    assert BUNDLE["targets"]["staging"]["mode"] == "production"


def test_staging_is_isolated_and_cheap():
    """Staging comparte el workspace con prod: esquema propio, schedules pausados, sin Serving."""
    staging = BUNDLE["targets"]["staging"]
    assert staging["variables"]["schema"] not in (
        BUNDLE["targets"]["prod"]["variables"]["schema"],
        BUNDLE["targets"]["dev"]["variables"]["schema"],
    )
    assert staging["presets"]["trigger_pause_status"] == "PAUSED"
    assert staging["presets"]["name_prefix"]
    assert staging["variables"]["deploy_serving"] == "false"
    assert staging["variables"]["data_source"] == "synthetic"
    assert "resources" not in staging  # las Apps solo viven en prod


def test_cd_promotes_to_prod_only_through_staging():
    cd = yaml.safe_load((PROJECT_ROOT / ".github" / "workflows" / "cd.yml").read_text(encoding="utf-8"))["jobs"]
    assert "integration-staging" in cd["deploy-prod"]["needs"]
    assert cd["integration-staging"]["needs"] == "deploy-dev"
    steps = " ".join(str(step.get("run", "")) for step in cd["integration-staging"]["steps"])
    assert "bundle run -t staging ct_training_pipeline" in steps
    assert "bundle run -t staging production_monitoring" in steps
    assert "smoke_check.py" in steps


@pytest.mark.parametrize("job_key", list(JOBS))
def test_tasks_are_serverless_and_files_exist(job_key):
    job = JOBS[job_key]
    envs = {e["environment_key"] for e in job.get("environments", [])}
    keys = {t["task_key"] for t in job["tasks"]}
    for task in job["tasks"]:
        for dep in task.get("depends_on", []):
            assert dep["task_key"] in keys
        if "spark_python_task" in task:
            assert (PROJECT_ROOT / "resources" / task["spark_python_task"]["python_file"]).resolve().exists()
            assert task["environment_key"] in envs


# Tareas que hacen append a tablas de log o cambian el estado del registro: reintentar duplicaría filas o
# intercambiaría champion/previous otra vez. Solo se reintenta lo idempotente (overwrite o API declarativa).
NON_IDEMPOTENT = {"train_and_register", "replay_production", "monitor_drift", "ab_evaluate", "rollback"}


@pytest.mark.parametrize("job_key", list(JOBS))
def test_jobs_have_reliability_policy(job_key):
    job = JOBS[job_key]
    assert job["timeout_seconds"] > 0
    assert job["email_notifications"]["on_failure"]
    assert job["health"]["rules"][0]["metric"] == "RUN_DURATION_SECONDS"
    if job.get("max_concurrent_runs") == 1:
        assert job["queue"]["enabled"] is True
    for task in job["tasks"]:
        if "spark_python_task" not in task:
            continue
        assert task["timeout_seconds"] > 0, task["task_key"]
        retries = task["max_retries"]
        if task["task_key"] in NON_IDEMPOTENT:
            assert retries == 0, f"{task['task_key']} no es idempotente: no debe reintentar"
        else:
            assert 0 <= retries <= 2


def test_no_hardcoded_emails_in_bundle():
    text = (PROJECT_ROOT / "resources" / "jobs.yml").read_text(encoding="utf-8")
    assert "@gmail.com" not in text and "@outlook" not in text
    assert "${workspace.current_user.userName}" in text


def test_no_classic_clusters():
    text = (PROJECT_ROOT / "resources" / "jobs.yml").read_text(encoding="utf-8")
    assert "new_cluster" not in text and "existing_cluster_id" not in text


def test_monitoring_triggers_ct_with_clock():
    tasks = {t["task_key"]: t for t in JOBS["production_monitoring"]["tasks"]}
    assert "monitor_drift.values.retrain" in tasks["needs_retraining"]["condition_task"]["left"]
    run = tasks["trigger_continuous_training"]["run_job_task"]
    assert run["job_id"] == "${resources.jobs.ct_training_pipeline.id}"
    assert "monitor_drift.values.clock" in run["job_parameters"]["as_of"]


def test_apps_defined_with_warehouse_and_sources():
    apps = BUNDLE["targets"]["prod"]["resources"]["apps"]
    for app in apps.values():
        assert (PROJECT_ROOT / app["source_code_path"] / "app.yaml").exists()
        assert app["resources"][0]["sql_warehouse"]["permission"] == "CAN_USE"
        manifest = yaml.safe_load((PROJECT_ROOT / app["source_code_path"] / "app.yaml").read_text(encoding="utf-8"))
        assert any(e.get("valueFrom") == "sql-warehouse" for e in manifest["env"])


def test_serving_config():
    names = UCNames(catalog="workspace", schema="credit_risk")
    solo = build_config(names, "3", None, 0.2)
    assert [e["name"] for e in solo["served_entities"]] == ["champion"]
    ab = build_config(names, "3", "4", 0.2)
    routes = {r["served_model_name"]: r["traffic_percentage"] for r in ab["traffic_config"]["routes"]}
    assert routes == {"champion": 80, "challenger": 20}
    assert ab["served_entities"][0]["entity_name"] == "workspace.credit_risk.credit_default_model"
    assert endpoint_for(UCNames("workspace", "credit_risk_dev")).endswith("-dev")
    assert not endpoint_for(names).endswith("-dev")
    assert endpoint_for(UCNames("workspace", "credit_risk_staging")).endswith("-staging")
    # los tres entornos nunca comparten endpoint
    schemas = ("credit_risk", "credit_risk_dev", "credit_risk_staging")
    assert len({endpoint_for(UCNames("workspace", sc)) for sc in schemas}) == 3
