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
