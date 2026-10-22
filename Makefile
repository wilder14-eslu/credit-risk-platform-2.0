.PHONY: install lint format test cov e2e validate deploy-dev deploy-prod train-dev replay-dev apps

install:
	pip install -r requirements-dev.txt

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing --cov-report=xml

e2e:
	python tests/e2e/run_pipeline_locally.py

validate:
	databricks bundle validate -t dev

deploy-dev:
	databricks bundle deploy -t dev

deploy-prod:
	databricks bundle deploy -t prod --var="warehouse_id=$(DATABRICKS_WAREHOUSE_ID)"

train-dev:
	databricks bundle run -t dev ct_training_pipeline

replay-dev:
	databricks bundle run -t dev production_monitoring --params months=6

apps:
	databricks bundle run -t prod credit_risk_api && databricks bundle run -t prod credit_risk_dashboard
