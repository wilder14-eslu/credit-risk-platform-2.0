"""Smoke check post-despliegue: el modelo vivo en Unity Catalog debe ser trazable.

Lo usa CD en el gate de staging (`.github/workflows/cd.yml`) contra el workspace real:

    python tests/e2e/smoke_check.py --model workspace.credit_risk_staging.credit_default_model

Falla (exit 1) si @champion no existe o si a su versión le falta trazabilidad de código/datos.
Requiere `mlflow-skinny` y DATABRICKS_HOST / DATABRICKS_TOKEN en el entorno.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from credit_risk.registry.lineage import missing_lineage  # noqa: E402


def check(model: str, alias: str) -> list[str]:
    import mlflow

    client = mlflow.MlflowClient(registry_uri="databricks-uc")
    try:
        version = client.get_model_version_by_alias(model, alias)
    except Exception as exc:  # alias o modelo inexistente
        return [f"@{alias} de {model} no existe: {exc}"]
    problems = missing_lineage(version.tags)
    print(f"{model} @{alias} -> v{version.version} | tags: {sorted((version.tags or {}).keys())}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="catalog.schema.modelo en Unity Catalog")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()
    problems = check(args.model, args.alias)
    for p in problems:
        print(f"::error::{p}")
    print("SMOKE OK" if not problems else f"SMOKE FALLÓ ({len(problems)} problemas)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
