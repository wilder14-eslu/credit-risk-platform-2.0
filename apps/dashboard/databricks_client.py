"""Acceso a la API de Databricks con la identidad de la app (service principal).

Dentro de Databricks Apps, `WorkspaceClient()` se autentica solo con las
credenciales OAuth que la plataforma inyecta; no hay tokens en el código.
"""

from __future__ import annotations

from typing import Any


class DatabricksClient:
    def __init__(self, workspace_client=None) -> None:
        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()
        self.w = workspace_client

    def invoke(self, endpoint: str, served_model: str, records: list[dict], params: dict | None = None) -> list[dict]:
        """Consulta una variante concreta (champion / challenger) del endpoint de Model Serving."""
        body: dict[str, Any] = {"dataframe_records": records}
        if params:
            body["params"] = params
        response = self.w.api_client.do(
            "POST", f"/serving-endpoints/{endpoint}/served-models/{served_model}/invocations", body=body
        )
        predictions = response.get("predictions", response)
        if isinstance(predictions, dict):  # formato columnar
            keys = list(predictions)
            predictions = [dict(zip(keys, vals, strict=False)) for vals in zip(*predictions.values(), strict=False)]
        return predictions

    def served_versions(self, endpoint: str) -> dict[str, str]:
        config = self.w.serving_endpoints.get(endpoint).config
        return {e.name: str(e.entity_version) for e in (config.served_entities or [])}

    def sql(self, warehouse_id: str, statement: str) -> list[dict]:
        from databricks.sdk.service.sql import StatementState

        resp = self.w.statement_execution.execute_statement(
            statement=statement, warehouse_id=warehouse_id, wait_timeout="30s"
        )
        if resp.status and resp.status.state == StatementState.FAILED:
            raise RuntimeError(resp.status.error.message if resp.status.error else "SQL failed")
        if not resp.manifest or not resp.result:
            return []
        cols = [c.name for c in resp.manifest.schema.columns]
        return [dict(zip(cols, row, strict=False)) for row in (resp.result.data_array or [])]
