"""Despliegue continuo del modelo en Databricks Model Serving (API de Databricks).

El endpoint expone dos served entities, `champion` y `challenger`, con un
split de tráfico. El gateway de Render consulta cada una por separado
(`/serving-endpoints/{name}/served-models/{variant}/invocations`) para hacer
una asignación A/B determinista y registrar qué variante respondió.
"""

from __future__ import annotations

import logging
from typing import Any

from credit_risk.config import UCNames, platform_config

logger = logging.getLogger(__name__)


def endpoint_for(names: UCNames) -> str:
    """Un endpoint por entorno: dev (`*_dev`) no pisa al de producción."""
    base = platform_config()["serving"]["endpoint_name"]
    return f"{base}-dev" if names.schema.endswith("_dev") else base


def build_config(
    names: UCNames,
    champion_version: str,
    challenger_version: str | None,
    challenger_traffic: float,
    workload_size: str = "Small",
    scale_to_zero: bool = True,
) -> dict[str, Any]:
    """Configuración declarativa (dict) del endpoint; se prueba sin SDK."""
    entities = [
        {
            "name": "champion",
            "entity_name": names.model_name,
            "entity_version": champion_version,
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero,
        }
    ]
    routes = [{"served_model_name": "champion", "traffic_percentage": 100}]
    if challenger_version and challenger_version != champion_version and challenger_traffic > 0:
        pct = int(round(challenger_traffic * 100))
        entities.append({**entities[0], "name": "challenger", "entity_version": challenger_version})
        routes = [
            {"served_model_name": "champion", "traffic_percentage": 100 - pct},
            {"served_model_name": "challenger", "traffic_percentage": pct},
        ]
    return {"served_entities": entities, "traffic_config": {"routes": routes}}


def deploy(
    names: UCNames,
    champion_version: str,
    challenger_version: str | None = None,
    challenger_traffic: float | None = None,
    endpoint_name: str | None = None,
    wait: bool = True,
) -> dict[str, Any]:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        Route,
        ServedEntityInput,
        TrafficConfig,
    )

    scfg = platform_config()["serving"]
    endpoint_name = endpoint_name or endpoint_for(names)
    challenger_traffic = (
        platform_config()["ab_testing"]["challenger_traffic"] if challenger_traffic is None else challenger_traffic
    )
    spec = build_config(
        names,
        champion_version,
        challenger_version,
        challenger_traffic,
        scfg["workload_size"],
        scfg["scale_to_zero"],
    )
    entities = [ServedEntityInput(**e) for e in spec["served_entities"]]
    traffic = TrafficConfig(routes=[Route(**r) for r in spec["traffic_config"]["routes"]])

    w = WorkspaceClient()
    existing = {e.name for e in w.serving_endpoints.list()}
    if endpoint_name in existing:
        logger.info("Actualizando endpoint %s", endpoint_name)
        call = w.serving_endpoints.update_config_and_wait if wait else w.serving_endpoints.update_config
        call(name=endpoint_name, served_entities=entities, traffic_config=traffic)
    else:
        logger.info("Creando endpoint %s", endpoint_name)
        config = EndpointCoreConfigInput(name=endpoint_name, served_entities=entities, traffic_config=traffic)
        call = w.serving_endpoints.create_and_wait if wait else w.serving_endpoints.create
        call(name=endpoint_name, config=config)
    return {"endpoint": endpoint_name, **spec}


def app_service_principals(app_names: list[str] | None = None) -> dict[str, str]:
    """client_id del service principal de cada Databricks App (las que existan)."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    found = {}
    for name in app_names or platform_config()["serving"]["app_names"]:
        try:
            found[name] = w.apps.get(name).service_principal_client_id
        except Exception as exc:  # la app todavía no se desplegó
            logger.warning("App %s no encontrada (%s): se omiten sus permisos", name, exc)
    return {k: v for k, v in found.items() if v}


def grant_query_to_apps(endpoint_name: str, app_names: list[str] | None = None) -> list[str]:
    """Da CAN_QUERY sobre el endpoint a los service principals de las apps (API y dashboard)."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        ServingEndpointAccessControlRequest,
        ServingEndpointPermissionLevel,
    )

    principals = app_service_principals(app_names)
    if not principals:
        return []
    w = WorkspaceClient()
    endpoint_id = w.serving_endpoints.get(endpoint_name).id
    w.serving_endpoints.update_permissions(
        serving_endpoint_id=endpoint_id,
        access_control_list=[
            ServingEndpointAccessControlRequest(
                service_principal_name=client_id,
                permission_level=ServingEndpointPermissionLevel.CAN_QUERY,
            )
            for client_id in principals.values()
        ],
    )
    logger.info("CAN_QUERY otorgado a: %s", ", ".join(principals))
    return list(principals.values())
