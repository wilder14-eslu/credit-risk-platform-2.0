"""Configuración central: esquema de datos, política de MLOps y nombres en Unity Catalog."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.getenv("CREDIT_RISK_CONFIG_DIR", PROJECT_ROOT / "config"))

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
PREVIOUS_CHAMPION_ALIAS = "previous_champion"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "/Shared/credit-risk-platform-2.0")


def _load_yaml(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def data_schema() -> dict[str, Any]:
    return _load_yaml("data_schema.yaml")


@lru_cache(maxsize=1)
def platform_config() -> dict[str, Any]:
    return _load_yaml("platform.yaml")


def numeric_features() -> dict[str, dict[str, Any]]:
    return data_schema()["numeric_features"]


def categorical_features() -> dict[str, dict[str, Any]]:
    return data_schema()["categorical_features"]


def input_features() -> tuple[str, ...]:
    """Variables que recibe el modelo (y la API) antes del feature engineering."""
    return tuple(numeric_features()) + tuple(categorical_features())


def target_name() -> str:
    return data_schema()["target"]


def date_column() -> str:
    return data_schema()["date_column"]


def id_column() -> str:
    return data_schema()["id_column"]


@dataclass(frozen=True)
class UCNames:
    """Nombres completos (catalog.schema.objeto) en Unity Catalog."""

    catalog: str = field(default_factory=lambda: os.getenv("UC_CATALOG", "workspace"))
    schema: str = field(default_factory=lambda: os.getenv("UC_SCHEMA", "credit_risk"))

    def fq(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}"

    # Medallion
    bronze = property(lambda self: self.fq("bronze_loans"))
    silver = property(lambda self: self.fq("silver_loans"))
    data_quality_log = property(lambda self: self.fq("data_quality_log"))
    feature_table = property(lambda self: self.fq("gold_loan_features"))
    # Producción y feedback loop
    inference_log = property(lambda self: self.fq("inference_log"))
    outcomes = property(lambda self: self.fq("outcomes"))
    reference_profile = property(lambda self: self.fq("reference_profile"))
    # Monitoreo y experimentación
    monitoring_metrics = property(lambda self: self.fq("monitoring_metrics"))
    drift_features = property(lambda self: self.fq("drift_by_feature"))
    ab_results = property(lambda self: self.fq("ab_test_results"))
    model_benchmark = property(lambda self: self.fq("model_benchmark"))
    retrain_events = property(lambda self: self.fq("retrain_events"))
    # Modelo
    model_name = property(lambda self: self.fq("credit_default_model"))
