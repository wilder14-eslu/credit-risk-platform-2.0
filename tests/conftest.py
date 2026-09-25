"""Fixtures: datos sintéticos con el formato EXACTO de Lending Club (el CSV real no se versiona)."""

from __future__ import annotations

import os
import tempfile

import pytest

from credit_risk.data.lending_club import parse_raw
from credit_risk.data.quality import clean
from credit_risk.data.synthetic import generate_raw
from credit_risk.models.training import fit_model, make_splits

# MLflow local fuera del repo (evita crear ./mlruns al correr los tests)
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:" + tempfile.mkdtemp(prefix="mlruns-"))


@pytest.fixture(scope="session")
def raw_lc():
    return generate_raw(40_000, seed=7)


@pytest.fixture(scope="session")
def canonical(raw_lc):
    return clean(parse_raw(raw_lc))


@pytest.fixture(scope="session")
def splits(canonical):
    return make_splits(canonical)


@pytest.fixture(scope="session")
def lr_model(splits):
    return fit_model("logistic_regression", {}, splits)


@pytest.fixture(scope="session")
def xgb_model(splits):
    return fit_model("xgboost", {"n_estimators": 150}, splits)
