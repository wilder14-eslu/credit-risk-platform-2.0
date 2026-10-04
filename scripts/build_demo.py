"""Arma la demo pública (Streamlit Community Cloud) con el modelo champion de Unity Catalog.

La demo es autocontenida (no llama a Databricks): lleva la app Streamlit, el paquete
`credit_risk`, la configuración YAML, el modelo exportado, `reports/results.json` y
las figuras del análisis. Antes de terminar corre una predicción DENTRO de la carpeta
generada, sin acceso al repo, para garantizar que la app arranca.

Uso:

    # champion de producción (requiere credenciales de Databricks: perfil DEFAULT o DATABRICKS_HOST/TOKEN)
    python scripts/build_demo.py
    # un modelo ya exportado (joblib de CreditRiskModel)
    python scripts/build_demo.py --model ruta/credit_model.joblib

El workflow `.github/workflows/demo.yml` publica la carpeta en la rama `demo`, de donde
la despliega Streamlit Community Cloud (archivo principal `app.py`, Python 3.12).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_SRC = ROOT / "apps" / "demo"
sys.path.insert(0, str(ROOT / "src"))  # para deserializar CreditRiskModel
logger = logging.getLogger("build_demo")

BASE_REQUIREMENTS = ["streamlit", "plotly", "pandas", "numpy", "scipy", "scikit-learn", "pyyaml", "joblib"]
ALGORITHM_REQUIREMENTS = {"catboost": ["catboost"], "xgboost": ["xgboost"], "lightgbm": ["lightgbm"]}
SMOKE = (
    "import demo_core as D; "
    "r = D.score(D.build_record(**dict(zip(D.FORM_FIELDS, D.EXAMPLES['Perfil típico'])))); "
    "assert 0 < r['probability'] < 1 and r['factors'], r; print(round(r['probability'], 4), r['decision'])"
)


def download_champion(alias: str, catalog: str, schema: str, dest: Path) -> dict:
    """Descarga el artefacto `credit_model.joblib` de la versión con el alias dado."""
    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    name = f"{catalog}.{schema}.credit_default_model"
    mv = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{mv.version}", dst_path=tmp))
        found = next(local.rglob("credit_model.joblib"), None)
        if found is None:
            raise FileNotFoundError(f"La versión {mv.version} de {name} no tiene credit_model.joblib")
        shutil.copy2(found, dest)
    tags = dict(mv.tags or {})
    return {
        "source": f"Unity Catalog {name}@{alias}",
        "version": str(mv.version),
        "test_roc_auc": tags.get("test_roc_auc"),
        "git_sha": tags.get("git_sha"),
    }


def pinned_requirements(algorithm: str) -> list[str]:
    """Fija las versiones del entorno donde se verificó el modelo: lo que carga aquí, carga en la demo."""
    lines = []
    for pkg in BASE_REQUIREMENTS + ALGORITHM_REQUIREMENTS.get(algorithm, []):
        try:
            lines.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            lines.append(pkg)
    return lines


STREAMLIT_CONFIG = """[theme]
primaryColor = "#2a78d6"

[server]
headless = true

[browser]
gatherUsageStats = false
"""


def demo_readme(info: dict) -> str:
    version_txt = f" v{info['version']}" if info.get("version") else ""
    return f"""# Credit Risk Platform 2.0 · demo pública

Rama generada automáticamente por `scripts/build_demo.py` (workflow `Demo pública`); no editar a mano.
El código fuente está en la rama `main`.

- Modelo: `{info["algorithm"]}`{version_txt} ({info["source"]}), umbral {info["threshold"]:.2f}
- Exportado: {info["exported_at"]}
- Despliegue: Streamlit Community Cloud, archivo principal `app.py`, Python 3.12

> Demo educativa con datos históricos públicos. No es una herramienta de decisión crediticia real.
"""


def build(out: Path, model_path: Path | None, alias: str, catalog: str, schema: str) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    (out / ".streamlit").mkdir()
    (out / ".streamlit" / "config.toml").write_text(STREAMLIT_CONFIG, encoding="utf-8")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")

    for name in ("app.py", "demo_core.py"):
        shutil.copy2(DEMO_SRC / name, out / name)
    shutil.copytree(ROOT / "src" / "credit_risk", out / "credit_risk", ignore=ignore)
    shutil.copytree(ROOT / "config", out / "config", ignore=ignore)
    if (ROOT / "reports" / "results.json").exists():
        (out / "results").mkdir()
        shutil.copy2(ROOT / "reports" / "results.json", out / "results" / "results.json")
    if (ROOT / "docs" / "figures").exists():
        shutil.copytree(ROOT / "docs" / "figures", out / "figures", ignore=ignore)

    target = out / "model" / "credit_model.joblib"
    if model_path:
        shutil.copy2(model_path, target)
        info = {"source": f"archivo local {model_path.name}", "version": None}
    else:
        info = download_champion(alias, catalog, schema, target)

    import joblib

    model = joblib.load(target)
    info.update(
        algorithm=model.algorithm,
        threshold=float(model.threshold),
        exported_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    (out / "model" / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "requirements.txt").write_text("\n".join(pinned_requirements(model.algorithm)) + "\n", encoding="utf-8")

    (out / "README.md").write_text(demo_readme(info), encoding="utf-8")

    # Prueba en la carpeta final, sin el repo en el path ni CREDIT_RISK_CONFIG_DIR heredado.
    env = {k: v for k, v in os.environ.items() if k not in ("CREDIT_RISK_CONFIG_DIR", "PYTHONPATH")}
    check = subprocess.run([sys.executable, "-c", SMOKE], cwd=out, env=env, capture_output=True, text=True)
    if check.returncode != 0:
        raise RuntimeError(f"La demo generada no puede puntuar:\n{check.stderr}")
    for cache in out.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    logger.info("Smoke test OK (PD, decisión): %s", check.stdout.strip())
    logger.info(
        "Demo lista en %s (%s%s)", out, info["algorithm"], f" v{info['version']}" if info.get("version") else ""
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, help="joblib de CreditRiskModel (si no, se descarga de Unity Catalog)")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--catalog", default=os.getenv("UC_CATALOG", "workspace"))
    parser.add_argument("--schema", default=os.getenv("UC_SCHEMA", "credit_risk"), help="credit_risk = producción")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "demo")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    build(args.out, args.model, args.alias, args.catalog, args.schema)


if __name__ == "__main__":
    main()
