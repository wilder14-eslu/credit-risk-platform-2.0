# Credit Risk Platform 2.0 · Lending Club on Databricks

[![CI](https://github.com/wilder14-eslu/credit-risk-platform-2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/wilder14-eslu/credit-risk-platform-2.0/actions/workflows/ci.yml)
[![CD](https://github.com/wilder14-eslu/credit-risk-platform-2.0/actions/workflows/cd.yml/badge.svg)](https://github.com/wilder14-eslu/credit-risk-platform-2.0/actions/workflows/cd.yml)
[![CodeQL](https://github.com/wilder14-eslu/credit-risk-platform-2.0/actions/workflows/codeql.yml/badge.svg)](https://github.com/wilder14-eslu/credit-risk-platform-2.0/actions/workflows/codeql.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Databricks](https://img.shields.io/badge/Databricks-Free%20Edition-FF3621?logo=databricks&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Unity%20Catalog-0194E2?logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Plataforma end-to-end de **riesgo crediticio** (probabilidad de default) sobre **2.26 millones de préstamos
reales de Lending Club (2007-2018)**, desplegada 100% en **Databricks Free Edition**: desde el planteamiento
del problema hasta el despliegue, el monitoreo en producción, el A/B testing y el reentrenamiento continuo,
con todo el ciclo automatizado como código.

> **Reporte técnico completo:** [docs/REPORTE_TECNICO.md](docs/REPORTE_TECNICO.md) (metodología, resultados,
> decisiones estadísticas y limitaciones).

## En 30 segundos

| | |
|---|---|
| **Problema** | Decidir si aprobar una solicitud de crédito estimando su probabilidad de default (PD) con información disponible solo al momento de solicitar |
| **Datos** | Lending Club 2007-2018, 151 columnas; se usan 26 variables de la solicitud más 5 derivadas |
| **Modelo en producción** | CatBoost elegido en un benchmark de 4 algoritmos con Optuna; AUC 0.690 y KS 0.275 en test fuera de tiempo |
| **Plataforma** | Databricks Free Edition: Jobs serverless, Delta Lake, Unity Catalog, MLflow, Model Serving y Databricks Apps |
| **MLOps** | Nivel 2 de Google: CI/CD con entornos dev, staging y prod; entrenamiento continuo disparado por drift; A/B testing con promoción y rollback |
| **Calidad** | 67 tests (cobertura ~93%), pipeline end-to-end en CI, CodeQL, Dependabot, lineage código-datos-modelo |

## Resultados

Benchmark sobre el **test fuera de tiempo** (préstamos emitidos en 2013-H2, nunca vistos al entrenar), corrida
de producción con los datos reales:

| Algoritmo | AUC | KS | Brier | Brecha train-test (AUC) | Diagnóstico |
|---|---|---|---|---|---|
| **CatBoost** (seleccionado) | **0.6902** | **0.2752** | 0.1235 | 0.033 | buen ajuste |
| LightGBM | 0.6893 | 0.2736 | 0.1236 | 0.047 | buen ajuste |
| XGBoost | 0.6892 | 0.2738 | 0.1235 | 0.038 | buen ajuste |
| Regresión logística | 0.6835 | 0.2682 | 0.1244 | 0.014 | buen ajuste |

**Cómo leerlo.** Un AUC cercano a 0.69 es lo esperable en Lending Club cuando solo se usan variables de la
solicitud; modelos publicados con AUC de 0.90 o más suelen filtrar información posterior al desembolso
(pagos, recuperaciones). Las diferencias entre los tres GBM son de 0.001 de AUC y no deberían interpretarse
como significativas sin un test pareado; ese análisis y la comparación formal contra la regresión logística
están en el [roadmap](#limitaciones-conocidas-y-roadmap).

## Arquitectura

```mermaid
flowchart LR
    subgraph GH["GitHub"]
        code["Código + config<br/>(platform.yaml)"] --> ci["CI: ruff, pytest (cov 85%+),<br/>e2e local, validar bundle, CodeQL"]
        ci --> dev["CD: deploy dev"]
        dev --> stg["Gate staging:<br/>CT + monitoreo + smoke check<br/>de trazabilidad"]
        stg --> prod["Deploy prod<br/>(con aprobación)"]
    end

    subgraph DBX["Databricks Free Edition (serverless)"]
        direction TB
        vol[("Volume raw<br/>CSV Lending Club")] --> bronze[("Bronze")]
        bronze --> silver[("Silver + data_quality_log")]
        silver --> gold[("Gold: feature table UC<br/>PK loan_id, versionada en Delta")]
        gold --> train["CT pipeline<br/>benchmark 4 modelos + Optuna"]
        train --> reg[["MLflow + UC Registry<br/>@champion, @challenger<br/>tags git_sha + versión Delta"]]
        reg --> serve["Model Serving<br/>champion 80%, challenger 20%"]
        gold --> replay["Replay de producción<br/>2014-2015 mes a mes"]
        replay --> log[("inference_log + outcomes")]
        log --> mon["Monitoreo<br/>PSI, KS, DDM, Page-Hinkley"]
        mon --> ab["A/B: bootstrap AUC + z-test"]
        ab -->|promover / detener| reg
        mon -->|drift: retrain con as_of| train
        api["App: API FastAPI"] -->|variante sticky por loan_id| serve
        api --> log
        dash["App: Dashboard Streamlit"] --> serve
        dash -->|SQL warehouse| log
    end

    prod --> DBX
    client(["Cliente / core bancario"]) --> api
```

| Pieza | Tecnología |
|---|---|
| Ingesta y calidad | Jobs serverless, Delta Lake, expectativas bloqueantes (nulos, rangos, duplicados, columnas de leakage) |
| Feature store | Feature table en Unity Catalog con PK `loan_id`; el preprocesador viaja dentro del modelo para evitar *training-serving skew* |
| Entrenamiento | Regresión logística, XGBoost, LightGBM, CatBoost + Optuna con penalización de sobreajuste |
| Registro | MLflow en Unity Catalog con aliases `@champion`, `@challenger`, `@previous_champion` |
| Serving | Model Serving con dos variantes y split de tráfico para A/B |
| API y panel | Databricks Apps: FastAPI (`apps/api`) y Streamlit (`apps/dashboard`) |
| Orquestación | Databricks Jobs con `condition_task` y `run_job_task` (monitoreo dispara reentrenamiento) |
| Infraestructura como código | Databricks Asset Bundles (`databricks.yml`, `resources/jobs.yml`) |
| CI/CD | GitHub Actions: dev, gate de integración en staging, prod con aprobación |

## Ciclo de vida cubierto

| Fase | Qué se hace | Dónde |
|---|---|---|
| 1. Planteamiento | Decisión de negocio, definición del target, costo asimétrico (otorgar a un moroso cuesta 5 veces más que rechazar a un buen pagador) | [Reporte, sección 2](docs/REPORTE_TECNICO.md#2-planteamiento-del-problema) |
| 2. Datos | Parser del CSV crudo, eliminación de 22 columnas de leakage, control del sesgo de censura | `src/credit_risk/data/`, [DATA.md](docs/DATA.md) |
| 3. Calidad | Expectativas bloqueantes registradas en `data_quality_log` | `src/credit_risk/data/quality.py` |
| 4. Variables | 26 variables de la solicitud + 5 ratios derivados + indicadores de faltantes | `src/credit_risk/features/` |
| 5. Validación | Split temporal: train 2007-2012, validación 2013-H1, test 2013-H2 | `src/credit_risk/models/training.py` |
| 6. Modelado | Benchmark, Optuna, umbral por costo esperado, quality gates absolutos y relativos | `jobs/03_train_register.py`, `jobs/04_validate_and_deploy.py` |
| 7. Explicabilidad | Contribuciones SHAP por solicitud (top 3 razones) y bandas de riesgo A-E | `src/credit_risk/models/credit_model.py` |
| 8. Despliegue | Registro en UC, Model Serving con A/B, API y dashboard | `src/credit_risk/registry/`, `apps/` |
| 9. Monitoreo | Data, prediction y concept drift; prior shift; etiquetas con retraso | `src/credit_risk/monitoring/`, `jobs/06_monitor.py` |
| 10. Experimentación | A/B champion vs challenger con promoción automática y rollback | `jobs/07_ab_evaluate.py`, `jobs/08_rollback.py` |
| 11. Reentrenamiento | Continuous Training semanal y disparado por drift, con ventanas desplazadas | `resources/jobs.yml` |
| 12. Operación | Reintentos idempotentes, timeouts, alertas, lineage, entornos aislados | [DEPLOYMENT.md](docs/DEPLOYMENT.md#7-entornos-confiabilidad-y-límites-de-free-edition) |

## Decisiones que hacen la diferencia

| Tema | Decisión | Por qué importa |
|---|---|---|
| **Validación fuera de tiempo** | Split por fecha de emisión, nunca aleatorio | Así se valida un scorecard en banca: el modelo se mide en préstamos posteriores a los que vio |
| **Sin data leakage** | Solo variables conocidas al solicitar; 22 columnas posteriores al desembolso se descartan y un test lo verifica | Es el error más común con este dataset e infla el AUC artificialmente |
| **Sin sesgo de censura** | Solo se etiquetan préstamos cuyo plazo terminó antes del corte del archivo | Entre los préstamos recientes, los que ya tienen desenlace son sobre todo defaults tempranos |
| **Umbral por costo** | El umbral minimiza el costo esperado en validación (FN = 5, FP = 1), no se fija en 0.5 | Alinea la decisión con el negocio en vez de con la exactitud |
| **Drift real, no simulado** | La producción es un replay mes a mes de la originación real 2014-2015 | El monitoreo detecta cambios reales del portafolio de Lending Club |
| **Etiquetas con retraso** | El desenlace llega después; el data drift actúa como alerta temprana | En crédito la performance real llega tarde |
| **A/B testing real** | Dos variantes en el mismo endpoint, asignación determinista por hash de `loan_id` | La promoción se decide con bootstrap del AUC y test z sobre la morosidad de aprobados |
| **Trazabilidad** | Cada versión del modelo guarda su commit (`git_sha`) y la versión Delta exacta de sus datos | Cualquier modelo en producción se puede reproducir con *time travel* |
| **Staging como gate** | Un pipeline completo en staging debe pasar antes de tocar prod | Un error de integración nunca llega a producción |

## Monitoreo: tipos de drift

| Tipo | Qué cambia | Detección |
|---|---|---|
| Data drift | P(X) | PSI por variable (numéricas y categóricas) + test KS |
| Prediction drift | P(ŷ) | PSI de la distribución de scores |
| Concept drift | P(y\|X) | Caída de AUC/KS, Brier, **DDM** sobre errores, **Page-Hinkley** sobre log-loss |
| Prior shift | P(y) | Test de dos proporciones sobre la tasa de default |

Umbrales versionados en [`config/platform.yaml`](config/platform.yaml): cambiarlos pasa por CI/CD.

## Madurez MLOps (Google Cloud)

| Nivel | Requisito | Implementación |
|---|---|---|
| 1 | Pipeline de entrenamiento automatizado | Job `ct_training_pipeline`: Bronze, Silver, Gold, entrenamiento, validación, despliegue |
| 1 | Validación de datos y de modelo | Expectativas bloqueantes; gates absolutos (AUC, sobreajuste, Brier, latencia) y relativos (champion vs challenger en el mismo test) |
| 1 | Feature store y metadata | Feature table en UC + MLflow con corridas anidadas por candidato |
| 1 | Continuous Training | Schedule semanal y disparo desde el monitoreo con ventanas desplazadas (`as_of`) |
| 2 | CI | Lint, tests (3.11 y 3.12), cobertura 85%+, pipeline end-to-end local, validación del bundle, CodeQL |
| 2 | CD del pipeline y del modelo | dev, gate de integración en staging con smoke check de trazabilidad, prod con aprobación |
| 2 | Monitoreo y experimentación online | Data, prediction y concept drift; A/B con promoción automática; rollback |

## Ingeniería y confiabilidad

- **Tests:** 67 tests unitarios y de contrato (cobertura ~93%) más un pipeline end-to-end que ejecuta todos los
  jobs en local con DuckDB y MLflow sobre SQLite, sin necesidad de Databricks.
- **Contratos de infraestructura:** los tests verifican que todas las tareas sean serverless, que solo las
  tareas idempotentes tengan reintentos, que staging y prod no compartan esquema ni endpoint y que no haya
  correos ni carpetas compartidas en la configuración.
- **Jobs:** timeout por tarea y por job, cola, alertas por fallo y por duración; reintentos solo donde
  repetir no duplica datos.
- **Seguridad:** secretos solo en GitHub Actions, CodeQL, Dependabot y despliegue bajo la carpeta del usuario.

## Limitaciones conocidas y roadmap

Lo que un revisor exigente señalaría, y cómo se va a abordar:

| Limitación | Impacto | Próximo paso |
|---|---|---|
| El modelo final se elige comparando AUC en el test | Ligero optimismo en el AUC reportado ("maldición del ganador") | Seleccionar con la validación y usar el test una sola vez |
| Las diferencias de AUC entre candidatos no tienen test estadístico | No se puede afirmar que CatBoost sea mejor que LightGBM o XGBoost | Test pareado de DeLong y regla de parsimonia |
| El target usa desenlaces que no se conocían al momento de entrenar (préstamos de 2012 terminan en 2015-2017) | El backtest respeta el orden de emisión, pero no la disponibilidad de la etiqueta | Target de PD a 12 meses (estándar Basilea) o modelo de supervivencia con censura |
| El replay asume que el desenlace se conoce a los 6 meses | Simplificación de la llegada real de etiquetas | Alinear el retraso con la definición del target |
| Las Databricks Apps requieren login del workspace y en Free Edition se apagan a las 24 horas | No hay demo pública | Demo en Hugging Face Spaces con el modelo `@champion` exportado |
| Sin baseline de scorecard (WoE + regresión logística) | Falta el punto de comparación estándar en banca | Agregarlo al benchmark |

## Estructura

```text
├── databricks.yml               # Asset Bundle: targets dev, staging y prod; Apps solo en prod
├── resources/jobs.yml           # CT pipeline, monitoreo + A/B + CT, rollback (con políticas de confiabilidad)
├── config/
│   ├── data_schema.yaml         # Variables, target, columnas de leakage
│   └── platform.yaml            # Cortes temporales, gates, drift, A/B, serving
├── src/credit_risk/
│   ├── data/                    # Parser de Lending Club, calidad, datos sintéticos (CI)
│   ├── features/                # Variables derivadas + preprocesador (viaja con el modelo)
│   ├── models/                  # Candidatos, métricas, split temporal, modelo pyfunc con SHAP
│   ├── monitoring/              # Data y concept drift, A/B, replay, política de CT
│   └── registry/                # MLflow/UC, Model Serving, trazabilidad (lineage)
├── jobs/                        # 01 ingesta ... 08 rollback
├── apps/api/                    # Databricks App: FastAPI
├── apps/dashboard/              # Databricks App: Streamlit
├── tests/                       # Tests unitarios y de contrato + tests/e2e (pipeline completo y smoke check)
└── docs/                        # Reporte técnico, datos y despliegue
```

## Inicio rápido

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

pytest                                         # 67 tests, sin Databricks
python tests/e2e/run_pipeline_locally.py       # ciclo completo con datos sintéticos
```

En Databricks (guía completa en [DEPLOYMENT.md](docs/DEPLOYMENT.md)):

```bash
databricks bundle deploy -t dev
databricks bundle run -t dev ct_training_pipeline --params source=synthetic
databricks bundle run -t dev production_monitoring --params months=6
```

## Documentación

- [Reporte técnico](docs/REPORTE_TECNICO.md): planteamiento, datos, metodología, resultados, limitaciones.
- [Datos](docs/DATA.md): perfil del archivo, sesgo de censura y cómo subirlo.
- [Despliegue](docs/DEPLOYMENT.md): entornos, CI/CD, confiabilidad y límites de Free Edition.

> Versión anterior (Docker + PostgreSQL + Prefect, dataset Give Me Some Credit):
> [credit-risk-ml-platform](https://github.com/wilder14-eslu/credit-risk-ml-platform).

## Autor

**Wilder Espinoza Luna** · Estadística (UNMSM) · Machine Learning Engineering
