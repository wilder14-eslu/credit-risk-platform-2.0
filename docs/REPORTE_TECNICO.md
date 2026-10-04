# Reporte técnico: Credit Risk Platform 2.0

**Autor:** Wilder Espinoza Luna · Estadística (UNMSM)
**Datos:** Lending Club 2007-2018 (Kaggle, `accepted_2007_to_2018Q4.csv`)
**Plataforma:** Databricks Free Edition
**Referencia de buenas prácticas:** D. Sinha, *Practical Machine Learning on Databricks* (Packt, 2023), adaptado a Free Edition

## Contenido

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Planteamiento del problema](#2-planteamiento-del-problema)
3. [Datos](#3-datos)
4. [Variables](#4-variables)
5. [Diseño de validación](#5-diseño-de-validación)
6. [Modelado](#6-modelado)
7. [Resultados](#7-resultados)
8. [Arquitectura y despliegue](#8-arquitectura-y-despliegue)
9. [Monitoreo y reentrenamiento continuo](#9-monitoreo-y-reentrenamiento-continuo)
10. [Operación y confiabilidad](#10-operación-y-confiabilidad)
11. [Limitaciones y riesgos](#11-limitaciones-y-riesgos)
12. [Próximos pasos](#12-próximos-pasos)
13. [Referencias](#13-referencias)

## 1. Resumen ejecutivo

Se construyó una plataforma que estima la probabilidad de default (PD) de una solicitud de crédito y decide
si aprobarla, cubriendo el ciclo completo de vida de un modelo: datos, validación, entrenamiento,
despliegue, monitoreo, experimentación online y reentrenamiento. Todo corre en Databricks Free Edition y se
despliega como código desde GitHub Actions, con entornos dev, staging y prod.

El modelo en producción es un **CatBoost** con **AUC 0.690 y KS 0.275** en un test fuera de tiempo
(préstamos emitidos en 2013-H2). Los cuatro algoritmos evaluados quedan en un rango de 0.007 de AUC, lo que
indica que la señal disponible en la solicitud, y no la elección del algoritmo, es el factor limitante. Ese
hallazgo motiva la agenda estadística de la sección 12.

## 2. Planteamiento del problema

**Decisión de negocio.** Ante cada solicitud, aprobar o rechazar. La plataforma entrega la PD, una decisión
(APROBAR o RECHAZAR), una banda de riesgo (A a E) y las tres variables que más empujaron la predicción, para
que la decisión sea explicable ante el solicitante y ante un regulador.

**Restricción de información.** Solo se usa lo que se conoce **al momento de solicitar**. Todo lo que ocurre
después del desembolso (pagos, recuperaciones, FICO posterior) se excluye. Esta restricción define el techo
de desempeño alcanzable y es la razón principal de que el AUC sea moderado.

**Definición del target.** `target_default = 1` si el estado final del préstamo es *Charged Off* o *Default*,
y `0` si es *Fully Paid*. Los préstamos *Current*, *Late* o *In Grace Period* no tienen desenlace y no se
etiquetan.

**Costo asimétrico.** Otorgar un préstamo a quien incumple (falso negativo) se considera 5 veces más costoso
que rechazar a un buen pagador (falso positivo). El umbral de decisión no se fija en 0.5: se elige el que
minimiza el costo esperado en el periodo de validación.

**Métricas.** Siguiendo la distinción entre métricas *offline* y *online* (Sinha, cap. 9):

| Tipo | Métrica | Uso |
|---|---|---|
| Offline (discriminación) | AUC, KS | Selección de modelo y quality gates |
| Offline (calibración) | Brier, ECE | Que la PD sea interpretable como probabilidad |
| Online (negocio) | Tasa de default entre aprobados | Decisión del A/B testing |
| Online (operación) | Latencia, drift | Monitoreo y disparo de reentrenamiento |

## 3. Datos

**Fuente.** 2,260,701 préstamos y 151 columnas, emitidos entre 2007 y 2018.

| Año de emisión | Préstamos | Con desenlace conocido |
|---|---|---|
| 2007-2012 | 95,902 | 100% |
| 2013 | 134,814 | ~100% |
| 2014 | 235,629 | 95% |
| 2015 | 421,095 | 89% |
| 2016 | 434,407 | 67% |
| 2017 | 443,579 | 38% |
| 2018 | 495,242 | 11% |

**Leakage.** Se identificaron 22 columnas con información posterior al desembolso (`total_pymnt`,
`recoveries`, `last_fico_range_high`, `debt_settlement_flag`, entre otras). Se descartan en la capa Silver y
una expectativa bloqueante (`no_leakage_columns`) detiene el pipeline si alguna reaparece.

**Sesgo de censura por la derecha.** Entre los préstamos recientes, los que ya tienen desenlace son sobre
todo los que incumplieron temprano. Etiquetarlos sesgaría el modelo: por ejemplo, casi todos los préstamos a
60 meses de 2016 con desenlace son defaults, y el modelo aprendería que "60 meses = default". Por eso un
préstamo solo se etiqueta si su plazo terminó antes del corte del archivo (`label_snapshot = 2019-01-01`).

**Calidad.** En cada corrida se evalúan expectativas registradas en la tabla `data_quality_log`: número
mínimo de filas, unicidad del identificador, fechas válidas, tasas máximas de nulos, categorías conocidas,
tasa de default entre 3% y 40%, y ausencia de columnas de leakage. Las de severidad *error* detienen el
pipeline.

## 4. Variables

- **18 numéricas** de la solicitud: monto, plazo, tasa, cuota, antigüedad laboral, ingreso, DTI, FICO,
  morosidades, consultas, líneas de crédito, saldo y utilización revolvente, hipotecas, quiebras y
  antigüedad crediticia.
- **8 categóricas:** grado y subgrado, vivienda, verificación de ingresos, propósito, estado, tipo de
  solicitud y listado inicial.
- **5 derivadas:** monto/ingreso, cuota anual/ingreso, saldo revolvente/ingreso, líneas abiertas/totales y
  logaritmo del ingreso.
- **Indicadores de faltantes** para las numéricas con nulos en el entrenamiento, y agrupación de categorías
  raras (menos de 0.5% de frecuencia).

El preprocesador se ajusta **solo con el periodo de entrenamiento** y se serializa dentro del modelo. Así la
misma transformación se aplica al entrenar, al puntuar en batch y al servir en línea, lo que evita el
*training-serving skew* (Sinha, cap. 3) sin depender de online tables, que Free Edition no ofrece.

## 5. Diseño de validación

El split es **temporal por fecha de emisión**, como se valida un scorecard en la industria:

| Periodo | Emisión | Préstamos | Rol |
|---|---|---|---|
| Entrenamiento | 2007 a 2012 | 95,902 | Ajuste de los modelos |
| Validación | 2013, primer semestre | parte de 134,814 | Tuning con Optuna y umbral de decisión |
| Test | 2013, segundo semestre | parte de 134,814 | Desempeño final y quality gates |
| Producción simulada | 2014 a 2015 | 656,724 | Replay mes a mes para monitoreo y A/B |

Ningún préstamo posterior entra al entrenamiento. Cuando el monitoreo dispara un reentrenamiento, las
ventanas se desplazan con la fecha del replay (`as_of`): test = últimos 6 meses con desenlace observado,
validación = 6 meses previos, entrenamiento = todo lo anterior.

**Advertencia metodológica.** El split respeta el orden de emisión, pero no la **disponibilidad de la
etiqueta**. En términos de procesos estocásticos, el target de un préstamo de 2012 (si incumplió antes de
terminar su plazo, en 2015 o 2017) no era medible respecto a la información disponible en enero de 2013. El
backtest es por tanto algo optimista como simulación de una decisión tomada en 2013. Ver la sección 11.

## 6. Modelado

**Candidatos.** Regresión logística (C = 0.5), XGBoost, LightGBM y CatBoost, con hiperparámetros por defecto
conservadores (árboles poco profundos, regularización L2, mínimo de muestras por hoja alto).

**Tuning.** Optuna (hasta 20 intentos o 15 minutos) sobre el algoritmo ganador del benchmark. El objetivo es
el AUC de validación **penalizado por sobreajuste**: si la brecha entre el AUC de entrenamiento y el de
validación supera 0.08, el exceso se resta del puntaje. Si el modelo ajustado no pasa los quality gates, se
recurre al mejor candidato del benchmark con parámetros por defecto que sí los pase.

**Umbral de decisión.** El que minimiza `5 · FN + 1 · FP` en validación.

**Quality gates (absolutos).** AUC de test ≥ 0.66, brecha de sobreajuste ≤ 0.08, Brier ≤ 0.20 y latencia
≤ 5 ms por predicción. Si alguno falla, no se registra nada.

**Quality gate (relativo).** Un challenger solo reemplaza al champion si mejora el AUC en al menos 0.002,
evaluando a ambos sobre el mismo test fuera de tiempo del challenger.

**Explicabilidad.** Contribuciones SHAP por solicitud, agrupadas por variable original (las dummies de una
categórica se suman), y bandas de riesgo: A (PD < 5%), B (< 10%), C (< 20%), D (< 35%) y E.

## 7. Resultados

Corrida de producción sobre los datos reales, test fuera de tiempo (2013-H2):

| Algoritmo | AUC | KS | Brier | Brecha train-test |
|---|---|---|---|---|
| **CatBoost** (seleccionado) | **0.6902** | **0.2752** | 0.1235 | 0.0327 |
| LightGBM | 0.6893 | 0.2736 | 0.1236 | 0.0467 |
| XGBoost | 0.6892 | 0.2738 | 0.1235 | 0.0376 |
| Regresión logística | 0.6835 | 0.2682 | 0.1244 | 0.0140 |

**Interpretación.**

- **El desempeño es coherente con el dataset.** Con solo variables de la solicitud, la literatura y la
  práctica sitúan el AUC de Lending Club alrededor de 0.70. Un AUC de 0.90 o más en este dataset es una señal
  casi segura de leakage.
- **Los tres GBM son estadísticamente indistinguibles a simple vista.** Diferencias de 0.001 de AUC están
  dentro del error de muestreo esperable para un test de este tamaño. Elegir CatBoost por el valor puntual no
  está respaldado por una prueba formal.
- **La regresión logística pierde 0.0067 de AUC y 0.007 de KS**, pero tiene la menor brecha de sobreajuste,
  es totalmente interpretable y es el estándar regulatorio en banca. Si esa diferencia resulta significativa
  y relevante para el negocio es una pregunta abierta (sección 12).
- **La calibración es prácticamente idéntica** entre los cuatro (Brier entre 0.1235 y 0.1244).

## 8. Arquitectura y despliegue

**Medallion en Delta Lake.** Volume `raw` (CSV), Bronze (crudo tipado), Silver (limpio y validado) y Gold
(feature table en Unity Catalog con PK `loan_id`).

**Registro.** MLflow en Unity Catalog con aliases `@champion`, `@challenger` y `@previous_champion` (el
reemplazo vigente de los *stages* que describe Sinha en el cap. 6). Cada versión guarda como tags su commit
(`git_sha`), la tabla y la **versión Delta** de los datos de entrenamiento (`gold_delta_version`) y los cortes
temporales. Cualquier modelo se puede reproducir con
`SELECT * FROM <gold_table> VERSION AS OF <gold_delta_version>`.

**Serving.** Un endpoint de Model Serving con dos variantes (champion 80%, challenger 20%). La API asigna la
variante por hash de `loan_id`, de modo que un mismo solicitante siempre ve la misma variante y la asignación
es reproducible en el replay.

**Aplicaciones.** Una API FastAPI (`/health`, `/api/v1/models`, `/api/v1/predict`, `/api/v1/outcomes`,
`/api/v1/monitoring/summary`) que registra cada predicción en Delta, y un dashboard Streamlit para evaluar
solicitudes y seguir el monitoreo, el A/B y el benchmark.

**Entornos (Sinha, cap. 10).** Free Edition tiene un solo workspace, así que el aislamiento es por esquema:

| Entorno | Esquema | Datos | Serving | Schedules |
|---|---|---|---|---|
| dev | `credit_risk_dev` | real o sintético | endpoint `-dev` | pausados |
| staging | `credit_risk_staging` | sintético | no | pausados |
| prod | `credit_risk` | real | endpoint + Apps | activos |

**CD.** El patrón es *deploy code* (Sinha, cap. 10): se despliega el código y el modelo se entrena en cada
entorno. Tras CI, el flujo es: deploy en dev, luego un **gate de integración en staging** (pipeline completo,
monitoreo con A/B y un smoke check que exige trazabilidad completa del `@champion`) y solo entonces prod, con
aprobación.

## 9. Monitoreo y reentrenamiento continuo

**Replay de producción.** En vez de simular drift, se "re-juega" la originación real de 2014-2015 mes a mes,
con el desenlace llegando con retraso. El monitoreo detecta así cambios reales del portafolio.

**Señales.**

| Tipo | Detección | Umbral de alerta |
|---|---|---|
| Data drift | PSI por variable, test KS | PSI ≥ 0.25 en 3 o más variables |
| Prediction drift | PSI de los scores | PSI ≥ 0.20 |
| Concept drift | Caída de AUC, de KS y aumento del Brier | 0.04, 0.08 y 0.02 respectivamente |
| Concept drift secuencial | DDM sobre errores, Page-Hinkley sobre log-loss | Niveles 2σ (aviso) y 3σ (drift) |
| Prior shift | Test de dos proporciones sobre la tasa de default | Cambio absoluto ≥ 3 puntos |

**Política de reentrenamiento.** Cualquier señal dispara Continuous Training, salvo que la ventana tenga
menos de 300 préstamos o que haya un reentrenamiento en los últimos 3 meses (cooldown). El job de monitoreo
dispara el pipeline de CT con `run_job_task`, pasándole la fecha del replay para desplazar las ventanas.

**A/B testing.** Con al menos 500 préstamos etiquetados por variante, se compara champion contra challenger
con un **bootstrap pareado de la diferencia de AUC** (500 réplicas, α = 0.05) y un **test z de dos
proporciones** sobre la tasa de default entre los aprobados. El challenger se promueve, se detiene o el test
continúa, con un máximo de 6 meses. Un job de rollback devuelve `@champion` a `@previous_champion`.

## 10. Operación y confiabilidad

Aplicando las prácticas de Sinha (cap. 8) dentro de los límites de Free Edition:

- **Timeouts** por tarea y por job, **cola** cuando hay una corrida en curso y **alertas** por fallo y por
  duración al usuario que despliega.
- **Reintentos solo en tareas idempotentes.** La ingesta y las capas Silver/Gold sobrescriben tablas y el
  despliegue del endpoint es declarativo, así que pueden reintentarse. El entrenamiento (un gate fallido no
  mejora reintentando), el replay, el monitoreo y el A/B (hacen *append*) y el rollback (no es idempotente) no
  se reintentan. Un test de contrato lo hace cumplir.
- **Límites de Free Edition considerados:** solo compute serverless, 5 tareas concurrentes, un SQL warehouse
  2X-Small, hasta 3 Apps con apagado a las 24 horas, sin online tables ni GPU.

## 11. Limitaciones y riesgos

1. **Selección sobre el test.** El modelo final se elige por el AUC de test y ese mismo AUC se reporta. Elegir
   el mejor de cuatro sobre el conjunto de reporte introduce un sesgo optimista ("maldición del ganador").
2. **Sin inferencia sobre las diferencias de AUC.** No hay intervalos de confianza ni test pareado entre
   candidatos.
3. **Disponibilidad de la etiqueta.** El target "default en toda la vida del préstamo" usa desenlaces que no
   se conocían al momento de entrenar; el backtest no respeta la filtración de información.
4. **Retraso de etiquetas simplificado.** El replay asume que el desenlace se conoce a los 6 meses, menos que
   el tiempo real hasta un default.
5. **Sesgo de selección.** El modelo solo aprende de préstamos **aprobados** (los rechazados no tienen desenlace); nunca vio a los
   rechazados (problema de *reject inference*). Esto también afecta al A/B, que solo observa desenlaces de
   aprobados.
6. **Equidad no evaluada.** No se midió impacto dispar por grupos (por ejemplo, por estado o nivel de
   ingreso).
7. **Sin demo pública.** Las Databricks Apps requieren login del workspace.

## 12. Próximos pasos

| Prioridad | Acción | Resuelve |
|---|---|---|
| Alta | Seleccionar el modelo con la validación y usar el test una sola vez | 11.1 |
| Alta | Test pareado de DeLong con intervalos de confianza y regla de parsimonia | 11.2 |
| Alta | Baseline de scorecard (WoE + regresión logística) en el benchmark | 11.2 |
| Alta | Experimento de ventana de entrenamiento (solo 2012, 2011-2012, todo, todo con pesos por antigüedad) | Trade-off actualidad vs tamaño de muestra |
| Media | Target de PD a 12 meses con etiquetas disponibles por fecha, o modelo de supervivencia en tiempo discreto que aproveche los préstamos censurados | 11.3, 11.4 |
| Media | Potencia estadística y pruebas secuenciales en el A/B | Error tipo I al revisar resultados cada mes |
| Media | Corrección por comparaciones múltiples en el drift (Benjamini-Hochberg) y Wasserstein como tamaño de efecto | Falsos positivos con muestras grandes |
| Media | Demo pública en Hugging Face Spaces con el `@champion` exportado | 11.7 |
| Baja | Análisis de equidad | 11.6 |

## 13. Referencias

- Sinha, D. (2023). *Practical Machine Learning on Databricks*. Packt.
- Google Cloud. *MLOps: Continuous delivery and automation pipelines in machine learning*.
- Gama, J., Medas, P., Castillo, G. y Rodrigues, P. (2004). Learning with drift detection. *SBIA 2004*.
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2).
- DeLong, E. R., DeLong, D. M. y Clarke-Pearson, D. L. (1988). Comparing the areas under two or more
  correlated receiver operating characteristic curves. *Biometrics*, 44(3).
- Lending Club loan data (2007-2018), Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
