# Despliegue en Databricks Free Edition

## 1. Preparar el workspace

1. Crea tu cuenta en <https://www.databricks.com/learn/free-edition> (catálogo por defecto: `workspace`).
2. **Token personal:** *Settings → Developer → Access tokens → Generate new token*.
3. **SQL Warehouse:** Free Edition trae uno (2X-Small). Copia su **ID** desde *SQL Warehouses*;
   lo usan las Databricks Apps para leer y escribir las tablas Delta.
4. Instala la CLI y autentícate:

   ```bash
   winget install Databricks.DatabricksCLI          # Windows
   databricks auth login --host https://<tu-workspace>.cloud.databricks.com
   ```

## 2. Entorno dev (esquema `credit_risk_dev`, schedules pausados)

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev ct_training_pipeline --params source=synthetic   # prueba rápida sin datos
```

Con datos reales: sube el CSV al Volume `workspace.credit_risk_dev.raw` (ver [DATA.md](DATA.md)) y ejecuta
`databricks bundle run -t dev ct_training_pipeline`. El primer run crea tablas y volúmenes,
entrena los 4 candidatos, registra el ganador como `@champion` y crea el endpoint
`credit-risk-lc-endpoint-dev` (la primera vez tarda ~15-20 min).

Después, re-juega producción y observa el monitoreo:

```bash
databricks bundle run -t dev production_monitoring --params months=6
```

Cada corrida avanza el reloj simulado; cuando el monitoreo detecta drift, dispara solo el
pipeline de CT con `as_of` = reloj actual.

> Si tu cuenta no permite Model Serving: `databricks bundle deploy -t dev --var="deploy_serving=false"`.
> Todo lo demás funciona (el replay puntúa con el modelo cargado desde Unity Catalog).

## 3. Producción + Databricks Apps

```bash
databricks bundle deploy -t prod --var="warehouse_id=<ID-del-warehouse>"
databricks bundle run -t prod ct_training_pipeline        # entrena y crea el endpoint de prod
databricks bundle run -t prod credit_risk_api             # despliega y arranca la API
databricks bundle run -t prod credit_risk_dashboard       # despliega y arranca el panel
databricks bundle run -t prod ct_training_pipeline        # 2da vez: otorga permisos a las apps ya creadas
```

`04_validate_and_deploy.py` otorga a los service principals de las apps `CAN_QUERY` sobre el endpoint
y `SELECT/MODIFY` sobre el esquema. Si tu workspace no permite esos `GRANT`, hazlo desde
*Catalog → credit_risk → Permissions* y *Serving → credit-risk-lc-endpoint → Permissions*.

**Límites de Free Edition a tener en cuenta:** máximo 3 apps, que se detienen solas a las 24 h
(reinícialas con `bundle run`); cuota diaria de compute (si se excede, el compute se apaga hasta el
día siguiente sin perder datos); 5 tareas de jobs en paralelo; un SQL warehouse.

## 4. Probar la API

La API vive detrás de la autenticación de Databricks (OAuth). Con la CLI autenticada:

```bash
TOKEN=$(databricks auth token | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
curl -H "Authorization: Bearer $TOKEN" https://<url-de-la-app>/health
curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  https://<url-de-la-app>/api/v1/predict -d @docs/example_request.json
```

La URL de la app aparece en *Compute → Apps → credit-risk-api*. La documentación interactiva está en `/docs`.

## 5. GitHub Actions

*Settings → Secrets and variables → Actions*:

| Secreto | Valor |
|---|---|
| `DATABRICKS_HOST` | `https://<tu-workspace>.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | token personal |
| `DATABRICKS_WAREHOUSE_ID` | ID del SQL warehouse |

Crea los environments `dev`, `staging` y `prod` (en `prod` activa *Required reviewers*). Sin secretos,
CI sigue en verde y CD se omite con un aviso.

Flujo de CD: **dev** (deploy) -> **staging** (CT completo con datos sintéticos, monitoreo y A/B de 2 meses,
y smoke check de trazabilidad) -> **prod** (con aprobación). Si staging falla, prod no se despliega.

## 6. Operación

| Acción | Comando |
|---|---|
| Reentrenar manualmente | `databricks bundle run -t prod ct_training_pipeline --params trigger=manual` |
| Rollback | `databricks bundle run -t prod model_rollback` |
| Cambiar tráfico A/B, umbrales o cortes temporales | editar `config/platform.yaml` y hacer push |

## 7. Entornos, confiabilidad y límites de Free Edition

Buenas prácticas tomadas de *Practical Machine Learning on Databricks* (D. Sinha, Packt 2023, caps. 7 a 10),
adaptadas a lo que Free Edition permite.

### Entornos (cap. 10: dev, staging y prod)

Free Edition tiene un solo workspace, así que el aislamiento es por **esquema** de Unity Catalog:

| Entorno | Esquema | Datos | Serving | Schedules |
|---|---|---|---|---|
| dev | `credit_risk_dev` | real o sintético | endpoint `-dev` | pausados |
| staging | `credit_risk_staging` | sintético | no se despliega (cuota y ~10 min de aprovisionamiento) | pausados |
| prod | `credit_risk` | real | endpoint de prod + Apps | activos |

Staging no prueba el endpoint de Model Serving: se asume ese riesgo a cambio de no gastar cuota. El endpoint
se valida en dev (manual) y el modelo en prod entra primero como `@challenger` con el A/B.

### Confiabilidad de jobs (cap. 8)

Cada job define timeout global, cola (`queue`), alerta por fallo y por duración (al usuario que despliega).
Cada tarea define su timeout. Solo se **reintentan** las tareas idempotentes (ingesta y silver/gold
sobrescriben tablas; el despliegue del endpoint es declarativo). No se reintentan `train_and_register`
(un quality gate fallido no mejora reintentando y Optuna consume cuota), `replay_production`, `monitor_drift` y
`ab_evaluate` (hacen append a tablas de log) ni `rollback` (reintentarlo volvería a intercambiar champion y
`@previous_champion`). Un test de contrato (`tests/test_bundle_and_serving.py`) lo hace cumplir.

### Trazabilidad de datos y código (cap. 10: lineage y versionado)

Cada versión registrada lleva los tags `git_sha`, `gold_table`, `gold_delta_version` y los cortes del split.
Para reconstruir exactamente los datos con los que se entrenó el modelo vigente:

```sql
-- versión del modelo -> tag gold_delta_version = 7
SELECT * FROM workspace.credit_risk.gold_loan_features VERSION AS OF 7;
```

La retención del historial Delta es de 7 días de `VACUUM` por defecto para los archivos, pero el log de
transacciones se conserva 30 días: si necesitas reproducir modelos más antiguos, extiende
`delta.logRetentionDuration` y `delta.deletedFileRetentionDuration` en la tabla Gold.
`tests/e2e/smoke_check.py` verifica en CD que `@champion` tenga toda la trazabilidad.

### Límites a tener presentes

De la [documentación oficial de Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations):
solo compute serverless, máximo 5 tareas de job concurrentes por cuenta, un SQL warehouse 2X-Small, hasta 3
Databricks Apps con apagado automático a las 24 horas (por eso el panel no es un demo permanente), sin online
tables ni GPU, salida a internet limitada a dominios de confianza y apagado del compute al exceder la cuota diaria.

De Model Serving (cap. 7, cifras de 2023, verifica las vigentes): payload máximo 16 MB, 200 QPS por workspace,
timeout de evaluación de 120 s, 4 GB de memoria por modelo, aprovisionamiento inicial de ~10 minutos y
`scale_to_zero` implica arranque en frío en la primera petición tras inactividad.

