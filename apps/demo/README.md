# Demo pública (Streamlit Community Cloud)

App Streamlit autocontenida con el modelo **champion** exportado de Unity Catalog: no se conecta a
Databricks, así que funciona aunque el workspace o el endpoint estén apagados.

- `demo_core.py`: formulario a variables del modelo, scoring, explicación SHAP y lectura de la foto de
  producción (sin UI, con tests).
- `app.py`: las mismas páginas que el dashboard de Databricks (*Scoring*, *Monitoreo*, *Modelos*, con
  `apps/dashboard/views.py` compartido) más *Por qué este modelo* y *Arquitectura*.

Monitoreo y Modelos muestran una **foto** de las tablas Delta de producción (`monitoring_metrics`,
`drift_by_feature`, `ab_test_results`, `retrain_events`, `model_benchmark`, `model_evaluation`) tomada al
publicar la demo, con las mismas consultas del dashboard. En Databricks esas páginas se leen en vivo.

## Publicación

1. `scripts/build_demo.py` arma `build/demo/` con la app, el paquete `credit_risk`, la configuración,
   el modelo, la foto de producción (si recibe `--warehouse-id`), `reports/results.json` y `docs/figures`, fija las versiones de las dependencias y corre una
   predicción dentro de esa carpeta (sin el repo) para garantizar que arranca.
2. El workflow **Demo pública** (`.github/workflows/demo.yml`, manual) hace lo anterior en GitHub Actions y
   publica la carpeta en la rama huérfana `demo`.
3. Streamlit Community Cloud despliega la rama `demo` (archivo `app.py`, Python 3.12) y se actualiza solo
   en cada publicación.

Para probarla en local con un modelo exportado:

```bash
python scripts/build_demo.py --model ruta/credit_model.joblib
streamlit run build/demo/app.py
```
