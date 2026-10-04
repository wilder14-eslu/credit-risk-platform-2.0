# Demo pública (Streamlit Community Cloud)

App Streamlit autocontenida con el modelo **champion** exportado de Unity Catalog: no se conecta a
Databricks, así que funciona aunque el workspace o el endpoint estén apagados.

- `demo_core.py`: formulario a variables del modelo, scoring y explicación SHAP (sin UI, con tests).
- `app.py`: pestañas *Evaluar solicitud*, *Por qué este modelo* y *Arquitectura*.

## Publicación

1. `scripts/build_demo.py` arma `build/demo/` con la app, el paquete `credit_risk`, la configuración,
   el modelo, `reports/results.json` y `docs/figures`, fija las versiones de las dependencias y corre una
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
