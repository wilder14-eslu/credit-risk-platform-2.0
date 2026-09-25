# Datos: Lending Club 2007-2018

- **Fuente:** [All Lending Club loan data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club),
  archivo `accepted_2007_to_2018Q4.csv` (~1.7 GB, 2,260,701 préstamos, 151 columnas).
- **No se versiona en Git** (tamaño y licencia): `data/` y `*.csv` están en `.gitignore`.

## Perfil del archivo (medido sobre el CSV completo)

| Año de emisión | Préstamos | Con desenlace (sin filtro de madurez) |
|---|---|---|
| 2007-2012 | 95,902 | 100% |
| 2013 | 134,814 | ~100% |
| 2014 | 235,629 | 95% |
| 2015 | 421,095 | 89% |
| 2016 | 434,407 | 67% |
| 2017 | 443,579 | 38% |
| 2018 | 495,242 | 11% |

`loan_status`: Fully Paid 1,076,751 · Current 878,317 · Charged Off 268,559 · Late/Grace ~34k ·
33 filas de pie de página sin datos (el parser las descarta).

## Por qué solo préstamos "maduros" tienen etiqueta

Entre los préstamos recientes, los que ya tienen desenlace son sobre todo los que
incumplieron temprano (o pagaron anticipado). Etiquetarlos sesga el modelo: por
ejemplo, los préstamos a 60 meses de 2016 con desenlace son casi todos defaults.
Por eso `parse_raw` solo asigna `target_default` si `issue_month + term <= label_snapshot`
(2019-01-01). Consecuencias:

- Entrenamiento inicial: 2007-2012; validación 2013-H1; test 2013-H2.
- Producción (replay): originaciones 2014-01 a 2015-12 (préstamos a 36 meses ya vencidos).
- Los préstamos a 60 meses de 2014-2015 se puntúan y entran al monitoreo de data drift,
  pero no tienen etiqueta.

## Cómo subirlo a Databricks

1. Descarga el CSV desde Kaggle a tu PC (`data/accepted_2007_to_2018Q4.csv`).
2. Crea el Volume ejecutando una vez el pipeline (o `CREATE VOLUME workspace.credit_risk.raw`).
3. Súbelo:
   - **UI:** *Catalog → workspace → credit_risk → Volumes → raw → Upload to this volume*.
   - **CLI:** `databricks fs cp data/accepted_2007_to_2018Q4.csv dbfs:/Volumes/workspace/credit_risk/raw/`
     (para dev usa `credit_risk_dev`).

Sin el archivo puedes probar todo con datos sintéticos del mismo formato:
`databricks bundle run -t dev ct_training_pipeline --params source=synthetic`.
