---
title: Credit Risk Platform 2.0
emoji: 💳
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: true
license: mit
short_description: PD de préstamos Lending Club con un modelo MLOps
tags:
  - credit-risk
  - mlops
  - databricks
  - catboost
  - shap
---

# Credit Risk Platform 2.0 (demo)

Demo pública del modelo **champion** de una plataforma MLOps de riesgo crediticio construida en
**Databricks Free Edition** sobre 2.26 millones de préstamos de Lending Club (2007-2018).

- **Evaluar solicitud:** probabilidad de default, decisión con umbral por costos y factores SHAP.
- **Por qué este modelo:** selección estadística (DeLong + Holm + parsimonia), calibración, deciles,
  estabilidad por cosecha y SHAP.
- **Arquitectura:** del dato crudo al reentrenamiento continuo con A/B testing.

El modelo se exporta desde Unity Catalog con `scripts/build_hf_space.py`; el Space no se conecta a Databricks.

Código y reporte técnico: https://github.com/wilder14-eslu/credit-risk-platform-2.0

> Demo educativa con datos históricos públicos. No es una herramienta de decisión crediticia real.
