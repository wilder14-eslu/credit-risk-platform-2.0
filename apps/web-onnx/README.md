# 💳 Credit Risk Platform 2.0 · ONNX Edge Web Application

Plataforma web de riesgo crediticio con inferencia nativa en el navegador mediante **ONNX Runtime WebAssembly (WASM)**.

Misma interfaz, páginas y capacidades que la versión Streamlit (`apps/demo` y `apps/dashboard`), pero con:
- **Carga Inmediata (< 200 ms)**: Cero segundos de *cold-start*, sin contenedores durmientes ni esperas de arranque de Python.
- **Inferencia en el Cliente (< 3 ms)**: Ejecución local directa en WebAssembly en el navegador (`credit_risk_model.onnx`).
- **100% Estática**: Se puede desplegar en GitHub Pages, Cloudflare Pages, Vercel, Netlify o dentro de Databricks Apps sin necesidad de servidor de inferencia.
- **Sin Costos de Servidor ni Límites de Red**: Inferencia ilimitada y offline.

---

## 🚀 Inicio Rápido Local

Para abrir la plataforma localmente:

```bash
cd "apps/web-onnx"
python serve.py
```

El servidor abrirá automáticamente la aplicación en tu navegador predeterminado (típicamente en `http://127.0.0.1:8502`).

Alternativamente, puedes usar cualquier servidor HTTP estático:

```bash
# Con Python
python -m http.server 8080 -d "apps/web-onnx"

# Con Node / npx
npx serve apps/web-onnx
```

---

## 📁 Estructura del Módulo

```
apps/web-onnx/
├── index.html              # Aplicación SPA completa
├── serve.py                # Servidor local con soporte WASM y auto-apertura
├── README.md               # Documentación técnica
├── css/
│   └── style.css           # Estilos modernos basados en la identidad visual de Streamlit
├── js/
│   ├── ort.min.js          # ONNX Runtime WebAssembly v1.18
│   ├── chart.min.js        # Librería de gráficos interactivos
│   ├── engine.js           # Motor de preprocesamiento e inferencia ONNX en cliente
│   ├── app.js              # Lógica de navegación, formularios y gráficos
│   ├── ort-wasm.wasm       # Binarios WebAssembly de inferencia
│   └── ort-wasm-simd.wasm
├── model/
│   ├── credit_risk_model.onnx   # Modelo ONNX (LightGBM Champion, 344 KB)
│   └── credit_risk_meta.json    # Metadatos, medianas, vocabularios y umbral
├── data/
│   └── app_data.json       # Datos precomputados para Monitoreo y Benchmark
└── figures/                # Figuras en alta resolución del análisis estadístico
```

---

## 🌐 Despliegue en Producción (GitHub Pages / Vercel)

Al ser una aplicación 100% estática (HTML/CSS/JS + WASM), se puede publicar gratis en GitHub Pages:

1. Ve a los ajustes de tu repositorio en GitHub: **Settings > Pages**.
2. Selecciona la rama y la carpeta `/apps/web-onnx` (o publica vía GitHub Actions con `actions/upload-pages-artifact`).
3. La aplicación estará disponible globalmente con CDN de alta velocidad y carga instantánea.
