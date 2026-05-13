---
title: YouTube Boost AI Definitivo
emoji: 🎬
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: "5.33.0"
app_file: app.py
python_version: "3.11"
pinned: false
---

# 🎬 YouTube Boost AI Definitivo

Prototipo académico funcional para **analizar videos orgánicos de YouTube** y recomendar si conviene **impulsar**, **ajustar antes de impulsar**, **monitorear**, **no impulsar** o enviar a **revisión humana**.

El sistema integra varias capas de IA y análisis:

- Modelo supervisado entrenado con datasets reales de YouTube.
- Descarga opcional de videos de YouTube en 360p mediante URL, solo para videos de hasta 60 segundos.
- Transcripción automática obligatoria con `faster-whisper`.
- OCR obligatorio sobre frames si hay video.
- Evaluación preliminar de políticas de YouTube Ads.
- Análisis semántico del guion/copy.
- Análisis visual ligero basado en composición: regla de tercios, contraste, figura-fondo, carga cognitiva y jerarquía visual.
- Predicción/estimación de métricas esperadas tras potenciación.
- LLM open source pequeño: `HuggingFaceTB/SmolLM2-135M-Instruct`, cargado desde Hugging Face si el entorno lo permite.
- Fallback por reglas si el LLM no está disponible, para no romper la demo.

> El sistema **no garantiza viralidad**, **no predice ROI causal** y **no aprueba anuncios oficialmente**. Funciona como apoyo a decisión humana.

---

## Mejoras de UI y robustez (cambio actual)

Esta versión introduce:

- **Estética dark inspiradora**: tema violeta/cyan sobre fondo `#0b0d12`, hero con pills, botón primario con gradiente. CSS personalizado en `app.py`.
- **Descarga robusta de YouTube**: el sistema prueba en cascada `yt-dlp → pytubefix → pytube` y, si los tres fallan, recupera metadata vía oEmbed con un mensaje claro al usuario para subir el MP4 manualmente. Video recortado automáticamente a ≤60 s con `ffmpeg` si la duración descargada excede el límite. Resolución forzada a 360p.
- **Resumen ejecutivo para marketing**: lenguaje claro, sin jerga técnica. Headline por acción, tabla de métricas clave en porcentajes y dólares formateados, bullets cortos en "Por qué" y "Qué hacer ahora". Módulo `src/exec_summary.py`.
- **Análisis visual con 10 frames anotados**: timestamps uniformemente espaciados, frame anotado en PNG con grid de tercios + intersecciones + centroide focal. Recomendaciones agrupadas por teoría (regla de tercios, composición geométrica, foco visual, iluminación/contraste, carga cognitiva). Módulo `src/visual_composition.py`.
- **Gráficos estadísticos dark**: 3 PNG generados en cada análisis — diagnóstico (potencial, retención, engagement, guion, riesgo política), proyección actual vs esperado con $ y categorías sensibles detectadas. Paleta consistente con la UI. Módulo `src/analytics_viz.py`.
- **Política permisiva pero estricta cuando importa**: solo se escala a `REVISIÓN HUMANA` cuando el evaluador detecta 3+ categorías de alta severidad o cuando NO hay nada de texto evaluable (ni título, ni descripción, ni transcripción). La transcripción ausente por sí sola ya **no bloquea** — el modelo + screening parcial continúan con título + descripción.

---

## Ejecución local

```bash
pip install -r requirements.txt
python tests/smoke_test.py    # 18 PASS · 1 SKIP · 0 FAIL en ~4s
python app.py                 # demo Gradio dark theme
```

---

## Despliegue en Hugging Face Spaces

Subir el contenido del repo a la raíz del Space. La raíz debe contener:

```text
app.py
README.md
requirements.txt
packages.txt
src/
models/
data/
demo_cache/
docs/
tests/
```

La demo funciona con:

- `app.py` en la raíz.
- `models/best_model.joblib` (modelo entrenado con 68 526 filas reales de Kaggle).
- `models/metric_regressors.joblib`.
- `ffmpeg` instalado por `packages.txt` (para extracción de audio y recorte).
- `faster-whisper` para transcripción.
- 3 descargadores en cascada: `yt-dlp`, `pytubefix`, `pytube` (todos en `requirements.txt`).
- `pytesseract` para OCR.

---

## Restricciones técnicas importantes

- Solo se procesan videos de hasta **60 segundos** en la demo. Si la descarga trae un video más largo, se recorta automáticamente con `ffmpeg`.
- Si una URL de YouTube no se puede descargar con ningún backend, el sistema muestra metadata oEmbed y pide al usuario subir el MP4 manualmente.
- La política de **REVISIÓN HUMANA** ahora solo se activa cuando hay 3+ categorías de alta severidad detectadas o cuando no hay nada de texto evaluable. Esto evita el bloqueo constante reportado en versiones anteriores.
- OCR se ejecuta si hay video. Si no encuentra texto, lo informa explícitamente.
- El LLM local es ligero, pero puede tardar en la primera carga. Si falla, el sistema usa reglas sin romper la demo.

---

## Fuentes de datos

El repositorio académico completo incluye datasets reales de YouTube y transcripciones públicas. La versión de despliegue conserva el modelo entrenado, patrones de guion y archivos necesarios para la demo.

---

## Referencias técnicas

- `faster-whisper`: transcripción local optimizada con CTranslate2.
- `HuggingFaceTB/SmolLM2-135M-Instruct`: LLM open source compacto para redacción de recomendaciones.
- `yt-dlp`: descarga opcional de video desde URL para análisis académico.
- YouTube Ads / Advertiser-friendly guidelines: base para evaluación preliminar de políticas.

## Cambios aplicados en esta versión full

Incluye todos los cambios del repositorio light y conserva los datasets, notebooks y archivos de entrenamiento del paquete full:

- Reemplazo de paneles JSON visibles por análisis narrativos en Markdown para métricas, LLM técnico, OCR, políticas, composición visual, guion y sentimiento.
- El JSON completo se mantiene únicamente en la pestaña **JSON / API** para depuración e integración.
- Análisis de sentimiento mejorado con porcentajes, gráfico de distribución y nubes de palabras positivas, neutras y negativas sin stopwords.
- Cálculo automático de horas desde publicación usando `publishedAt` de YouTube cuando hay `YOUTUBE_API_KEY`.
- Transcripción automática de MP4 con `faster-whisper` local como motor preferido y Google Speech Recognition como respaldo.
- Análisis visual ampliado con composición áurea, retícula 0.382 / 0.618, score áureo y conclusión visual accionable en la misma ventana.
- Se conservan datasets raw/processed, notebooks y scripts de entrenamiento para trabajar fuera del modo demo.


## Actualización generada: QLoRA + XGBoost + tesis

Esta versión fue regenerada fusionando la versión `youtube-ai-recomendations` con la versión full. Incluye:

- Adaptador Qwen/Qwen2.5-0.5B-Instruct QLoRA en `models/qwen_marketing_qlora/`.
- Segundo modelo `models/xgboost_paid_ads.joblib` para estimar rendimiento pagado y CPM.
- Gate metodológico: regresión logística primero; si `probability >= 0.51`, se ejecuta XGBoost.
- Auditoría de tesis en `docs/THESIS_SUSTAINABILITY_AUDIT.md`.
- Notebook principal full y notebook light reproducible en `notebooks/`.

### Ejecutar

```bash
pip install -r requirements.txt
python app.py
```

### Reentrenar XGBoost

```bash
pip install -r requirements-train.txt xgboost
python scripts/train_xgboost_paid_ads.py
```
