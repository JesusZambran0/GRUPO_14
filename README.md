---
title: YouTube AI Recomendations
emoji: 🎬
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: "5.33.0"
app_file: app.py
python_version: "3.11"
pinned: false
---

# 🎬 YouTube AI Recomendations

**YouTube AI Recomendations** es un prototipo académico desarrollado en Gradio para analizar videos cortos de YouTube y apoyar la decisión de si un contenido tiene potencial para ser impulsado mediante pauta publicitaria.

El sistema permite cargar videos de máximo **1 minuto de duración** o ingresar una URL de YouTube. A partir del video, analiza audio, texto, frames, métricas, composición visual, guion, sentimiento, riesgos publicitarios y modelos predictivos para generar una recomendación final.

> Este software es un **prototipo académico**. No garantiza viralidad, ventas, ROI ni aprobación oficial en YouTube Ads. Su objetivo es servir como herramienta de apoyo para análisis, investigación y toma de decisiones preliminar.

---

## 🚀 ¿Para qué sirve?

El prototipo ayuda a responder preguntas como:

- ¿Este video tiene potencial para ser impulsado con publicidad?
- ¿El mensaje es claro para una audiencia?
- ¿El video tiene buen gancho, buena composición visual y buen CTA?
- ¿Hay riesgos preliminares de políticas publicitarias?
- ¿Qué debería mejorarse antes de invertir presupuesto?
- ¿Cuál podría ser el rendimiento estimado si se pauta?
- ¿Qué CPM o eficiencia aproximada podría esperarse según el modelo disponible?

---

## 🧠 ¿Qué analiza el sistema?

YouTube AI Recomendations combina varias capas de análisis:

### 1. Video

El sistema procesa videos cortos de hasta **60 segundos**.

Puede trabajar con:

- Archivo MP4 cargado manualmente.
- URL de YouTube.
- Metadata ingresada por el usuario.
- Transcripción manual si el audio no puede procesarse.

Del video se extraen señales audiovisuales, frames y datos técnicos básicos para alimentar el análisis.

### 2. Transcripción automática

El audio del video se convierte en texto para analizar el mensaje hablado.

Según la configuración del entorno, el sistema puede usar:

- `faster-whisper`.
- Google Speech Recognition.
- Transcripción manual ingresada por el usuario.

La transcripción permite evaluar claridad del mensaje, estructura del guion, hook inicial, propuesta de valor, CTA o cierre y coherencia entre lo que se dice y lo que se muestra.

### 3. OCR del texto en pantalla

El sistema analiza frames del video para detectar texto visible.

Esto ayuda a identificar subtítulos, frases destacadas, llamados a la acción, promociones, mensajes de urgencia, señales de confianza, exceso de texto en pantalla y coherencia entre texto visual, título y guion.

Si el OCR detecta texto imperfecto, el sistema puede apoyarse en LLM o reglas para interpretarlo de forma más comprensible.

### 4. Análisis visual y composición

El prototipo revisa la composición visual del video mediante frames representativos.

Evalúa aspectos como:

- Regla de tercios.
- Composición áurea.
- Balance visual.
- Claridad focal.
- Contraste.
- Complejidad visual.
- Movimiento.
- Legibilidad del texto en pantalla.

Este módulo busca determinar si el video tiene una estructura visual clara y si puede captar atención en formatos de pauta digital.

### 5. Análisis de guion y lenguaje natural

A partir del título, descripción, transcripción y OCR, el sistema evalúa el contenido textual y narrativo.

Puede identificar si el video se acerca a un formato comercial, educativo, informativo, humorístico, musical, branding o entretenimiento.

También analiza claridad comunicacional, hook, tono, complejidad del mensaje, coherencia semántica, fuerza del CTA, debilidades del guion y mejoras recomendadas.

### 6. Análisis de sentimiento

Si se ingresan comentarios o se obtienen mediante API, el sistema analiza la percepción del público.

Entrega porcentaje de sentimiento positivo, neutro y negativo, palabras frecuentes y lectura general de aceptación, duda o rechazo.

### 7. Evaluación preliminar de políticas publicitarias

El sistema realiza un screening preliminar para detectar posibles riesgos antes de pautar.

Puede advertir sobre lenguaje sensible, promesas fuertes, claims riesgosos, temas que podrían requerir revisión, falta de información textual suficiente o posibles conflictos con lineamientos de anuncios.

> Esta evaluación no reemplaza la revisión oficial de YouTube Ads. Es una aproximación académica para anticipar riesgos.

### 8. Modelo predictivo principal

El sistema utiliza un modelo de clasificación para estimar si el video tiene potencial publicitario.

Este modelo puede generar acciones como:

- **Impulsar**.
- **Ajustar antes de impulsar**.
- **Monitorear**.
- **No impulsar**.
- **Enviar a revisión humana**.

La decisión se basa en una combinación de métricas, señales textuales, visuales, OCR, transcripción y variables operativas.

### 9. XGBoost para estimación de pauta

El prototipo puede incluir un segundo modelo basado en **XGBoost** para estimar resultados relacionados con pauta pagada.

La lógica es:

1. Primero se ejecuta el modelo principal.
2. Si el video supera el umbral definido de potencial publicitario, se activa el análisis XGBoost.
3. XGBoost estima posibles resultados de pauta.

Este módulo puede estimar rendimiento esperado con publicidad, CPM probable, impresiones estimadas, eficiencia aproximada por dólar invertido y nicho o categoría probable del anuncio.

Si el modelo XGBoost no está disponible, el sistema utiliza un CPM manual como respaldo.

### 10. Recomendaciones con LLM o reglas

El sistema genera recomendaciones estratégicas usando Gemini, un modelo local open source si está disponible, o reglas internas como fallback.

Las recomendaciones explican por qué conviene o no impulsar, qué ajustar antes de pautar, qué mejorar en el guion, qué mejorar visualmente, qué riesgos revisar y cómo interpretar gráficos y métricas.

---

## 🖥️ ¿Cómo se usa?

### Opción 1: Cargar un video MP4

1. Abre la aplicación.
2. Sube un video en formato MP4.
3. Verifica que dure máximo **1 minuto**.
4. Completa los campos disponibles: título, descripción, categoría o tema, visualizaciones, likes, comentarios, shares, retención estimada, presupuesto de pauta y CPM estimado.
5. Selecciona el motor LLM: `auto`, `gemini`, `local_open_source` o `rules`.
6. Ejecuta el análisis.
7. Revisa las pestañas de resultados.

### Opción 2: Analizar una URL de YouTube

1. Copia la URL del video.
2. Pégala en el campo correspondiente.
3. Si la API de YouTube está configurada, la app puede recuperar metadata.
4. El sistema intentará descargar el video para analizarlo.
5. Si la descarga falla, sube el MP4 manualmente.

### Opción 3: Usar datos manuales

Si no se puede procesar el video completo, puedes ingresar manualmente transcripción, texto visible en pantalla, comentarios, métricas públicas, información del canal, presupuesto y CPM.

Esto permite hacer un análisis parcial aunque algunos módulos automáticos no estén disponibles.

---

## 📊 ¿Qué resultados entrega?

### Resumen ejecutivo

Presenta una lectura clara del caso:

- Acción recomendada.
- Potencial del video.
- Razones principales de la decisión.
- Riesgos detectados.
- Recomendaciones inmediatas.
- Lectura del modelo principal.
- Lectura del módulo XGBoost, si aplica.

### Métricas

Muestra visualizaciones, likes, comentarios, engagement, shares, retención, tiempo desde publicación, views por hora y métricas públicas del canal si están disponibles.

### Transcripción

Incluye texto transcrito, fuente de la transcripción, cantidad de palabras, interpretación del guion, fortalezas, debilidades y mejoras sugeridas.

### OCR

Incluye texto detectado en pantalla, texto corregido o interpretado, cobertura de texto en frames, densidad visual, relevancia del texto y detección de CTA, promoción, urgencia o confianza.

### Análisis visual

Incluye score global de composición, regla de tercios, composición áurea, balance geométrico, claridad focal, contraste, complejidad visual y recomendaciones visuales.

### Políticas

Incluye nivel de riesgo, estado estimado, categorías sensibles, explicación del riesgo y recomendaciones antes de pautar.

### Sentimiento

Incluye distribución positiva, neutra y negativa, gráficos de sentimiento, palabras frecuentes y lectura general del público.

### Gráficos

Incluye visualizaciones del diagnóstico general, proyección de pauta, riesgo publicitario, XGBoost de pauta e interpretación textual de cada gráfico.

### JSON / API

Incluye la salida completa en formato JSON para depuración, revisión técnica, integración con otros sistemas y trazabilidad académica.

---

## ⚙️ Instalación local

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Ejecuta la aplicación:

```bash
python app.py
```

Luego abre la URL local generada por Gradio.

---

## 🚀 Despliegue en Hugging Face Spaces

La estructura mínima recomendada es:

```text
app.py
README.md
requirements.txt
packages.txt
src/
models/
demo_cache/
docs/
tests/
```

Archivos recomendados dentro de `models/`:

```text
best_model.joblib
metric_regressors.joblib
xgboost_paid_ads.joblib
xgboost_paid_ads_metadata.json
```

Dependencias del sistema recomendadas en `packages.txt`:

```text
ffmpeg
tesseract-ocr
```

Secrets recomendados en Hugging Face Spaces:

```text
GEMINI_API_KEY
YOUTUBE_API_KEY
```

Estas claves deben configurarse como **Secrets**, no subirse dentro del código ni en archivos `.env`.

---

## 📁 Estructura general del proyecto

```text
YouTube-AI-Recomendations/
├── app.py
├── README.md
├── requirements.txt
├── packages.txt
├── src/
│   ├── analytics_viz.py
│   ├── comment_sentiment.py
│   ├── exec_summary.py
│   ├── features.py
│   ├── llm_provider.py
│   ├── ocr_video.py
│   ├── paid_ads_xgboost.py
│   ├── predict.py
│   ├── script_analyzer.py
│   ├── transcription.py
│   ├── video_processing.py
│   ├── visual_composition.py
│   └── youtube_api.py
├── models/
│   ├── best_model.joblib
│   ├── metric_regressors.joblib
│   ├── xgboost_paid_ads.joblib
│   └── xgboost_paid_ads_metadata.json
├── demo_cache/
├── docs/
├── tests/
└── notebooks/
```

---

## 🔬 Alcance académico

Este proyecto fue desarrollado como prototipo experimental para investigación aplicada sobre análisis de videos orgánicos y decisión preliminar de pauta publicitaria.

Integra aprendizaje automático supervisado, procesamiento de lenguaje natural, OCR, transcripción automática, análisis de sentimiento, análisis visual, evaluación preliminar de riesgo publicitario, estimación de rendimiento mediante XGBoost y recomendaciones generadas por LLM o reglas.

El objetivo es demostrar que una arquitectura multimodal puede apoyar la evaluación de contenido antes de invertir presupuesto publicitario.

---

## ⚠️ Limitaciones

- No garantiza viralidad.
- No garantiza ventas.
- No predice ROI causal real.
- No reemplaza la revisión oficial de YouTube Ads.
- No accede a métricas privadas de YouTube Studio.
- La descarga automática de YouTube puede fallar según el entorno.
- La calidad del análisis depende del audio, video, metadata y texto disponible.
- El modelo XGBoost debe recalibrarse con campañas reales para uso comercial.
- Los resultados deben interpretarse como apoyo a decisión, no como verdad absoluta.

---

## 🧪 Reentrenamiento

Para reentrenar modelos o extender el prototipo, usa los scripts y notebooks del repositorio académico completo.

Ejemplo para reentrenar XGBoost:

```bash
pip install -r requirements-train.txt
python scripts/train_xgboost_paid_ads.py
```

---

## 🧾 Uso recomendado

Este software debe usarse como herramienta de análisis preliminar en contextos académicos o experimentales.

Antes de usarlo en producción se recomienda validar modelos con datos propios, revisar cumplimiento legal y de privacidad, evaluar sesgos del sistema, medir resultados con campañas reales, documentar supuestos metodológicos y ajustar reglas de política publicitaria.

---

## 👤 Autor

Proyecto académico: **YouTube AI Recomendations**

Prototipo de análisis multimodal para videos de YouTube y decisión preliminar de impulso publicitario.
