# Implementación final del repositorio

## Alcance técnico definitivo

El repositorio queda delimitado a YouTube. La solución no busca automatizar presupuesto ni garantizar viralidad; estima potencial relativo de rendimiento, analiza riesgos de pauta y recomienda acciones bajo revisión humana.

## Mejoras incorporadas

1. **Descarga por URL de YouTube**: se implementa `yt-dlp` para descargar videos de hasta 60 segundos en 360p cuando el usuario pega un enlace. Si el video supera el límite o la descarga falla, el sistema bloquea análisis completo y solicita revisión humana o carga de MP4.
2. **Transcripción automática obligatoria**: la ruta crítica usa `ffmpeg` + `faster-whisper`. Si la transcripción no alcanza un mínimo de palabras, no se evalúan políticas ni guion como aptos.
3. **OCR obligatorio**: se analizan frames del video. Si no se detecta texto visual, el sistema lo declara explícitamente.
4. **LLM open source**: se integra `HuggingFaceTB/SmolLM2-135M-Instruct` como redactor opcional desde Hugging Face. La decisión no depende del LLM; el LLM redacta a partir del diagnóstico estructurado.
5. **Predicción de métricas esperadas**: se agrega `metric_regressors.joblib` y una capa de proyección de pauta que estima views, likes, comments y shares esperados según presupuesto, CPM, probabilidad y riesgo.
6. **Análisis visual cinematográfico**: se extraen keyframes y se calculan señales de regla de tercios, contraste, brillo, densidad de bordes, movimiento y carga cognitiva. Las recomendaciones se basan en composición visual y comunicación.
7. **Interfaz moderna**: Gradio usa tema oscuro, tarjetas visuales, pestañas organizadas, gráficos y botones destacados.

## Límite metodológico

La proyección de resultados no es ROI causal. La evaluación de políticas no reemplaza la revisión oficial de YouTube/Google Ads. La detección visual no identifica todos los objetos o contextos posibles; opera como análisis composicional ligero para aportar recomendaciones creativas.

## Comandos validados

```bash
python tests/smoke_test.py
pytest -q
```

Resultado validado en entorno de construcción: 15 pruebas superadas.
