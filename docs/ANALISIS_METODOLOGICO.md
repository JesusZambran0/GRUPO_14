# Análisis Metodológico — YouTube Boost AI

Documento de defensa para el proyecto integrador de maestría en IA.

## 1. Alcance actual

**El sistema analiza únicamente videos de YouTube.** No incluye otras redes sociales y no se delimita a un sector vertical específico (no es inmobiliario). El objetivo es predecir el potencial de rendimiento de un video y sugerir si conviene impulsarlo con pauta.

### Salidas obligatorias del sistema

1. Predicción de rendimiento esperado (`bajo / medio / alto / muy_alto`).
2. Probabilidad o score asociada a esa predicción.
3. Métricas calculadas (engagement, views/día, etc.).
4. Recomendación canónica (`impulsar / no impulsar / ajustar antes de impulsar / monitorear`).
5. Lista de ajustes sugeridos sobre el video.
6. **Estimación parametrizada** de alcance por dólar (CPM × multiplicador de potencial).
7. Explicación basada en transcripción, OCR y métricas.

## 2. Justificación del cambio de alcance

**Por qué se abandonó el sector inmobiliario.** Los datasets públicos disponibles del sector inmobiliario:

- mezclan listings con anuncios pagos sin etiquetas claras de gasto;
- carecen de métricas comparables de rendimiento por video;
- presentan sesgos geográficos fuertes que dificultan la generalización;
- raramente incluyen contenido audiovisual procesable para OCR/transcripción;
- no permiten construir una etiqueta supervisada confiable de "vale la pena impulsar".

**Por qué YouTube específicamente.** YouTube ofrece ventajas que sostienen un proyecto integrador:

- API pública (YouTube Data API v3) con metadatos estandarizados.
- Datasets abiertos en Kaggle con miles de videos etiquetados (trending, regional).
- Métricas comparables entre canales (views, likes, comments, duración).
- Contenido visual y sonoro real, que justifica OCR + transcripción.
- Casos de uso de pauta concretos y económicamente significativos.

**Por qué solo YouTube y no multi-red.** Cada red tiene métricas, formatos y APIs distintas. Construir un sistema multi-red comparable requeriría normalización de criterios de éxito sin un estándar consensuado, lo que comprometería la rigurosidad del modelo predictivo. Limitar el alcance a YouTube permite una evaluación más limpia.

## 3. Arquitectura

```
URL/MP4/Datos manuales
        ↓
[YouTube Data API v3] ──► metadata (snippet, contentDetails, statistics)
        ↓
[Whisper / Gemini / manual] ──► transcripción
        ↓
[EasyOCR / pytesseract] ──► OCR de frames seleccionados
        ↓
[features.py] ──► engagement_rate, views_per_day, text_power_score, ocr_word_count, …
        ↓
[predict.py] ──► modelo entrenado (boost_candidate) o fallback heurístico
        ↓
[recommender.py] ──► reglas → recomendación + ajustes + alcance/$
        ↓
[llm_analyzer.py] ──► LLM o reglas locales para análisis semántico-sintáctico
        ↓
Salida JSON contrato
```

**Importante sobre las fuentes de entrada:**

- Una **URL de YouTube** alimenta solo la rama de **metadatos** (vía YouTube Data API v3). La API pública **no entrega el archivo de video**, por lo que la URL sola no permite OCR ni transcripción.
- Para que el sistema procese **OCR del texto visible** o **transcripción del audio**, el usuario debe **cargar el MP4** en la app (o pegar manualmente la transcripción/OCR).
- Esta separación es deliberada y se documenta en la UI y en la API para evitar expectativas incorrectas durante la defensa.

## 4. Datasets

### 4.1 Datasets reales actualmente integrados

`data/raw/` contiene dos datasets de Kaggle (711.493 filas crudas, 68.526 únicas tras dedup por `video_id`):

1. **`daily_trending_videos.csv`** — Kaggle, scrap diario de trending YouTube por región. 92 países, 2025-01 a 2025-07. 674.141 filas. Sin description/tags/category_id/duration. Contribuye con muchos videos globales.
2. **`youtube_trending_in_kaggle_datasnaek_2017_2018.csv`** — Kaggle `datasnaek/youtube-new`. 37.352 filas, India 2017-2018. Trae description, tags, category_id, region, dislikes. Es el dataset rico en señales textuales.

`models/model_metadata.json` registra `"dataset_kind": "real_kaggle"` y la lista de archivos fuente. Ver `docs/DATASETS.md` para detalles.

### 4.2 Sesgo de selección reconocido

**Ambos datasets contienen exclusivamente videos que estuvieron en trending.** El modelo aprende a distinguir videos "muy fuertes" de videos "razonablemente fuertes", **no** "buenos" de "malos". Esto se documenta abiertamente en defensa.

Implicaciones:

- Las probabilidades tienden a ser altas incluso para inputs débiles.
- La regla del recomendador compensa marcando "ajustar antes de impulsar" cuando faltan señales textuales clave (CTA, beneficio, urgencia).
- Para entrenar con la distribución completa de YouTube se requeriría una muestra aleatoria, que no está disponible públicamente.

### 4.3 Dataset sintético (descontinuado)

Existió `data/raw/demo_youtube_public_training.csv` para validación funcional sin red. Fue retirado al integrar los datasets reales. El generador `scripts/generate_demo_dataset.py` se conserva por si alguien necesita un sintético en CI.

## 5. Variable objetivo

Se construye como un **score compuesto normalizado por percentil**:

```
organic_performance_score =
    0.35 · views_per_day_norm
  + 0.25 · engagement_rate_norm
  + 0.15 · like_rate_norm
  + 0.10 · comment_rate_norm
  + 0.10 · text_power_score
  + 0.05 · duration_fit_score
```

`boost_candidate = 1` si el score supera el percentil 70 (clase positiva ≈ 30%).

**Limitación reconocida:** la etiqueta es **proxy del potencial orgánico relativo**, no del retorno real de pauta.

## 6. Anti-leakage de features (cambio metodológico clave)

**Problema detectado.** En una versión inicial, las features de entrenamiento incluían `engagement_rate`, `views_per_day`, `like_rate`, `comment_rate`, `text_power_score` y `duration_fit_score` — exactamente las **6 variables que componen el target**. El modelo memorizaba la fórmula del score y obtenía F1 ≈ 0.99 / AUC ≈ 0.9999, números que parecían excelentes pero eran artificiales.

**Decisión metodológica.** El set de features de entrenamiento (`src/train.py::NUMERIC_COLS`) **excluye deliberadamente** las variables que componen el score. El modelo predice usando únicamente:

- `text_total` (TF-IDF de title + description + tags),
- `title_len`, `description_len`, `text_total_word_count`,
- `cta_flag`, `urgency_flag`, `trust_flag`, `promo_flag`, `benefit_flag`, `price_flag` (flags léxicos),
- `duration_seconds` (nominal, no normalizada),
- `category_id` (one-hot).

**Justificación.** Esta es la situación real de un creador evaluando un video **antes de publicarlo** o decidiendo si lo impulsa **sin fiarse de métricas tempranas inestables**. Las métricas de engagement no están disponibles antes de publicar, y son ruidosas en las primeras horas.

**Resultado.** F1 cae a 0.52, AUC a 0.73. Estos números son honestos y reflejan la dificultad real del problema.

## 7. Modelos comparados

Cuatro familias, todas con sklearn (sin dependencias pesadas):

| Modelo | Fortaleza | F1 actual | AUC actual |
|---|---|---|---|
| LogisticRegression + TF-IDF | Baseline lineal probabilístico | **0.521** | 0.733 |
| RandomForest | No linealidad y robustez | 0.472 | 0.744 |
| LinearSVC + CalibratedClassifierCV | Margen máximo en texto disperso | 0.425 | 0.735 |
| HistGradientBoostingClassifier | Boosting basado en histogramas (opt-in) | 0.390 | 0.755 |

**Criterio de selección.** F1 → ROC-AUC → recall (no accuracy). Esta priorización refleja que el costo de pasar por alto un buen video (FN) es mayor que sobre-recomendar (FP) en una herramienta de apoyo.

`outputs/model_comparison.csv` registra todas las métricas y `models/model_metadata.json` documenta el modelo elegido (LogisticRegression con la configuración actual del dataset).

## 8. OCR

**No se entrena un OCR propio.** Se usan modelos preentrenados con caché por hash de archivo:

- **EasyOCR** (principal, multi-idioma).
- **pytesseract** (fallback automático si EasyOCR falla).
- Bloque vacío con advertencia si ambos fallan, sin tirar la app.

**Selección de frames adaptativa.** `select_frame_timestamps()` cambia el paso según el tipo de video:

- video con narrador o duración > 8 s → paso 4 s.
- video sin narrador o duración ≤ 8 s → paso 0.75 s.
- tope global `MAX_OCR_FRAMES` (20 por defecto) con submuestreo uniforme.

**Señales que extrae el OCR.** Conteo de palabras, cobertura de frames, densidad visual, flags de CTA, promoción, urgencia y confianza.

## 9. Transcripción

Estrategia con tres niveles de fallback:

1. Texto manual si el usuario lo proporciona.
2. **Whisper local** (`base`) si está instalado, con extracción de audio vía MoviePy.
3. Mensaje de fallback que permite continuar; la app **no se cae** si falta audio o si Whisper no está.

Caché por hash en `.cache/transcription/` evita reprocesar el mismo video.

## 10. LLM como capa auxiliar

El LLM **no es el predictor principal**. Recibe el contexto completo (textos + métricas + salida del modelo) y devuelve un JSON con análisis semántico-sintáctico:

- `claridad_mensaje` (0-100)
- `fuerza_cta` (0-100)
- `coherencia_semantica` (0-100)
- `complejidad_sintactica` (`baja/media/alta`)
- `tipo_contenido` (educativo, promocional, testimonial, …)
- `riesgo_comunicacional` (`bajo/medio/alto`)
- `fortalezas`, `debilidades`, `ajustes_sugeridos`
- `recomendacion_llm`
- `justificacion`

**Prioridad:** Gemini → Ollama → reglas locales deterministas. La función `_coerce_to_contract` garantiza que la salida cumpla el contrato aunque el LLM devuelva JSON parcial.

## 11. Alcance por dólar (parametrizado)

```
alcance_base_por_dolar = 1000 / CPM
multiplicador_potencial = {bajo: 0.70, medio: 1.00, alto: 1.30, muy_alto: 1.60}
alcance_estimado_por_dolar = base × multiplicador
alcance_estimado_total = alcance_estimado_por_dolar × presupuesto
```

**No es ROI causal.** Es una estimación de impresiones bajo un CPM hipotético, ajustada por el potencial predicho. Se documenta en cada salida (`nota_metodologica`) y en el resumen ejecutivo.

## 12. Limitaciones

- **La etiqueta `boost_candidate` es proxy de potencial orgánico**, no de retorno real.
- **Sesgo de selección.** Los datasets son de videos en *trending*. El modelo aprende a distinguir "muy buenos" del resto, no "buenos" de "malos". Las probabilidades tienden a ser altas; la regla del recomendador compensa.
- **`category_id` parcialmente desconocido.** ~76% de las filas tienen `category_id="unknown"` porque `daily_trending_videos.csv` no la trae. El modelo aprende del texto principalmente.
- **`duration_seconds` ausente en datasnaek.** El archivo de Kaggle datasnaek tiene duración=0 para todas las filas.
- **Dataset histórico parcial.** Datasnaek es de 2017-2018 (India), daily_trending es de 2025 (global). El idioma y la mezcla cultural varían.
- **F1 ≈ 0.52 / AUC ≈ 0.73** son el techo realista de este pipeline sin métricas tempranas. El AUC > 0.5 confirma que el modelo discrimina mejor que azar.
- Las **probabilidades de inferencia se recortan a `[0.02, 0.98]`** (`PROBABILITY_FLOOR` / `PROBABILITY_CEILING` en `src/predict.py`).
- Los multiplicadores de potencial (`0.70 / 1.00 / 1.30 / 1.60`) son **convencionales** y deben calibrarse con datos reales de pauta.
- OCR y transcripción dependen de calidad de imagen y audio, y **requieren MP4** (no se obtienen de la URL).
- El modelo no fue entrenado con datos de gasto publicitario, solo con métricas orgánicas.
- La capa LLM puede alucinar si no se valida el JSON; mitigado con `_coerce_to_contract`.

## 13. Trabajo futuro

- Sustituir el dataset sintético por datasets públicos reales de Kaggle.
- Calibrar los multiplicadores de potencial con experimentos A/B reales.
- Añadir explicabilidad SHAP en `notebooks/03_model_explainability.ipynb`.
- Soporte para Whisper API remoto cuando no hay GPU local.
- Comparación con XGBoost/LightGBM en el modo `requirements-full.txt`.
- Integración con YouTube Studio API para series temporales de retención.
- Pruebas de fairness por categoría e idioma.
