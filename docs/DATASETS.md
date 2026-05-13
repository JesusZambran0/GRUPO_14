# Datasets para YouTube Boost AI

Este documento explica los datasets reales que actualmente entrenan el modelo y cómo agregar más en el futuro.

## 1. Datasets actualmente integrados

`data/raw/` contiene **dos datasets reales** de Kaggle:

### 1.1 `daily_trending_videos.csv`

- **Origen:** scrap diario de YouTube Trending por región.
- **Tamaño:** ~100 MB, **674.141 filas** crudas.
- **Cobertura:** 92 países, periodo 2025-01 → 2025-07.
- **Columnas:** `video_id, title, channel, country, views, likes, comments, published_at, fetch_date`.
- **Limitaciones:** sin `description`, sin `tags`, sin `category_id`, sin `duration_seconds`. Cada `video_id` se repite varias veces (uno por día en trending).

### 1.2 `youtube_trending_in_kaggle_datasnaek_2017_2018.csv`

- **Origen:** Kaggle `datasnaek/youtube-new` (Trending YouTube Video Statistics).
- **Tamaño:** ~67 MB, **37.352 filas** crudas.
- **Cobertura:** India (sólo región IN), periodo 2017-2018.
- **Columnas:** `video_id, title, description, tags, category_id, views, likes, dislikes, comments, published_at, region, country, ...`.
- **Limitaciones:** `duration_seconds=0` siempre (la fuente no la trae); duplicados por `video_id` para apariciones diarias.

## 2. Pipeline de ingesta

Cuando ejecutas `python -m src.train`:

1. **Carga** ambos CSV concatenados → 711.493 filas crudas.
2. **Normaliza** columnas con `src.features.normalize_training_dataframe` (vectorizado, ~20s).
3. **Deduplica** por `video_id`, **quedándose con la fila de máximo views** → 68.526 filas únicas.
4. **Etiqueta** `boost_candidate` por percentil 70 del `organic_performance_score`.
5. Guarda dataset procesado en `data/processed/youtube_training_dataset.csv`.
6. Genera `outputs/eda_summary.{md,json}` con estadísticas reales del corpus.
7. Entrena 3-4 modelos (HGB es opt-in con `ENABLE_HGB=1`).
8. Persiste el mejor en `models/best_model.joblib` y metadata en `models/model_metadata.json` con `"dataset_kind": "real_kaggle"`.

## 3. Verificación

Después de entrenar:

```bash
# Confirmar origen real
python -c "import json; print(json.load(open('models/model_metadata.json'))['dataset_kind'])"
# → real_kaggle

# Ver EDA generado
head -30 outputs/eda_summary.md
```

## 4. Métricas obtenidas

Con percentil del target = 0.70 (clase positiva ≈ 30%):

| Modelo | F1 | ROC-AUC | Recall | Precision |
|---|---|---|---|---|
| LogisticRegression (best) | 0.521 | 0.733 | 0.567 | 0.482 |
| RandomForest | 0.472 | 0.744 | 0.378 | 0.628 |
| LinearSVC calibrado | 0.425 | 0.735 | 0.301 | 0.722 |
| HistGradientBoosting | 0.390 | 0.755 | 0.255 | 0.826 |

> **Nota metodológica importante:** estas métricas son **honestas** porque el set de features de entrenamiento **excluye deliberadamente** las variables que componen el target (`engagement_rate`, `views_per_day`, `text_power_score`, etc.). Sin esta exclusión el modelo tendría F1 ≈ 0.99, pero solo estaría memorizando su propia etiqueta. Ver sección "Anti-leakage" en `docs/ANALISIS_METODOLOGICO.md`.

## 5. Sesgo de selección a declarar en defensa

Ambos datasets contienen únicamente videos que **estuvieron en trending**. El modelo aprende a distinguir "muy buenos" del resto, no "buenos" de "malos". Implicaciones:

- Las probabilidades del modelo tienden a ser altas incluso para inputs débiles.
- La regla del recomendador (`ajustar antes de impulsar` cuando faltan señales) compensa esta tendencia.
- Para una versión sin sesgo se necesitaría una muestra aleatoria de YouTube (no disponible públicamente).

## 6. Cómo agregar más datasets

Cualquier CSV agregado a `data/raw/` se ingiere automáticamente. Las columnas reconocidas (con autorenombrado):

| Columna canónica | Variantes aceptadas |
|---|---|
| `views` | `view_count`, `views_count` |
| `likes` | `like_count`, `likes_count` |
| `comments` | `comment_count`, `comments_count` |
| `published_at` | `publishedAt`, `publish_time` |
| `category_id` | `categoryId` |
| `channel_title` | `channelTitle`, `channel` |

Si tu CSV usa otros nombres, agrégalos al diccionario `rename_candidates` en `src/features.py::normalize_training_dataframe`.

### Datasets adicionales recomendados

| Slug Kaggle | Contenido | Tamaño |
|---|---|---|
| `rsrishav/youtube-trending-video-dataset` | Trending diario, 10+ regiones | ~150 MB |
| `nelgiriyewithana/global-youtube-statistics-2023` | Estadísticas globales de canales top | ~1 MB |

Para descargar automáticamente con credenciales Kaggle:

```bash
pip install kaggle
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
python scripts/download_kaggle_datasets.py
python -m src.train
```

## 7. Subset rápido para iterar

Si quieres entrenar más rápido sobre un subset (estratificado por clase):

```bash
MAX_TRAIN_ROWS=10000 python -m src.train
```

## 8. Cuándo regenerar el modelo

- Tras agregar un nuevo CSV a `data/raw/`.
- Tras modificar `NUMERIC_COLS`, `CATEGORICAL_COLS` o `TEXT_COL` en `src/train.py`.
- Tras modificar la lógica de `normalize_training_dataframe` en `src/features.py`.
- Tras modificar la fórmula de `organic_performance_score` en `create_boost_candidate_target`.
