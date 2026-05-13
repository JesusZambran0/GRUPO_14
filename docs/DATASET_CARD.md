# Dataset Card — YouTube Boost AI

## Datasets utilizados

### 1. `daily_trending_videos.csv` (Kaggle)
- **Origen:** scrap diario de YouTube Trending por región.
- **Filas:** 674.141 crudas.
- **Cobertura:** 92 países, periodo 2025-01 → 2025-07.
- **Columnas:** `video_id, title, channel, country, views, likes, comments, published_at, fetch_date`.
- **Limitaciones:** sin `description`, sin `tags`, sin `category_id`, sin `duration_seconds`. Cada `video_id` se repite varias veces (uno por día en trending).

### 2. `youtube_trending_in_kaggle_datasnaek_2017_2018.csv` (Kaggle datasnaek/youtube-new)
- **Origen:** dataset clásico de Kaggle.
- **Filas:** 37.352 crudas.
- **Cobertura:** India (IN), 2017-2018.
- **Columnas:** `video_id, title, description, tags, category_id, views, likes, dislikes, comments, published_at, region, country, ...`.
- **Limitaciones:** `duration_seconds=0` siempre. Duplicados por `video_id` para apariciones diarias.

## Pipeline de ingesta

`python -m src.train` ejecuta:

1. **Carga** y concatena ambos CSV → 711.493 filas crudas.
2. **Normaliza** columnas con `src.features.normalize_training_dataframe` (vectorizado, ~20s).
3. **Deduplica** por `video_id`, máximo `views` → **68.526 filas únicas**.
4. **Etiqueta** `boost_candidate` por percentil 70 del `organic_performance_score`.
5. Persiste dataset procesado en `data/processed/youtube_training_dataset.csv`.
6. Genera EDA en `outputs/eda_summary.{md,json}`.

## Variables disponibles

| Variable | Tipo | ¿Predictor del modelo? | ¿Usada en score híbrido? |
|---|---|---|---|
| `title` | str | sí (TF-IDF) | — |
| `description` | str | sí (TF-IDF) | — |
| `tags` | str | sí (TF-IDF) | — |
| `category_id` | str | sí | — |
| `duration_seconds` | int | sí | — |
| `cta_flag, urgency_flag, trust_flag, promo_flag, benefit_flag, price_flag` | int | sí | — |
| `title_len, description_len, text_total_word_count` | int | sí | — |
| `views`, `likes`, `comments` | int | **NO** (anti-leakage) | sí (engagement) |
| `engagement_rate` | float | **NO** (anti-leakage) | sí |
| `views_per_day`, `views_per_hour` | float | **NO** (anti-leakage) | sí |
| `shares`, `retention_rate`, `average_watch_time` | float | **NO** (no en dataset) | sí (entrada manual usuario) |
| `hours_since_publication` | float | **NO** | sí (entrada manual) |
| `followers_count`, `avg_channel_reach` | float | **NO** | informativos |

> **Importante:** `shares` y `retention_rate` **no están en los datasets crudos** y deben ingresarse manualmente desde YouTube Analytics. Se usan únicamente en el score híbrido del recomendador, no como predictores del modelo.

## Variable objetivo

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

**Proxy de potencial orgánico relativo, no ROI causal.**

## Sesgos conocidos

- **Sesgo de selección.** Ambos datasets son de videos que estuvieron en trending → todos son razonablemente exitosos. El modelo no aprende a separar "bueno" de "malo", solo "muy bueno" de "bueno".
- **Sesgo geográfico.** datasnaek es exclusivamente India; daily_trending es global 2025. La mezcla es desigual.
- **Sesgo temporal.** datasnaek es 2017-2018; daily_trending es 2025. El lenguaje, tendencias y plataforma han cambiado.
- **`category_id` 76% unknown**, casi siempre de daily_trending (que no la trae).

## Privacidad

Los datasets son públicos de Kaggle. No contienen información personal sensible: solo metadatos de videos públicos (título, descripción, métricas agregadas).

## Licencia / términos

Los datasets se distribuyen bajo los términos de Kaggle. Verifica la página del dataset original antes de redistribuir.

## Verificación rápida

```bash
python -c "import json; m=json.load(open('models/model_metadata.json')); print(m['dataset_kind'], m['rows'], 'rows')"
# → real_kaggle 68526 rows
```

## Dataset adicional de transcripciones

Se incorporó el dataset público `YouTube Trending Videos Transcripts (900+)`, entregado como archivos JSON de transcripciones. En la preparación local se procesaron 902 transcripciones válidas. Este dataset se utiliza exclusivamente para construir patrones de referencia del guion, no para prometer viralidad ni para sustituir el modelo de rendimiento.

Archivos incluidos o generados:

```text
data/raw/youtube_transcripts_900/Transcripts/*.json
data/processed/youtube_transcripts_script_features.csv
data/processed/script_patterns_reference.json
```

Variables derivadas principales:

- cantidad de palabras de la transcripción;
- primeras 30 palabras del guion;
- presencia de pregunta o beneficio en el gancho;
- conteo de CTA;
- señales de urgencia;
- señales de confianza;
- promesas exageradas;
- puntaje de claridad;
- puntaje de calidad de guion.

## Fuente complementaria de Hugging Face

El proyecto deja documentada una fuente adicional para ampliar transcripciones:

```python
from datasets import load_dataset

dataset = load_dataset("ZelonPrograms/Youtube")
print(dataset["train"][0])
```

Esta fuente no se descarga durante la demo pública para evitar dependencia de internet. Puede incorporarse en una iteración posterior o en un entrenamiento local ampliado.
