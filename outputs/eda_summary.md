# EDA — YouTube Boost AI
## Origen del dataset
- Tipo detectado: **real_kaggle**
- Archivos usados:
  - `daily_trending_videos.csv`
  - `youtube_trending_in_kaggle_datasnaek_2017_2018.csv`

## Dimensiones
- Filas: **68526**
- Columnas: **48**

## Columnas
`video_id`, `title`, `channel`, `country`, `views`, `likes`, `comments`, `published_at`, `fetch_date`, `source_file`, `youtube_url`, `trending_date`, `channel_title`, `category_id`, `tags`, `dislikes`, `thumbnail_link`, `comments_disabled`, `ratings_disabled`, `video_error_or_removed`, `description`, `region`, `duration_seconds`, `source_dataset`, `source_url`, `kaggle_url`, `text_total`, `title_len`, `description_len`, `text_total_word_count`, `engagement_rate`, `like_rate`, `comment_rate`, `log_views`, `log_likes`, `log_comments`, `views_per_day`, `cta_flag`, `urgency_flag`, `trust_flag`, `promo_flag`, `benefit_flag`, `price_flag`, `text_power_score`, `duration_fit_score`, `organic_performance_score`, `boost_candidate`, `target_threshold`

## Calidad
- Duplicados: **0**
- Faltantes detectados:
  - `channel`: 16307
  - `fetch_date`: 16307
  - `youtube_url`: 52219
  - `trending_date`: 52219
  - `channel_title`: 52219
  - `dislikes`: 52219
  - `thumbnail_link`: 52219
  - `comments_disabled`: 52219
  - `ratings_disabled`: 52219
  - `video_error_or_removed`: 52219
  - `region`: 52219
  - `source_dataset`: 52219
  - `source_url`: 52219
  - `kaggle_url`: 52219

## Señales textuales
- Tasa de presencia de CTA: 5.60%
- Tasa de presencia de promo: 0.02%

## Top categorías (share)
- `unknown`: 76.20%
- `24.0`: 11.01%
- `25.0`: 3.66%
- `22.0`: 1.80%
- `10.0`: 1.77%
- `23.0`: 1.63%
- `27.0`: 1.16%
- `1.0`: 0.76%
- `26.0`: 0.70%
- `17.0`: 0.51%
- `28.0`: 0.44%
- `43.0`: 0.18%
- `29.0`: 0.10%
- `2.0`: 0.05%
- `20.0`: 0.03%

## Estadísticas principales
### views
- count: 68526 | mean: 2125376.1813 | std: 7796023.6150
- min: 0.0000 | p25: 180623.2500 | p50: 419863.5000 | p75: 1174300.2500 | max: 316012253.0000
### likes
- count: 68526 | mean: 57746.1859 | std: 219953.2957
- min: 0.0000 | p25: 3062.2500 | p50: 10251.0000 | p75: 33358.0000 | max: 10872921.0000
### comments
- count: 68526 | mean: 1860.6122 | std: 7612.2422
- min: 0.0000 | p25: 137.0000 | p50: 501.0000 | p75: 1520.0000 | max: 807558.0000
### engagement_rate
- count: 68526 | mean: 7.6255 | std: 1274.3873
- min: 0.0000 | p25: 0.0120 | p50: 0.0244 | p75: 0.0469 | max: 243886.0000
### like_rate
- count: 68526 | mean: 7.5426 | std: 1262.7380
- min: 0.0000 | p25: 0.0108 | p50: 0.0227 | p75: 0.0441 | max: 239858.0000
### comment_rate
- count: 68526 | mean: 0.0829 | std: 16.0562
- min: 0.0000 | p25: 0.0003 | p50: 0.0011 | p75: 0.0027 | max: 4028.0000
### views_per_day
- count: 68526 | mean: 168196.2100 | std: 1193162.7979
- min: 0.0000 | p25: 778.9962 | p50: 2446.6243 | p75: 37130.0979 | max: 125432237.0000
### duration_seconds
- count: 68526 | mean: 0.0000 | std: 0.0000
- min: 0.0000 | p25: 0.0000 | p50: 0.0000 | p75: 0.0000 | max: 0.0000
### title_len
- count: 68526 | mean: 59.3674 | std: 24.4689
- min: 1.0000 | p25: 40.0000 | p50: 58.0000 | p75: 80.0000 | max: 100.0000
### description_len
- count: 68526 | mean: 211.8500 | std: 544.1240
- min: 0.0000 | p25: 0.0000 | p50: 0.0000 | p75: 0.0000 | max: 5135.0000
### text_total_word_count
- count: 68526 | mean: 55.6901 | std: 106.6726
- min: 0.0000 | p25: 7.0000 | p50: 11.0000 | p75: 20.0000 | max: 1404.0000

## Variable objetivo `boost_candidate`
- Percentil para etiquetar como candidato: **0.70**
- Umbral del score orgánico: **0.2025**
- Clase `0`: 70.00%
- Clase `1`: 30.00%

## Lectura metodológica
Las variables de YouTube presentan asimetría fuerte; por eso el pipeline aplica `log1p` a views, likes y comments y crea tasas relativas (engagement_rate, like_rate, comment_rate) y `views_per_day`. La etiqueta `boost_candidate` representa el **potencial orgánico relativo** calculado como un score ponderado, **no un ROI causal real**.
