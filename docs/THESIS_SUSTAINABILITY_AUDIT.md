# Auditoría de sustentabilidad metodológica — YouTube Boost AI

## Veredicto

La tesis **sí es sustentable como sistema de apoyo a decisiones para priorizar impulsos publicitarios**, siempre que el alcance se formule como predicción/estimación y no como garantía causal de ROI. El repositorio reconstruido separa tres capas:

1. **Clasificación de candidatura**: regresión logística sobre señales textuales/metadata públicas para estimar si un video merece ser impulsado.
2. **Evaluación multimodal y de riesgo**: OCR, transcripción, NLP, sentimiento, composición visual, políticas y QLoRA/Qwen como soporte explicativo.
3. **Estimación pagada**: XGBoost de segunda etapa para estimar score de rendimiento pagado y CPM cuando la probabilidad de candidatura supera 51%.

## Evidencia de datos internos

- Dataset principal de predicción: 68,526 filas, 48 columnas, fuentes: daily_trending_videos.csv, youtube_trending_in_kaggle_datasnaek_2017_2018.csv.
- Etiqueta objetivo: `boost_candidate`, derivada de rendimiento orgánico; por eso debe declararse como proxy de potencial publicitario, no como compra real de anuncios.
- Tasa positiva aproximada: 30.00%.
- QLoRA: base `Qwen/Qwen2.5-0.5B-Instruct`, dominio `marketing_youtube_ads_recommendations`, entrenamiento en español, `max_steps=80`.

## Por qué se conserva regresión logística

La comparación de modelos muestra que `hist_gradient_boosting` tuvo más accuracy, pero la regresión logística obtuvo el mejor F1 y mejor recall de los modelos comparados. Para esta tesis el costo principal es dejar pasar videos con potencial, por eso F1/recall pesan más que accuracy.

|   accuracy |   precision |   recall |       f1 |   roc_auc |   tn |   fp |   fn |   tp | model                  |
|-----------:|------------:|---------:|---------:|----------:|-----:|-----:|-----:|-----:|:-----------------------|
|   0.687509 |    0.482324 | 0.567364 | 0.521399 |  0.732783 | 7090 | 2504 | 1779 | 2333 | logistic_regression    |
|   0.746097 |    0.627625 | 0.377918 | 0.471767 |  0.744242 | 8672 |  922 | 2558 | 1554 | random_forest          |
|   0.755509 |    0.722125 | 0.300827 | 0.424721 |  0.735366 | 9118 |  476 | 2875 | 1237 | linear_svc             |
|   0.760397 |    0.825984 | 0.255107 | 0.389818 |  0.755184 | 9373 |  221 | 3063 | 1049 | hist_gradient_boosting |

## XGBoost agregado

El XGBoost es viable como **segunda etapa** porque existen datasets públicos con impresiones, clicks, gasto, conversiones y CPM. El modelo entrenado en este repo usa 8,610 filas de entrenamiento y 2,153 de test.

| target                 |     mae |     rmse |       r2 |   mape_pct |   test_mean |
|:-----------------------|--------:|---------:|---------:|-----------:|------------:|
| paid_performance_score | 8.81276 | 10.9026  | 0.377508 |    19.4576 |     50.607  |
| cpm                    | 4.50506 |  5.91092 | 0.508643 |   149.559  |     10.0111 |

## Límite metodológico crítico

Los datasets disponibles no son datos internos de YouTube Ads de tu cuenta. Por tanto:

- El modelo predice **tendencia y costo estimado**, no ROI garantizado.
- La hipótesis se sostiene si la tesis se redacta como “sistema de recomendación y priorización de inversión publicitaria”, no como “modelo que garantiza rendimiento real de pauta”.
- La validación final debe incluir una prueba A/B o campañas reales pequeñas para calibrar CPM, CTR y conversiones por nicho.

## Recomendación de redacción de tesis

Título sugerido: **Sistema multimodal de recomendación para priorizar videos de YouTube candidatos a impulso publicitario mediante aprendizaje automático, análisis audiovisual y modelos de lenguaje ajustados con QLoRA**.

Pregunta sustentable: ¿En qué medida un sistema multimodal puede estimar la conveniencia de impulsar un video y generar recomendaciones accionables antes de invertir presupuesto publicitario?
