# Model Card — YouTube Boost AI

## Resumen

Modelo predictivo de **potencial orgánico relativo** para videos de YouTube. La salida es una probabilidad calibrada que entra como **una señal más** en un score híbrido junto con métricas operativas (engagement, retención, share rate, views/hora) y un evaluador de políticas de YouTube Ads.

**El modelo no decide la acción final.** La acción la decide el recomendador a partir del score híbrido + riesgo publicitario.

## Detalles

| Campo | Valor |
|---|---|
| Tipo | Clasificación binaria (boost_candidate ∈ {0, 1}) |
| Algoritmo seleccionado | LogisticRegression (con TF-IDF + flags + categoría + duración) |
| Algoritmos comparados | LogisticRegression, LinearSVC calibrado, RandomForest, HistGradientBoosting |
| Criterio de selección | F1 → ROC-AUC → recall (no accuracy) |
| Features de entrada | `text_total` (TF-IDF), `title_len`, `description_len`, `text_total_word_count`, flags textuales (`cta`, `urgency`, `trust`, `promo`, `benefit`, `price`), `duration_seconds`, `category_id` |
| Probabilidad clipeada a | `[0.02, 0.98]` para evitar absolutos |
| Calibración | `CalibratedClassifierCV(method="sigmoid")` para LinearSVC |

## Anti-leakage

**Las features de entrenamiento excluyen deliberadamente las variables que componen el target.** Concretamente NO se usan como predictores:

- `engagement_rate`, `like_rate`, `comment_rate`
- `views_per_day`
- `text_power_score`
- `duration_fit_score`
- `views`, `likes`, `comments` y sus log

Esas variables forman el `organic_performance_score` que define `boost_candidate`. Incluirlas como features producía F1 ≈ 0.99 / AUC ≈ 0.9999 — un modelo que memorizaba su propia etiqueta. Las métricas honestas tras la corrección están en la sección siguiente.

Las variables operativas (`shares`, `retention_rate`, `views_per_hour`) **no son predictores del modelo**; entran en el score híbrido del recomendador.

## Métricas (datos reales, anti-leakage)

| Modelo | F1 | ROC-AUC | Recall | Precision |
|---|---|---|---|---|
| **LogisticRegression** (seleccionado) | **0.521** | 0.733 | 0.567 | 0.482 |
| RandomForest | 0.472 | 0.744 | 0.378 | 0.628 |
| LinearSVC calibrado | 0.425 | 0.735 | 0.301 | 0.722 |
| HistGradientBoosting | 0.390 | 0.755 | 0.255 | 0.826 |

AUC ≈ 0.73-0.76 confirma discriminación moderada por encima del azar. F1 ≈ 0.52 refleja la dificultad real de predecir potencial sin métricas tempranas.

## Uso previsto

- **Sí**: ayudar a un creador o equipo de marketing a evaluar si un video tiene potencial suficiente para asignar presupuesto de pauta, identificar riesgos de políticas de YouTube Ads y obtener una recomendación de acción.
- **No**: tomar decisiones automáticas de pauta sin revisión humana, predecir ingresos publicitarios reales, evaluar contenido para audiencias menores de edad.

## Limitaciones

- **Sesgo de selección**: ambos datasets son de videos en *trending*. El modelo distingue "muy buenos" de "razonablemente buenos", no "buenos" de "malos".
- **`category_id` parcialmente unknown** (~76% del corpus).
- **`duration_seconds=0`** en el subset de datasnaek.
- El modelo nunca vio datos de gasto publicitario real.

## Cuándo regenerar

```bash
python -m src.train
```

Genera `models/best_model.joblib`, `models/model_metadata.json` y `outputs/model_comparison.csv`.

## Contacto

Proyecto integrador, Maestría en IA. Repositorio: ver README.md.

## Capa adicional: análisis de guion para pauta

La versión final incorpora `src/script_analyzer.py`, una capa semántica que analiza título, descripción y transcripción del video. Esta capa no reemplaza al modelo predictivo; complementa la recomendación final cuando el contenido podría promocionarse como anuncio.

Entradas principales:

- título/copy;
- descripción;
- transcripción;
- categoría o tema;
- duración.

Salidas principales:

- `script_quality_score`;
- `hook_score`;
- `clarity_score`;
- `value_proposition_score`;
- `cta_score`;
- `policy_claim_risk`;
- fortalezas y debilidades;
- mejoras sugeridas;
- gancho, CTA y copy sugeridos.

La decisión final puede cambiar de `IMPULSAR` a `AJUSTAR ANTES DE IMPULSAR` cuando el modelo detecta buen potencial, pero el guion tiene calidad inferior a 70/100. Esta decisión se adopta para evitar que un contenido con señales cuantitativas positivas se promocione sin una revisión mínima del mensaje.

## Uso de LLM local

El sistema no depende de Gemini. El modo predeterminado usa reglas deterministas. Existe soporte opcional para modelos open source pequeños, como Qwen2.5 o SmolLM2, cargados desde Hugging Face si el usuario decide instalarlos. En la demo pública, el modo recomendado es `rules` porque no depende de credenciales ni de GPU.
