# Análisis de guion y copy para pauta

El sistema incorpora una capa semántica destinada a evaluar si el texto del video está preparado para funcionar como anuncio. Esta capa analiza título, descripción y transcripción.

## Criterios evaluados

1. Gancho inicial.
2. Claridad del mensaje.
3. Propuesta de valor.
4. Llamado a la acción.
5. Riesgo de promesas exageradas.
6. Tono comunicacional.

## Dataset de apoyo

Se incorporó el dataset `YouTube Trending Videos Transcripts (900+)`, compuesto por archivos JSON de transcripciones de videos populares. En esta versión se procesaron 902 transcripciones válidas.

Archivos generados:

```text
data/processed/youtube_transcripts_script_features.csv
data/processed/script_patterns_reference.json
```

## Dataset complementario citado

Para ampliar el sistema, se deja documentada la fuente de Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("ZelonPrograms/Youtube")
print(dataset["train"][0])
```

Esta fuente puede utilizarse para ampliar el corpus de transcripciones con metadatos y vistas. No se descarga automáticamente en la demo pública para evitar dependencia de internet.

## Límite metodológico

El análisis de guion no garantiza rendimiento publicitario. Sirve para detectar señales textuales que suelen afectar claridad, persuasión, cumplimiento de políticas y preparación creativa antes de promocionar un video.
