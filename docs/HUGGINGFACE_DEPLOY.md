# Despliegue público en Hugging Face Spaces

## Objetivo

La profesora debe poder abrir un link público y repetir exactamente lo mostrado en el video de defensa. Para eso se debe desplegar la carpeta de aplicación en Hugging Face Spaces con SDK Gradio.

## Repositorio recomendado para el Space

Usar la versión ligera de despliegue, no la versión completa con datasets crudos grandes. Debe contener:

```text
app.py
requirements.txt
packages.txt
README.md
src/
models/best_model.joblib
models/model_metadata.json
data/processed/script_patterns_reference.json
demo_cache/
docs/
```

Los datasets crudos completos se conservan en la versión académica local como evidencia, pero no son necesarios para que la demo pública funcione.

## Pasos

1. Crear cuenta en https://huggingface.co.
2. Crear un Space nuevo.
3. Elegir SDK: Gradio.
4. Hardware: CPU Basic.
5. Visibilidad: Public.
6. Subir los archivos de la carpeta de despliegue.
7. Verificar que `app.py` quede en la raíz del Space.
8. Esperar el build.
9. Abrir el link público en ventana incógnita.
10. Probar un caso manual con transcripción pegada.

## Configuración segura de demo

- Usar `llm_mode = rules`.
- Usar transcripción manual pegada en el formulario.
- No depender de Gemini.
- No depender de Ollama.
- No ejecutar Whisper en vivo durante la defensa si el Space está en CPU básico.
- OCR y transcripción automática quedan como opciones, no como requisito para la prueba pública.

## Caso manual replicable

Título: `Cómo mejorar tu productividad en 3 pasos`

Descripción: `Tutorial breve para organizar tareas, evitar distracciones y mejorar la concentración diaria.`

Transcripción: `En este video aprenderás tres pasos para mejorar tu productividad: planificar tus tareas, eliminar distracciones y revisar tus avances. Guarda este video y prueba estos consejos hoy.`

Categoría: `educación`

Views: `8500`
Likes: `620`
Comments: `48`
Shares: `120`
Retention rate: `0.46`
Horas desde publicación: `12`
Duración: `45`

## Mensaje metodológico

El sistema no predice viralidad ni ROI. Estima potencial relativo de rendimiento y recomienda acciones de impulso con revisión humana. La evaluación de políticas publicitarias es preliminar y no sustituye la revisión oficial de Google Ads.
