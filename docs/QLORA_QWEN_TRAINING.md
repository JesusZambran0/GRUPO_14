# QLoRA básico de Qwen para recomendaciones de marketing en español

Este repositorio incluye un flujo reproducible para entrenar un adaptador LoRA/QLoRA pequeño sobre Qwen, enfocado en redactar recomendaciones de marketing para videos de YouTube.

## Qué se entrena

El adaptador no reemplaza el pipeline de datos. Solo aprende el estilo de redacción de:

- recomendación final;
- lectura estratégica de métricas;
- interpretación de sentimiento;
- riesgos de pauta;
- mejoras de guion, OCR y composición visual.

El OCR, sentimiento, metadata, transcripción, métricas y gráficas siguen calculándose en Python.

## Por qué no va dentro del Space principal

El Space principal debe ser estable para demo. Entrenar QLoRA necesita `torch`, `transformers`, `peft`, `trl`, `datasets` y normalmente GPU. Si metes todo eso en el Space de demo, la build se vuelve lenta y aumenta el riesgo de caída.

## Datos incluidos

- `data/qlora/marketing_recommendations_es.csv`: dataset sintético pequeño en español, con columnas `input` y `output`.
- `data/qlora/train.jsonl` y `data/qlora/val.jsonl`: formato conversacional listo para TRL/SFTTrainer.

Puedes ampliar el CSV con ejemplos reales de tus campañas o con datasets públicos de sentimiento/social media. Para recomendación de marketing, el formato input→output personalizado suele funcionar mejor que un dataset genérico.

## Entrenamiento rápido recomendado

En Colab, Kaggle o un Space de entrenamiento con GPU:

```bash
pip install -r requirements-train.txt
python scripts/build_qlora_dataset.py
python scripts/train_qlora_qwen.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --train_file data/qlora/train.jsonl \
  --eval_file data/qlora/val.jsonl \
  --output_dir models/qwen_marketing_qlora \
  --epochs 1 \
  --max_steps 80
```

El resultado exportado será:

```text
models/qwen_marketing_qlora/
  adapter_config.json
  adapter_model.safetensors
  tokenizer files...
```

## Usarlo en la app

Copia la carpeta `models/qwen_marketing_qlora` al repo de demo y configura:

```env
ENABLE_LOCAL_LLM=true
LOCAL_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
LOCAL_LORA_PATH=models/qwen_marketing_qlora
LLM_LOCAL_MAX_NEW_TOKENS=260
```

Para no hacer lenta la build, instala dependencias locales solo cuando de verdad vayas a usar el LLM open source:

```bash
pip install -r requirements-llm.txt
```

## App de entrenamiento

Incluí `app_qlora_train.py`. Puedes desplegarlo como un Space separado con GPU o ejecutarlo en Colab/Kaggle:

```bash
python app_qlora_train.py
```

No uses ese archivo como app principal de la demo.
