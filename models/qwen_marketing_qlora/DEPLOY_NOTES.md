# Qwen Marketing QLoRA adapter

Ubicación esperada por la app:

```text
models/qwen_marketing_qlora/
```

Base model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Variables para activar inferencia local en Hugging Face Spaces:

```env
ENABLE_LOCAL_LLM=true
LOCAL_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
LOCAL_LORA_PATH=models/qwen_marketing_qlora
LLM_LOCAL_MAX_NEW_TOKENS=300
LLM_LOCAL_TEMPERATURE=0.35
```

Dependencias opcionales necesarias para el modo local:

```text
torch
transformers
peft
accelerate
sentencepiece
safetensors
```

No se incluyen carpetas `checkpoint-*`, `optimizer.pt`, `scheduler.pt` ni estados de entrenamiento porque no son necesarios para inferencia y vuelven el repo más pesado.
