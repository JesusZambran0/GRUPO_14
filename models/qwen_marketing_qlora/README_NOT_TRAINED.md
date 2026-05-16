# Adaptador Qwen QLoRA

Esta carpeta es el destino del entrenamiento. Todavía no contiene `adapter_model.safetensors` porque el entrenamiento requiere GPU y dependencias que no están disponibles en esta sandbox.

Ejecuta:

```bash
pip install -r requirements-train.txt
python scripts/build_qlora_dataset.py
python scripts/train_qlora_qwen.py --model_id Qwen/Qwen2.5-0.5B-Instruct --max_steps 80 --epochs 1 --output_dir models/qwen_marketing_qlora
```
