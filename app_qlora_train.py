from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
import gradio as gr

DEFAULT_CMD = "python scripts/build_qlora_dataset.py && python scripts/train_qlora_qwen.py --model_id Qwen/Qwen2.5-0.5B-Instruct --max_steps 80 --epochs 1 --output_dir models/qwen_marketing_qlora"


def run_training(command: str):
    command = command.strip() or DEFAULT_CMD
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logs = []
    for line in proc.stdout or []:
        logs.append(line)
        if len(logs) > 500:
            logs = logs[-500:]
        yield "".join(logs)
    proc.wait()
    logs.append(f"\nProceso terminado con código {proc.returncode}.\n")
    yield "".join(logs)


with gr.Blocks(title="Qwen QLoRA Marketing Trainer") as demo:
    gr.Markdown("""
# Qwen QLoRA Marketing Trainer

Usa este archivo solo en un Space/Colab/Kaggle con GPU. No está pensado para correr dentro del Space principal de la demo.

1. Instala `requirements-train.txt`.
2. Revisa `data/qlora/marketing_recommendations_es.csv`.
3. Ejecuta el comando sugerido.
4. Copia `models/qwen_marketing_qlora/` al repo de la demo y configura `LOCAL_LORA_PATH`.
""")
    cmd = gr.Textbox(label="Comando de entrenamiento", value=DEFAULT_CMD, lines=4)
    out = gr.Textbox(label="Logs", lines=22)
    btn = gr.Button("Entrenar / exportar adaptador", variant="primary")
    btn.click(run_training, inputs=[cmd], outputs=[out])

if __name__ == "__main__":
    demo.launch()
