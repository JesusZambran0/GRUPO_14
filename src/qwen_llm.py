"""Utilidades opcionales para Qwen + LoRA/QLoRA.

Este módulo no se importa desde la app principal por defecto. Sirve para pruebas
de inferencia del adaptador entrenado en ``models/qwen_marketing_qlora``.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple

MODEL_ID = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LORA_PATH = Path(os.getenv("LOCAL_LORA_PATH", "models/qwen_marketing_qlora"))


@lru_cache(maxsize=1)
def load_qwen_for_inference():
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)

    if LORA_PATH.exists() and (LORA_PATH / "adapter_config.json").exists():
        try:
            from peft import PeftModel  # type: ignore
            model = PeftModel.from_pretrained(model, str(LORA_PATH))
        except Exception:
            pass

    model.eval()
    return tokenizer, model


def generate_marketing_text(prompt: str, max_new_tokens: int = 280) -> str:
    import torch  # type: ignore
    tokenizer, model = load_qwen_for_inference()
    messages = [
        {"role": "system", "content": "Eres un analista senior de marketing en español. Responde sin JSON, con recomendaciones accionables."},
        {"role": "user", "content": prompt},
    ]
    try:
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    except TypeError:
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        inputs = {"input_ids": input_ids}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.35, top_p=0.9, repetition_penalty=1.08, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
