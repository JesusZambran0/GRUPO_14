"""Precarga el LLM local pequeño la primera vez (opcional).

La demo funciona sin LLM local (cae a rule_based_recommendation). Este script
solo es necesario si quieres habilitar el modo `local` en la app.

Uso:
    python scripts/download_local_llm.py
    # o
    LLM_LOCAL_MODEL_ID=HuggingFaceTB/SmolLM2-1.7B-Instruct python scripts/download_local_llm.py

Modelos recomendados:
    - Qwen/Qwen2.5-1.5B-Instruct        (~3 GB, default)
    - HuggingFaceTB/SmolLM2-1.7B-Instruct  (~3.5 GB)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    model_id = os.getenv("LLM_LOCAL_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct").strip()
    print(f"[llm] Descargando '{model_id}' (puede tomar varios minutos)...")
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"[llm] transformers/torch no están instalados: {exc}")
        print("       Ejecuta: pip install transformers torch accelerate sentencepiece")
        return 1
    try:
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForCausalLM.from_pretrained(model_id)
        print(f"[llm] OK: '{model_id}' disponible en caché local.")
        print("[llm] Ahora puedes correr la app con LLM_LOCAL_OFFLINE=1 y elegir modo 'local' o 'auto'.")
        return 0
    except Exception as exc:
        print(f"[llm] Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
