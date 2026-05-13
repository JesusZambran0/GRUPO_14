"""Fine-tuning LoRA OPCIONAL del LLM local para recomendaciones.

ESTE SCRIPT NO ES REQUERIDO PARA LA DEMO. La demo corre con rule_based_recommendation
o con el LLM base sin LoRA. Este script existe solo si alguien quiere experimentar
ajustando el redactor a un estilo específico.

Requisitos:
    pip install peft datasets
    + las dependencias de transformers/torch del requirements-full.txt

Uso:
    python scripts/lora_finetune_optional.py path/to/recommendations.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

print(
    "NOTA: este script es opcional y no es requisito de la demo.\n"
    "La demo académica corre sin LoRA, con rule_based_recommendation como fallback.\n"
    "Si decides usarlo, prepara un dataset JSONL con {prompt, response} y revisa\n"
    "la guía oficial de peft (https://huggingface.co/docs/peft).\n"
    "Este archivo es un placeholder intencional para no añadir complejidad innecesaria."
)
sys.exit(0)
