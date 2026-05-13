from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

SYSTEM = "Eres un analista senior de marketing, contenido audiovisual y pauta digital. Responde en español natural, profesional y accionable. No uses JSON."


def row_to_messages(row: dict) -> dict:
    inp = row.get("input") or row.get("prompt") or row.get("entrada") or ""
    out = row.get("output") or row.get("response") or row.get("respuesta") or ""
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": inp}, {"role": "assistant", "content": out}]}


def main():
    ap = argparse.ArgumentParser(description="Convierte CSV input/output a JSONL conversacional para QLoRA/SFT.")
    ap.add_argument("--csv", default="data/qlora/marketing_recommendations_es.csv")
    ap.add_argument("--out_dir", default="data/qlora")
    ap.add_argument("--val_ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(src.open("r", encoding="utf-8")))
    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio)) if len(rows) > 8 else 1
    val, train = rows[:n_val], rows[n_val:]

    for name, split in [("train.jsonl", train), ("val.jsonl", val)]:
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row_to_messages(row), ensure_ascii=False) + "\n")
    print(f"OK: {len(train)} train / {len(val)} val en {out_dir}")


if __name__ == "__main__":
    main()
