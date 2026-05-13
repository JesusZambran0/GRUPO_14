from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="QLoRA/SFT básico de Qwen para recomendaciones de marketing en español.")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", default="data/qlora/train.jsonl")
    parser.add_argument("--eval_file", default="data/qlora/val.jsonl")
    parser.add_argument("--output_dir", default="models/qwen_marketing_qlora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default="")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    use_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    quant_config = None
    if use_cuda:
        try:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        except Exception:
            quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.eval_file})

    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="steps",
        save_steps=max(20, args.max_steps // 2),
        eval_strategy="steps" if len(ds["validation"]) else "no",
        eval_steps=max(20, args.max_steps // 2),
        max_seq_length=args.max_seq_length,
        packing=False,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id or None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        peft_config=peft_config,
        formatting_func=formatting_func,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adaptador guardado en: {args.output_dir}")


if __name__ == "__main__":
    main()
