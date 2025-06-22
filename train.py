import os
import sys
from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict
import argparse
import json


def tokenize_fn(tokenizer, prompt, max_length: int):
    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    # append eos if missing
    if tokens["input_ids"][-1] != tokenizer.eos_token_id and len(tokens["input_ids"]) < max_length:
        tokens["input_ids"].append(tokenizer.eos_token_id)
        tokens["attention_mask"].append(1)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def generate_prompt(data_point: dict) -> str:
    instr = data_point.get("instruction", "")
    inp = data_point.get("input", "")
    out = data_point.get("output", "")
    if inp:
        return f"""
Below is an instruction paired with an input. Write a response that completes the request.

### Instruction:
{instr}

### Input:
{inp}

### Response:
{out}"""
    else:
        return f"""
Below is an instruction. Write a response that completes the request.

### Instruction:
{instr}

### Response:
{out}"""


class RegressionTrainer(Trainer):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        # token ids for ratings 1–5
        self.rating_tokens = [16, 17, 18, 19, 20]
        self.rating_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits  # (batch, seq_len, vocab)
        
        # 获取倒数第4个 token 的预测分数
        last = logits[:, -5, :]  # (batch, vocab)
        probs = F.softmax(last, dim=-1)
        probs_r = probs[:, self.rating_tokens]  # (batch, 5)
        preds = probs_r.matmul(self.rating_values.to(probs_r.device))  # (batch,)

        # 获取评分 token 的 ID
        last_token_ids = inputs["labels"][:, -4]
        labels_float = torch.zeros(preds.shape, dtype=torch.float, device=preds.device)
        for i in range(len(last_token_ids)):
            token_id = last_token_ids[i].item()
            if token_id in self.rating_tokens:
                rating_idx = self.rating_tokens.index(token_id)
                labels_float[i] = self.rating_values[rating_idx]
            else:
                print(f"Warning: Token ID {token_id} is not a rating token.")

        loss = F.mse_loss(preds, labels_float)
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Train LLaMA with LoRA for 1-5 rating prediction")
    parser.add_argument("--base_model", type=str, default="/root/autodl-tmp/Meta-Llama-3-8B-Instruct", help="Pretrained LLaMA model identifier")
    parser.add_argument("--train_data", type=str, default="./data/task1/train.json", help="Path to training data (json)")
    parser.add_argument("--val_data", type=str, default="./data/task1/val.json", help="Path to validation data (json)")
    parser.add_argument("--output_dir", type=str, default="./lora-regression/task1", help="Directory for saving LoRA-adapted model")
    parser.add_argument("--batch_size", type=int, default=16, help="Total batch size across accumulation")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Batch size per device per step")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--cutoff_len", type=int, default=2048, help="Maximum token length")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume from checkpoint")
    args = parser.parse_args()

    # load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["o_proj", "gate_proj", "k_proj", "up_proj", "q_proj", "down_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    if args.resume_from_checkpoint:
        checkpoint_name = os.path.join(
                args.resume_from_checkpoint, "adapter_model.bin"
            )
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # load and preprocess datasets
    train_ds = load_dataset("json", data_files={"train": args.train_data})["train"]
    val_ds = load_dataset("json", data_files={"train": args.val_data})["train"]

    def preprocess(ex):
        prompt = generate_prompt(ex)
        tok = tokenize_fn(tokenizer, prompt, args.cutoff_len)
        tok["rating_float"] = float(ex["output"])
        return tok
    
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)

    # training setup
    gradient_accumulation = args.batch_size // args.micro_batch_size
    training_args = TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        evaluation_strategy="epoch",  # 启用 epoch 评估策略
        save_strategy="epoch",
        output_dir=args.output_dir,
        save_total_limit=1,
        report_to=None,
        # 添加学习率调度器
        lr_scheduler_type="cosine",
        
    )
    # 新增：先创建一个标准的 Seq2Seq collator
    seq2seq_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = RegressionTrainer(
        tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=seq2seq_collator,
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()