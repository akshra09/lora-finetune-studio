"""
train.py — Main LoRA fine-tuning script.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --model mistralai/Mistral-7B-Instruct-v0.2
    python train.py --config config.yaml --epochs 5 --lr 1e-4
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import click
import torch
import yaml
from datasets import disable_progress_bar
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

from dataset import load_training_dataset, preview_dataset, tokenize_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(config: dict):
    """Load base model with optional 4-bit quantization (QLoRA)."""
    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    base_model_id = model_cfg["base_model"]

    console.print(f"\n[bold green]Loading base model:[/bold green] {base_model_id}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4-bit quantization config (QLoRA)
    bnb_config = None
    if quant_cfg.get("use_4bit", True):
        compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_double_quant=quant_cfg.get("use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        console.print("[dim]  → 4-bit QLoRA quantization enabled[/dim]")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
        attn_implementation=(
            "flash_attention_2" if model_cfg.get("use_flash_attention") else "eager"
        ),
    )

    # Prepare for k-bit training (needed for QLoRA)
    if bnb_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
        )

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Configuration
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model, config: dict):
    """Wrap the model with PEFT LoRA adapters."""
    lora_cfg = config["lora"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        inference_mode=False,
    )

    model = get_peft_model(model, peft_config)
    return model, peft_config


def print_trainable_parameters(model):
    """Print a summary of trainable vs total parameters."""
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    pct = 100 * trainable / total

    table = Table(title="Model Parameters", show_header=True, header_style="bold magenta")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("% of Total", justify="right")
    table.add_row("Trainable (LoRA)", f"{trainable:,}", f"{pct:.4f}%")
    table.add_row("Frozen (base)", f"{total - trainable:,}", f"{100 - pct:.4f}%")
    table.add_row("Total", f"{total:,}", "100%")
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Training Callback (nice progress display)
# ─────────────────────────────────────────────────────────────────────────────

class RichProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            total = state.max_steps
            loss = logs.get("loss", "—")
            lr = logs.get("learning_rate", "—")
            epoch = logs.get("epoch", "—")
            if isinstance(lr, float):
                lr = f"{lr:.2e}"
            console.print(
                f"  [dim]step {step}/{total}[/dim]  "
                f"loss=[yellow]{loss:.4f}[/yellow]  "
                f"lr=[blue]{lr}[/blue]  "
                f"epoch=[cyan]{epoch}[/cyan]"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict):
    """Run the full training pipeline."""

    train_cfg = config["training"]
    set_seed(train_cfg.get("seed", 42))

    # ── Banner ───────────────────────────────────────────────────────────────
    console.print(Panel.fit(
        "[bold]LoRA Fine-Tuning Studio[/bold]\n"
        f"Model: [cyan]{config['model']['base_model']}[/cyan]\n"
        f"Dataset: [cyan]{config['dataset']['name']}[/cyan]\n"
        f"LoRA rank: [cyan]{config['lora']['r']}[/cyan]  "
        f"alpha: [cyan]{config['lora']['alpha']}[/cyan]",
        title="🧠 Training Run",
        border_style="green",
    ))

    # ── Load model ───────────────────────────────────────────────────────────
    model, tokenizer = load_base_model(config)
    model, peft_config = apply_lora(model, config)
    print_trainable_parameters(model)

    # ── Load dataset ─────────────────────────────────────────────────────────
    splits = load_training_dataset(config)
    tokenized = tokenize_dataset(splits, tokenizer, config)
    preview_dataset(tokenized, tokenizer, n=2)

    # ── Training arguments ───────────────────────────────────────────────────
    output_dir = train_cfg.get("output_dir", "outputs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Reporting
    report_to = []
    if config.get("tracking", {}).get("tensorboard", True):
        report_to.append("tensorboard")
    if config.get("tracking", {}).get("wandb", {}).get("enabled", False):
        report_to.append("wandb")
        os.environ["WANDB_PROJECT"] = config["tracking"]["wandb"].get(
            "project", "lora-finetune-studio"
        )
    if not report_to:
        report_to = ["none"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.001),
        optim=train_cfg.get("optimizer", "paged_adamw_32bit"),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 100),
        evaluation_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 100),
        logging_steps=train_cfg.get("logging_steps", 10),
        max_steps=train_cfg.get("max_steps", -1),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        group_by_length=train_cfg.get("group_by_length", True),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        seed=train_cfg.get("seed", 42),
        push_to_hub=False,  # We handle this separately
    )

    # ── Data collator ────────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        peft_config=peft_config,
        callbacks=[RichProgressCallback()],
        dataset_text_field=None,  # We pre-tokenize above
        max_seq_length=config["model"].get("max_length", 512),
        packing=train_cfg.get("packing", False),
    )

    # ── Train ────────────────────────────────────────────────────────────────
    console.print("\n[bold green]Starting training...[/bold green]\n")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    final_dir = Path(output_dir) / "checkpoint-final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"  Adapters saved to: [cyan]{final_dir}[/cyan]")

    # ── Auto-push to Hub ─────────────────────────────────────────────────────
    hub_cfg = config.get("hub", {})
    if hub_cfg.get("push_to_hub", False):
        repo_id = hub_cfg["repo_id"]
        console.print(f"\n[bold]Pushing to HuggingFace Hub:[/bold] {repo_id}")
        trainer.model.push_to_hub(repo_id, private=hub_cfg.get("private", False))
        tokenizer.push_to_hub(repo_id, private=hub_cfg.get("private", False))
        console.print(f"[green]✓ Pushed to Hub: https://huggingface.co/{repo_id}[/green]")

    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config.yaml")
@click.option("--model", default=None, help="Override base model ID")
@click.option("--epochs", default=None, type=int, help="Override num_epochs")
@click.option("--lr", default=None, type=float, help="Override learning rate")
@click.option("--rank", default=None, type=int, help="Override LoRA rank")
@click.option("--output-dir", default=None, help="Override output directory")
@click.option("--debug", is_flag=True, help="Use tiny data split for quick testing")
def main(config, model, epochs, lr, rank, output_dir, debug):
    cfg = load_config(config)

    # Apply CLI overrides
    if model:
        cfg["model"]["base_model"] = model
    if epochs:
        cfg["training"]["num_epochs"] = epochs
    if lr:
        cfg["training"]["learning_rate"] = lr
    if rank:
        cfg["lora"]["r"] = rank
        cfg["lora"]["alpha"] = rank * 2
    if output_dir:
        cfg["training"]["output_dir"] = output_dir
    if debug:
        cfg["dataset"]["sample_fraction"] = 0.01
        cfg["training"]["num_epochs"] = 1
        cfg["training"]["save_steps"] = 10
        cfg["training"]["eval_steps"] = 10
        console.print("[yellow]⚠ Debug mode: using 1% of data, 1 epoch[/yellow]")

    train(cfg)


if __name__ == "__main__":
    main()
