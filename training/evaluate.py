"""
evaluate.py — Evaluate a fine-tuned LoRA model on held-out examples.

Usage:
    python evaluate.py --config config.yaml
    python evaluate.py --config config.yaml --checkpoint outputs/checkpoint-final
    python evaluate.py --config config.yaml --samples 50
"""

import json
import logging
import time
from pathlib import Path

import click
import torch
import yaml
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from dataset import format_alpaca_prompt, load_training_dataset

logger = logging.getLogger(__name__)
console = Console()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_for_inference(base_model_id: str, adapter_path: str):
    """Load base model + LoRA adapters for inference (full precision, no quantization)."""
    console.print(f"[bold green]Loading model for evaluation...[/bold green]")
    console.print(f"  Base: [cyan]{base_model_id}[/cyan]")
    console.print(f"  Adapter: [cyan]{adapter_path}[/cyan]")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate a response from the fine-tuned model."""
    # Build prompt (without the output part)
    if input_text and input_text.strip():
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            scores["rouge1"].append(s["rouge1"].fmeasure)
            scores["rouge2"].append(s["rouge2"].fmeasure)
            scores["rougeL"].append(s["rougeL"].fmeasure)
        return {k: sum(v) / len(v) for k, v in scores.items()}
    except ImportError:
        console.print("[yellow]rouge_score not installed; skipping ROUGE metrics[/yellow]")
        return {}


def run_evaluation(config: dict, checkpoint: str, n_samples: int = 100):
    base_model = config["model"]["base_model"]
    model, tokenizer = load_model_for_inference(base_model, checkpoint)

    # Load validation split
    splits = load_training_dataset(config)
    val_data = splits["test"]

    n = min(n_samples, len(val_data))
    console.print(f"\n[bold]Evaluating on {n} samples...[/bold]\n")

    predictions, references = [], []
    results = []
    start = time.time()

    for i in range(n):
        sample = val_data[i]
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        reference = sample.get("output", "")

        pred = generate_response(model, tokenizer, instruction, input_text)
        predictions.append(pred)
        references.append(reference)

        results.append({
            "instruction": instruction,
            "input": input_text,
            "reference": reference,
            "prediction": pred,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            console.print(f"  [{i+1}/{n}] {elapsed:.1f}s elapsed")

    elapsed = time.time() - start

    # ROUGE scores
    rouge = compute_rouge(predictions, references)

    # Display results table
    table = Table(title="Evaluation Results", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="green")

    table.add_row("Samples evaluated", str(n))
    table.add_row("Total time (s)", f"{elapsed:.1f}")
    table.add_row("Avg time/sample (s)", f"{elapsed/n:.2f}")
    if rouge:
        table.add_row("ROUGE-1", f"{rouge['rouge1']:.4f}")
        table.add_row("ROUGE-2", f"{rouge['rouge2']:.4f}")
        table.add_row("ROUGE-L", f"{rouge['rougeL']:.4f}")
    console.print(table)

    # Show a few examples
    console.print("\n[bold]Sample Predictions:[/bold]\n")
    for r in results[:3]:
        console.print(f"[dim]Instruction:[/dim] {r['instruction'][:100]}")
        console.print(f"[yellow]Reference:[/yellow] {r['reference'][:200]}")
        console.print(f"[green]Prediction:[/green] {r['prediction'][:200]}")
        console.print()

    # Save results
    out_path = Path(checkpoint) / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"metrics": rouge, "samples": results}, f, indent=2)
    console.print(f"[green]✓ Results saved to {out_path}[/green]")

    return rouge


@click.command()
@click.option("--config", "-c", default="config.yaml")
@click.option("--checkpoint", default="outputs/checkpoint-final", help="Path to LoRA checkpoint")
@click.option("--samples", default=100, type=int, help="Number of eval samples")
def main(config, checkpoint, samples):
    cfg = load_config(config)
    run_evaluation(cfg, checkpoint, samples)


if __name__ == "__main__":
    main()
