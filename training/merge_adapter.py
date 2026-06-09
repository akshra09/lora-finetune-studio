"""
merge_adapter.py — Merge LoRA adapter weights into the base model for faster inference.

After merging, you get a standalone model (no PEFT library needed at inference time).
The merged model is larger but runs faster and can be deployed anywhere.

Usage:
    python merge_adapter.py \\
        --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --adapter-path outputs/checkpoint-final \\
        --output-path outputs/merged-model
"""

import click
import torch
from pathlib import Path
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()


@click.command()
@click.option("--base-model", "-b", required=True, help="HF base model ID")
@click.option("--adapter-path", "-a", required=True, help="Path to LoRA checkpoint")
@click.option("--output-path", "-o", required=True, help="Where to save merged model")
@click.option("--push-to-hub", default=None, help="Optionally push merged model to HF Hub repo")
@click.option("--token", default=None, help="HF token for hub push")
def main(base_model, adapter_path, output_path, push_to_hub, token):
    console.print(Panel.fit(
        f"Base model: [cyan]{base_model}[/cyan]\n"
        f"Adapter: [cyan]{adapter_path}[/cyan]\n"
        f"Output: [cyan]{output_path}[/cyan]",
        title="🔀 Merging LoRA Adapter",
        border_style="magenta",
    ))

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load base model in full precision
    console.print("\n[bold]Step 1:[/bold] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    # 2. Load and attach LoRA adapter
    console.print("[bold]Step 2:[/bold] Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # 3. Merge weights
    console.print("[bold]Step 3:[/bold] Merging weights (this may take a minute)...")
    model = model.merge_and_unload()

    # 4. Save merged model
    console.print(f"[bold]Step 4:[/bold] Saving merged model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    console.print(f"\n[bold green]✓ Merged model saved to:[/bold green] {output_path}")
    console.print(
        "\n[dim]The merged model is a standard HuggingFace model — no PEFT needed.\n"
        "Load it with: AutoModelForCausalLM.from_pretrained('" + output_path + "')[/dim]"
    )

    # 5. Optional: push to Hub
    if push_to_hub:
        console.print(f"\n[bold]Step 5:[/bold] Pushing merged model to Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub, token=token, safe_serialization=True)
        tokenizer.push_to_hub(push_to_hub, token=token)
        console.print(f"[green]✓ Pushed to: https://huggingface.co/{push_to_hub}[/green]")


if __name__ == "__main__":
    main()
