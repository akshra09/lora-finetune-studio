"""
push_to_hub.py — Upload LoRA adapter weights to HuggingFace Hub.

Usage:
    python push_to_hub.py --checkpoint outputs/checkpoint-final --repo-name username/my-lora
    python push_to_hub.py --checkpoint outputs/checkpoint-final --repo-name username/my-lora --private
"""

import click
from huggingface_hub import HfApi, create_repo
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml, torch
from pathlib import Path

console = Console()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--checkpoint", "-c", required=True, help="Path to saved LoRA checkpoint dir")
@click.option("--repo-name", "-r", required=True, help="HF Hub repo e.g. username/model-name")
@click.option("--config", default="config.yaml", help="config.yaml path (for base model ID)")
@click.option("--private", is_flag=True, default=False, help="Make the repo private")
@click.option("--token", default=None, help="HF token (or set HF_TOKEN env var)")
def main(checkpoint, repo_name, config, private, token):
    cfg = load_config(config)
    base_model_id = cfg["model"]["base_model"]

    console.print(Panel.fit(
        f"Uploading LoRA adapters\n"
        f"Checkpoint: [cyan]{checkpoint}[/cyan]\n"
        f"Repo: [cyan]{repo_name}[/cyan]\n"
        f"Private: [cyan]{private}[/cyan]",
        title="📤 Push to HuggingFace Hub",
        border_style="blue",
    ) if False else f"\n[bold blue]📤 Pushing to Hub:[/bold blue] {repo_name}\n")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_name, private=private, token=token, exist_ok=True)
        console.print(f"[green]✓ Repo ready:[/green] https://huggingface.co/{repo_name}")
    except Exception as e:
        console.print(f"[red]Error creating repo: {e}[/red]")
        raise

    # Load tokenizer + model from checkpoint
    console.print("Loading adapter from checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint)

    # Push — only adapter weights are uploaded (not the base model)
    console.print("Uploading adapter weights...")
    model.push_to_hub(repo_name, token=token, private=private)
    tokenizer.push_to_hub(repo_name, token=token, private=private)

    # Write a model card
    model_card = f"""---
base_model: {base_model_id}
library_name: peft
tags:
  - lora
  - peft
  - fine-tuned
  - causal-lm
---

# {repo_name.split("/")[-1]}

LoRA fine-tuned adapter for **{base_model_id}**, created with
[LoRA Fine-Tuning Studio](https://github.com/YOUR_USERNAME/lora-finetune-studio).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = "{base_model_id}"
adapter_repo = "{repo_name}"

tokenizer = AutoTokenizer.from_pretrained(adapter_repo)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter_repo)

prompt = "### Instruction:\\nExplain LoRA in simple terms.\\n\\n### Response:\\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
    readme_path = Path(checkpoint) / "README.md"
    readme_path.write_text(model_card)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_name,
        token=token,
    )

    console.print(f"\n[bold green]✓ Successfully pushed![/bold green]")
    console.print(f"  View at: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    main()
