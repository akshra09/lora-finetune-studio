"""
dataset.py — Dataset loading, preprocessing, and formatting for LoRA fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Formatting
# ─────────────────────────────────────────────────────────────────────────────

ALPACA_TEMPLATE = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """\
### Instruction:
{instruction}

### Response:
{output}"""


def format_alpaca_prompt(
    instruction: str,
    output: str,
    input_text: str = "",
) -> str:
    """Format a single example using the Alpaca prompt template."""
    if input_text and input_text.strip():
        return ALPACA_TEMPLATE.format(
            instruction=instruction.strip(),
            input=input_text.strip(),
            output=output.strip(),
        )
    return ALPACA_TEMPLATE_NO_INPUT.format(
        instruction=instruction.strip(),
        output=output.strip(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_training_dataset(config: dict) -> DatasetDict:
    """
    Load dataset from HuggingFace Hub or local JSON file.

    Expects config['dataset'] section from config.yaml.
    """
    dataset_cfg = config["dataset"]
    name = dataset_cfg["name"]

    logger.info(f"Loading dataset: {name}")

    if name == "local":
        local_path = Path(__file__).parent / "data" / "dataset.json"
        if not local_path.exists():
            raise FileNotFoundError(
                f"Local dataset not found at {local_path}. "
                "Create training/data/dataset.json in Alpaca format, or "
                "change dataset.name in config.yaml to a HuggingFace dataset ID."
            )
        with open(local_path) as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset(
            name,
            split=dataset_cfg.get("split", "train"),
        )

    # Sample a fraction if requested
    fraction = dataset_cfg.get("sample_fraction", 1.0)
    if fraction < 1.0:
        n = int(len(dataset) * fraction)
        dataset = dataset.select(range(n))
        logger.info(f"Using {n}/{len(dataset)} samples ({fraction*100:.0f}%)")

    # Train/val split
    val_ratio = dataset_cfg.get("val_split_ratio", 0.05)
    splits = dataset.train_test_split(test_size=val_ratio, seed=42)
    logger.info(
        f"Dataset split — Train: {len(splits['train'])}, Val: {len(splits['test'])}"
    )

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_dataset(
    splits: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    config: dict,
) -> DatasetDict:
    """
    Convert raw text examples to tokenized tensors.
    Handles prompt formatting + tokenization in one pass.
    """
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    max_length = model_cfg.get("max_length", 512)

    instruction_col = dataset_cfg.get("instruction_column", "instruction")
    input_col = dataset_cfg.get("input_column", "input")
    output_col = dataset_cfg.get("output_column", "output")

    def _tokenize(examples):
        texts = []
        for i in range(len(examples[instruction_col])):
            instruction = examples[instruction_col][i] or ""
            inp = examples.get(input_col, [""] * len(examples[instruction_col]))[i] or ""
            output = examples[output_col][i] or ""

            prompt = format_alpaca_prompt(instruction, output, inp)
            texts.append(prompt)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_overflowing_tokens=False,
        )

        # For causal LM, labels = input_ids (next-token prediction)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    cols_to_remove = list(splits["train"].column_names)

    tokenized_splits = splits.map(
        _tokenize,
        batched=True,
        remove_columns=cols_to_remove,
        desc="Tokenizing dataset",
    )

    return tokenized_splits


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

def preview_dataset(splits: DatasetDict, tokenizer: PreTrainedTokenizer, n: int = 3):
    """Print a few examples from the training set for sanity checking."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(f"\n[bold cyan]Dataset Preview[/bold cyan] (showing {n} examples)\n")

    for i in range(min(n, len(splits["train"]))):
        ids = splits["train"][i]["input_ids"]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        console.print(
            Panel(text[:500] + "..." if len(text) > 500 else text,
                  title=f"Example {i+1}",
                  border_style="dim")
        )
