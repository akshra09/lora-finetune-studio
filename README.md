# 🧠 LoRA Fine-Tuning Studio

A full-stack project for fine-tuning Large Language Models using **LoRA (Low-Rank Adaptation)** with a production-ready web UI deployed on Vercel.

![LoRA Fine-Tuning Studio](https://img.shields.io/badge/LLM-LoRA%20Fine--Tuning-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)
![Next.js](https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js)
![Vercel](https://img.shields.io/badge/Deployed-Vercel-black?style=for-the-badge&logo=vercel)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow?style=for-the-badge)

---

## 📋 Table of Contents

- [What is LoRA?](#what-is-lora)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Part 1: Fine-Tuning (Python Training Pipeline)](#part-1-fine-tuning-python-training-pipeline)
- [Part 2: Push Adapters to HuggingFace Hub](#part-2-push-adapters-to-huggingface-hub)
- [Part 3: Deploy Web UI to Vercel](#part-3-deploy-web-ui-to-vercel)
- [Part 4: Running Locally](#part-4-running-locally)
- [Configuration Reference](#configuration-reference)
- [Customization Guide](#customization-guide)
- [Troubleshooting](#troubleshooting)

---

## What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning (PEFT) technique that freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture.

Instead of updating all parameters (billions), LoRA only trains a tiny fraction:

```
W' = W + ΔW = W + (A × B)
```

Where:
- `W` = original frozen weight matrix (d × d)
- `A` = trainable matrix (d × r) — randomly initialized
- `B` = trainable matrix (r × d) — zero initialized
- `r` = rank (hyperparameter, typically 4–64)

**Key benefits:**
- 🔥 **10,000x fewer trainable parameters** than full fine-tuning
- 💾 **Tiny adapter files** (~10-100 MB vs multi-GB models)
- ⚡ **Runs on consumer GPUs** (even a single RTX 3090 or T4 on Colab)
- 🔀 **Swappable adapters** — one base model, many tasks

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
│  Local Machine / Google Colab / Kaggle Notebooks        │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Dataset  │───▶│  Train   │───▶│  LoRA Adapters   │  │
│  │ (JSON/   │    │  (PEFT + │    │  (adapter_model  │  │
│  │  HF Hub) │    │  QLoRA)  │    │   .safetensors)  │  │
│  └──────────┘    └──────────┘    └────────┬─────────┘  │
└──────────────────────────────────────────┼─────────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │  HuggingFace Hub │
                                  │  (your-org/      │
                                  │   model-lora)    │
                                  └────────┬────────┘
                                           │
                          ┌────────────────┴──────────────────┐
                          │                                    │
                          ▼                                    ▼
               ┌──────────────────┐                ┌──────────────────┐
               │  Vercel Web UI   │                │  HF Inference    │
               │  (Next.js 14)    │◀───────────────│  API / TGI       │
               │                  │   API calls    │                  │
               │  Chat Interface  │                │  Model Serving   │
               │  Model Selector  │                │                  │
               └──────────────────┘                └──────────────────┘
```

---

## Project Structure

```
lora-finetune-studio/
│
├── 📁 training/                    # Python training pipeline
│   ├── train.py                    # Main training script
│   ├── dataset.py                  # Dataset loading & preprocessing
│   ├── evaluate.py                 # Evaluation & benchmarking
│   ├── push_to_hub.py              # Upload adapters to HF Hub
│   ├── merge_adapter.py            # Merge LoRA into base model
│   └── config.yaml                 # All hyperparameters
│
├── 📁 notebooks/
│   └── LoRA_Finetune_Colab.ipynb  # One-click Colab notebook
│
├── 📁 app/                         # Next.js web application
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx            # Home / chat page
│   │   │   ├── layout.tsx          # Root layout
│   │   │   ├── globals.css         # Global styles
│   │   │   └── api/
│   │   │       ├── chat/
│   │   │       │   └── route.ts    # Chat API (calls HF)
│   │   │       ├── models/
│   │   │       │   └── route.ts    # List available models
│   │   │       └── health/
│   │   │           └── route.ts    # Health check endpoint
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx   # Main chat UI component
│   │   │   ├── MessageBubble.tsx   # Individual message
│   │   │   ├── ModelSelector.tsx   # Model picker dropdown
│   │   │   ├── SettingsPanel.tsx   # Temperature, max tokens
│   │   │   └── Header.tsx          # Top navigation
│   │   └── lib/
│   │       ├── hf-client.ts        # HuggingFace API client
│   │       └── types.ts            # TypeScript types
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.ts
│   └── tsconfig.json
│
├── .env.example                    # Environment variables template
├── .gitignore
├── vercel.json                     # Vercel deployment config
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Prerequisites

### For Training (Python)
- Python 3.10+
- CUDA-capable GPU (at minimum 8GB VRAM; 16GB+ recommended)
- OR Google Colab / Kaggle (free GPU access)

### For Web UI (Node.js)
- Node.js 18+
- npm or yarn
- Vercel account (free tier works)
- HuggingFace account + API token

---

## Part 1: Fine-Tuning (Python Training Pipeline)

### Step 1: Clone the repo and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/lora-finetune-studio.git
cd lora-finetune-studio

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure your training run

Edit `training/config.yaml` to set your model, dataset, and hyperparameters:

```yaml
model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change this

lora:
  r: 16          # LoRA rank — higher = more capacity, more VRAM
  alpha: 32      # LoRA alpha — scaling factor (usually 2x rank)
  dropout: 0.05

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
```

### Step 3: Prepare your dataset

Your dataset should be a JSON file with instruction-input-output format:

```json
[
  {
    "instruction": "Summarize the following text.",
    "input": "The quick brown fox...",
    "output": "A fox jumps over a dog."
  }
]
```

Place it at `training/data/dataset.json` or point to a HuggingFace dataset in `config.yaml`.

### Step 4: Run training

```bash
cd training
python train.py --config config.yaml
```

Training outputs will be saved to `training/outputs/`:
- `adapter_model.safetensors` — your LoRA weights
- `adapter_config.json` — LoRA config
- `training_args.bin` — training arguments
- `trainer_state.json` — loss curves, metrics

### Step 5: Evaluate the model

```bash
python evaluate.py --config config.yaml --checkpoint outputs/checkpoint-final
```

### Colab Alternative

Open `notebooks/LoRA_Finetune_Colab.ipynb` in Google Colab — it's fully self-contained with step-by-step cells, works on the free T4 GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/lora-finetune-studio/blob/main/notebooks/LoRA_Finetune_Colab.ipynb)

---

## Part 2: Push Adapters to HuggingFace Hub

### Step 1: Login to HuggingFace

```bash
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

### Step 2: Push your LoRA adapters

```bash
cd training
python push_to_hub.py \
  --checkpoint outputs/checkpoint-final \
  --repo-name YOUR_HF_USERNAME/my-lora-model \
  --private   # optional: make it private
```

This uploads only the adapter weights (~10-50MB), NOT the full base model.

---

## Part 3: Deploy Web UI to Vercel

### Step 1: Set up environment variables

```bash
cd app
cp ../.env.example .env.local
```

Edit `.env.local`:

```env
HUGGINGFACE_API_KEY=hf_your_token_here
DEFAULT_MODEL_ID=YOUR_HF_USERNAME/my-lora-model
```

### Step 2: Install dependencies and test locally

```bash
npm install
npm run dev
# Visit http://localhost:3000
```

### Step 3: Deploy to Vercel

**Option A: Vercel CLI (Recommended)**

```bash
npm install -g vercel
vercel login
vercel --cwd app
```

Follow the prompts. When asked for environment variables, add `HUGGINGFACE_API_KEY`.

**Option B: GitHub Integration**

1. Push to GitHub
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import your repository
4. Set **Root Directory** to `app`
5. Add environment variables:
   - `HUGGINGFACE_API_KEY` → your HF token
   - `DEFAULT_MODEL_ID` → your model repo ID
6. Click **Deploy** ✅

### Step 4: Add your custom domain (optional)

In Vercel dashboard → Project → Settings → Domains → Add Domain.

---

## Part 4: Running Locally

To run everything locally end-to-end:

```bash
# Terminal 1: Training (one-time)
cd training && python train.py --config config.yaml

# Terminal 2: Web UI
cd app && npm run dev
```

Visit `http://localhost:3000` to chat with your fine-tuned model.

---

## Configuration Reference

### `training/config.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.base_model` | TinyLlama-1.1B | Base HF model ID |
| `lora.r` | 16 | LoRA rank |
| `lora.alpha` | 32 | LoRA scaling |
| `lora.dropout` | 0.05 | Dropout probability |
| `lora.target_modules` | q_proj, v_proj | Which layers to adapt |
| `training.epochs` | 3 | Number of epochs |
| `training.batch_size` | 4 | Per-device batch size |
| `training.learning_rate` | 2e-4 | AdamW learning rate |
| `training.use_4bit` | true | QLoRA quantization |
| `dataset.name` | alpaca | HF dataset or local path |
| `dataset.max_length` | 512 | Max token length |

---

## Customization Guide

### Swap the base model

Change `model.base_model` in `config.yaml`. Good choices for limited GPU:

| Model | VRAM Required | Quality |
|-------|--------------|---------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~6GB | ⭐⭐ |
| `microsoft/phi-2` | ~8GB | ⭐⭐⭐ |
| `mistralai/Mistral-7B-Instruct-v0.2` | ~16GB | ⭐⭐⭐⭐ |
| `meta-llama/Llama-3-8B-Instruct` | ~20GB | ⭐⭐⭐⭐⭐ |

### Use your own dataset

Create `training/data/dataset.json` in Alpaca format, or reference any HF dataset by setting:

```yaml
dataset:
  name: "your-hf-dataset/name"
  split: "train"
  text_column: "text"
```

### Merge LoRA weights (for faster inference)

```bash
python training/merge_adapter.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/checkpoint-final \
  --output-path outputs/merged-model
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `batch_size` to 1-2, enable `use_4bit: true`, reduce `max_length` |
| `No module named 'peft'` | Run `pip install -r requirements.txt` |
| HF API 503 error | Model is loading on HF; wait 20s and retry |
| Vercel build fails | Check `HUGGINGFACE_API_KEY` is set in Vercel env vars |
| Training loss not decreasing | Increase `lora.r`, decrease `learning_rate` |
| Output is gibberish | Increase training epochs, check dataset quality |

---

## 📚 Learn More

- [LoRA Paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [HuggingFace PEFT Docs](https://huggingface.co/docs/peft)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)

---

## License

MIT License — free to use, modify, and distribute.
