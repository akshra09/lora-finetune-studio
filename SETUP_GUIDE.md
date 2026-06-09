# 📖 Complete Setup Guide — LoRA Fine-Tuning Studio

This guide walks you through **every single step** from a blank computer to a
fully deployed, working application. No prior ML experience assumed.

---

## Table of Contents

1. [Prerequisites & What You Need](#1-prerequisites--what-you-need)
2. [Get the Code (Clone from GitHub)](#2-get-the-code)
3. [Python Environment Setup](#3-python-environment-setup)
4. [Fine-Tuning on Google Colab (Recommended — Free GPU)](#4-fine-tuning-on-google-colab)
5. [Fine-Tuning Locally (If You Have a GPU)](#5-fine-tuning-locally)
6. [Evaluating Your Model](#6-evaluating-your-model)
7. [Push Your Model to HuggingFace Hub](#7-push-to-huggingface-hub)
8. [Set Up the Web App](#8-set-up-the-web-app)
9. [Deploy to Vercel](#9-deploy-to-vercel)
10. [Testing the Live App](#10-testing-the-live-app)
11. [Troubleshooting A-Z](#11-troubleshooting)
12. [What to Do Next](#12-what-to-do-next)

---

## 1. Prerequisites & What You Need

### Software to install on your computer

#### A) Git
Used to download the code.
- **Windows**: Download from https://git-scm.com/download/win — use default options
- **Mac**: Open Terminal, type `git --version` — if not installed, macOS will prompt you to install it
- **Linux**: `sudo apt install git` or `sudo dnf install git`

Verify: open a terminal/command prompt and run:
```
git --version
```
You should see something like `git version 2.44.0`.

---

#### B) Python 3.10 or newer
Required for the training scripts.

- **Windows**: Download from https://www.python.org/downloads/  
  ⚠️ During install, check **"Add Python to PATH"** — very important!
- **Mac**: Use the official installer at https://www.python.org/downloads/macos/  
  Or use Homebrew: `brew install python@3.11`
- **Linux**: `sudo apt install python3.11 python3.11-venv python3-pip`

Verify:
```
python --version        # Windows
python3 --version       # Mac/Linux
```
Should show `Python 3.10.x` or newer.

---

#### C) Node.js 18 or newer
Required for the web application.

- Download from: https://nodejs.org/en/download  
  Choose **LTS** (Long Term Support) version
- Install with default options

Verify:
```
node --version    # should show v18.x.x or higher
npm --version     # should show 9.x.x or higher
```

---

#### D) A Code Editor (optional but helpful)
- **VS Code**: https://code.visualstudio.com/ (recommended — free)
- Any editor works

---

### Accounts to create (all free)

| Service | URL | Why needed |
|---------|-----|-----------|
| **GitHub** | https://github.com | Store your code |
| **HuggingFace** | https://huggingface.co | Download models + store your fine-tuned adapters |
| **Vercel** | https://vercel.com | Deploy the web app |
| **Google** | (you likely have one) | Run training on Google Colab (free GPU) |

---

## 2. Get the Code

### Option A: Clone from GitHub (after you push it)
```bash
git clone https://github.com/YOUR_USERNAME/lora-finetune-studio.git
cd lora-finetune-studio
```

### Option B: Extract the zip
If you downloaded the zip file:
1. Right-click the zip → Extract All (Windows) or double-click (Mac)
2. Open a terminal and navigate to the folder:
   ```bash
   cd path/to/lora-finetune-studio
   ```

### To push this project to your own GitHub:
```bash
cd lora-finetune-studio

git init
git add .
git commit -m "Initial commit: LoRA Fine-Tuning Studio"

# Create a new repo at https://github.com/new
# Then run these commands (replace YOUR_USERNAME):
git remote add origin https://github.com/YOUR_USERNAME/lora-finetune-studio.git
git branch -M main
git push -u origin main
```

---

## 3. Python Environment Setup

A virtual environment keeps your project's dependencies isolated from the rest of your computer.

```bash
# Make sure you're in the project root
cd lora-finetune-studio

# Create a virtual environment named "venv"
python -m venv venv            # Windows
python3 -m venv venv           # Mac/Linux

# Activate it
venv\Scripts\activate          # Windows Command Prompt
venv\Scripts\Activate.ps1      # Windows PowerShell
source venv/bin/activate       # Mac/Linux
```

You'll see `(venv)` at the start of your terminal prompt — that means it's active.

**Install Python packages:**
```bash
pip install -r requirements.txt
```

This installs PyTorch, HuggingFace Transformers, PEFT, and everything else needed. It may take 5–10 minutes depending on your internet speed.

**Deactivate when done:**
```bash
deactivate
```

---

## 4. Fine-Tuning on Google Colab

This is the **recommended path** if you don't have a powerful GPU. Colab gives you a free T4 GPU (16GB VRAM), which is enough to fine-tune 1B–7B models with QLoRA.

### Step 4.1: Open the notebook

1. Go to https://colab.research.google.com
2. Click **File → Upload notebook**
3. Upload `notebooks/LoRA_Finetune_Colab.ipynb` from this project
4. Or go to **File → Open notebook → GitHub** and paste your repo URL

### Step 4.2: Enable GPU

In Colab:
1. Click **Runtime** in the top menu
2. Click **Change runtime type**
3. Under "Hardware accelerator", select **T4 GPU**
4. Click **Save**

### Step 4.3: Configure your training

Find the "Configuration" cell (Step 2 in the notebook) and edit:

```python
BASE_MODEL   = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'  # Keep this for your first run
DATASET_NAME = 'yahma/alpaca-cleaned'                  # Free dataset of 52K instructions
SAMPLE_FRAC  = 0.1   # Use 10% of dataset — about 5,000 examples, ~30 min training
NUM_EPOCHS   = 1
```

**For your first run, keep these defaults** — training will take about 30 minutes.

### Step 4.4: Run all cells

Click **Runtime → Run all** and wait. You'll see output like:
```
Loading TinyLlama/TinyLlama-1.1B-Chat-v1.0...
✓ Model loaded with LoRA adapters
  Trainable params: 4,194,304 (0.3% of 1,100,000,000 total)
...
Step 100/500 — loss: 1.8432
Step 200/500 — loss: 1.6201
...
✓ Training complete! Saved to /content/lora-output/checkpoint-final
```

### Step 4.5: Test the model

The "Test the fine-tuned model" cell will run inference. You should see your model responding to instructions.

### Step 4.6: Save your adapter

**Option A (push to HuggingFace Hub — recommended):**
Fill in `HF_USERNAME` in the config cell, then run the "Push to Hub" cell.

**Option B (download as zip):**
Run the "Download checkpoint" cell, then find `lora-checkpoint.zip` in the Files panel (folder icon on the left sidebar) and download it.

---

## 5. Fine-Tuning Locally

Only do this if you have a CUDA-capable NVIDIA GPU with at least 8GB VRAM.

### Step 5.1: Check your GPU

```bash
nvidia-smi
```

You need an NVIDIA GPU. AMD GPUs are not supported by the current setup.

### Step 5.2: Edit the config

Open `training/config.yaml` and change settings as needed:

```yaml
model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

dataset:
  name: "yahma/alpaca-cleaned"
  sample_fraction: 0.1   # 10% for a quick test run

training:
  num_epochs: 1
  per_device_train_batch_size: 2   # Reduce to 1 if you get out-of-memory errors
```

### Step 5.3: Login to HuggingFace (to download gated models)

```bash
huggingface-cli login
```

Paste your token from https://huggingface.co/settings/tokens (Read permission is enough).

### Step 5.4: Run training

```bash
# Activate your virtual environment first
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

cd training
python train.py --config config.yaml
```

For a quick test with minimal data (great for verifying everything works):
```bash
python train.py --config config.yaml --debug
```

The `--debug` flag uses 1% of data and 1 epoch — finishes in a few minutes.

### Step 5.5: Monitor training

Open TensorBoard to see live loss curves:
```bash
tensorboard --logdir training/outputs
```
Then visit http://localhost:6006 in your browser.

---

## 6. Evaluating Your Model

After training completes:

```bash
cd training
python evaluate.py \
  --config config.yaml \
  --checkpoint outputs/checkpoint-final \
  --samples 50
```

This runs 50 inference examples and computes ROUGE scores. Results are saved to
`outputs/checkpoint-final/eval_results.json`.

---

## 7. Push to HuggingFace Hub

This step uploads your LoRA adapter weights (~10–50MB) to HuggingFace so the web app can use them.

### Step 7.1: Create a HuggingFace account

Go to https://huggingface.co and sign up (free).

### Step 7.2: Create an API token

1. Log in to HuggingFace
2. Click your profile picture → Settings → Access Tokens
3. Click "New token"
4. Name it `lora-studio`, permission: **Write**
5. Copy the token (starts with `hf_`)

### Step 7.3: Login on your machine

```bash
huggingface-cli login
```

Paste your token and press Enter.

### Step 7.4: Push the adapters

```bash
cd training

python push_to_hub.py \
  --checkpoint outputs/checkpoint-final \
  --repo-name YOUR_HF_USERNAME/tinyllama-alpaca-lora \
  --config config.yaml
```

Replace `YOUR_HF_USERNAME` with your actual HuggingFace username.

After this completes, your model will be live at:
`https://huggingface.co/YOUR_HF_USERNAME/tinyllama-alpaca-lora`

---

## 8. Set Up the Web App

The web app is a Next.js application that lets you chat with your fine-tuned model via HuggingFace's free Inference API.

### Step 8.1: Navigate to the app folder

```bash
cd app
```

### Step 8.2: Install Node.js dependencies

```bash
npm install
```

This downloads ~200MB of Node.js packages. Takes 1–3 minutes.

### Step 8.3: Create your environment file

```bash
# You must be in the app/ folder
cp ../.env.example .env.local
```

Now open `.env.local` in any text editor and fill in your values:

```env
HUGGINGFACE_API_KEY=hf_your_actual_token_here
DEFAULT_MODEL_ID=YOUR_HF_USERNAME/tinyllama-alpaca-lora
NEXT_PUBLIC_DEFAULT_MODEL=YOUR_HF_USERNAME/tinyllama-alpaca-lora
```

Replace both values with your actual HuggingFace token and model ID.

### Step 8.4: Run the app locally

```bash
npm run dev
```

You'll see output like:
```
▲ Next.js 14.2.3
- Local:        http://localhost:3000
- Environments: .env.local
```

Open http://localhost:3000 in your browser. You should see the LoRA Studio chat interface!

### Step 8.5: Test a chat

Type a message like "What is LoRA?" and press Enter. The app will call HuggingFace's
Inference API and stream back a response.

**Note:** The first request may take 20–30 seconds if the model is "cold" (not loaded on HF servers yet). Subsequent requests are faster.

---

## 9. Deploy to Vercel

Vercel is the fastest way to go from local to production. The free tier is more than enough for this project.

### Option A: Deploy via GitHub (Recommended)

This is the best approach — Vercel auto-deploys every time you push to GitHub.

**Step 9.1:** Make sure your code is on GitHub (see Section 2).

**Step 9.2:** Go to https://vercel.com and sign up with your GitHub account.

**Step 9.3:** Click **"Add New Project"** (top right).

**Step 9.4:** Import your GitHub repository.

**Step 9.5:** On the "Configure Project" screen:
- **Framework Preset**: Next.js (Vercel detects this automatically)
- **Root Directory**: Click "Edit" and type `app`
- Leave all other settings as default

**Step 9.6:** Add Environment Variables. Click "Environment Variables" and add:

| Name | Value |
|------|-------|
| `HUGGINGFACE_API_KEY` | `hf_your_token_here` |
| `DEFAULT_MODEL_ID` | `YOUR_HF_USERNAME/tinyllama-alpaca-lora` |
| `NEXT_PUBLIC_DEFAULT_MODEL` | `YOUR_HF_USERNAME/tinyllama-alpaca-lora` |

**Step 9.7:** Click **Deploy**.

Vercel will build and deploy your app. In about 2 minutes, you'll get a live URL like:
`https://lora-finetune-studio-abc123.vercel.app`

---

### Option B: Deploy via Vercel CLI

If you prefer the command line:

```bash
# Install Vercel CLI globally
npm install -g vercel

# Login
vercel login
# Opens browser, log in with GitHub

# Deploy from the app folder
cd app
vercel

# Follow the prompts:
# ? Set up and deploy "app"? → Y
# ? Which scope? → Your personal account
# ? Link to existing project? → N
# ? Project name? → lora-finetune-studio
# ? Directory? → ./ (press Enter)

# Add environment variables
vercel env add HUGGINGFACE_API_KEY
# Paste your HF token and press Enter

vercel env add DEFAULT_MODEL_ID
# Type your model ID and press Enter

# Deploy to production
vercel --prod
```

---

### Setting up custom domain (optional)

1. In Vercel dashboard → your project → Settings → Domains
2. Click "Add" and type your domain name
3. Follow the DNS configuration instructions

---

## 10. Testing the Live App

Once deployed, visit your Vercel URL and:

1. **The chat interface loads** — you see the LoRA Studio UI
2. **Model is shown in the header** — your model name appears
3. **Send a test message**: "What is LoRA fine-tuning?"
4. **Receive a response** — may take 20–30 seconds on first request (model cold start)

### Common behaviors:
- **"Model is loading"** → Normal! HuggingFace loads models on demand. Wait 30 seconds and retry.
- **Fast responses** → Model is warm. Enjoy!
- **API key error** → Check environment variables in Vercel dashboard

### Health check
Visit `YOUR_URL/api/health` — you should see:
```json
{
  "status": "ok",
  "hfApiConfigured": true,
  "defaultModel": "your-username/your-model",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

---

## 11. Troubleshooting

### Python / Training Issues

**`CUDA out of memory` during training**
```yaml
# In config.yaml, reduce these:
training:
  per_device_train_batch_size: 1   # Was 4
  gradient_accumulation_steps: 8   # Keep effective batch size the same
model:
  max_length: 256                  # Was 512
```

**`No module named 'bitsandbytes'` on Windows**
BitsAndBytesConfig (4-bit quantization) has limited Windows support. Try:
```yaml
quantization:
  use_4bit: false   # Disable QLoRA, use full bfloat16
```
Or use Google Colab instead (Linux environment, full support).

**`ValueError: You are attempting to load a model with Flash Attention`**
```yaml
model:
  use_flash_attention: false   # Already the default
```

**Training loss not decreasing after many steps**
- Check your dataset quality — are inputs/outputs correctly formatted?
- Try a higher learning rate: `learning_rate: 5e-4`
- Try a larger rank: `lora.r: 32`

**`ModuleNotFoundError: No module named 'peft'`**
```bash
source venv/bin/activate   # Make sure venv is active!
pip install -r requirements.txt
```

---

### Web App Issues

**`HUGGINGFACE_API_KEY is not configured`**
- Check `app/.env.local` — does the file exist?
- Does the key start with `hf_`?
- Restart the dev server after editing `.env.local`: Ctrl+C, then `npm run dev`

**HuggingFace API returns 503**
- The model is loading (cold start). Wait 20–30 seconds and retry.
- This happens after the model hasn't been used for a while.

**HuggingFace API returns 404**
- The model ID is wrong. Check `DEFAULT_MODEL_ID` in `.env.local`.
- Go to `https://huggingface.co/YOUR_USERNAME` and verify the model name.

**`npm install` fails**
- Make sure Node.js 18+ is installed: `node --version`
- Try clearing npm cache: `npm cache clean --force`
- Delete node_modules and try again: `rm -rf node_modules && npm install`

**Vercel build fails**
- Go to Vercel dashboard → your project → Deployments → click the failed deployment
- Check the build logs for the specific error
- Most common cause: missing environment variables — add them in Settings → Environment Variables

**Blank page on Vercel**
- Check that **Root Directory** is set to `app` in Vercel project settings
- Check build logs for TypeScript errors

---

### HuggingFace Issues

**`401 Unauthorized`**
Your token is wrong or expired.
- Go to https://huggingface.co/settings/tokens
- Create a new token with Write permission
- Update `HUGGINGFACE_API_KEY` in your `.env.local` and Vercel env vars

**"This model is gated" error when downloading**
Some models (like Llama 3) require you to accept terms on HuggingFace first.
- Go to the model page on HuggingFace
- Click "Agree and access repository"
- Then re-run training

---

## 12. What to Do Next

Now that you have a working fine-tuned model and web app, here are some ideas:

### Improve the model
1. **Use more data**: Set `sample_fraction: 1.0` and train on the full dataset
2. **Train longer**: `num_epochs: 3` improves quality significantly
3. **Try a better base model**: Swap to `mistralai/Mistral-7B-Instruct-v0.2` for much better output
4. **Use your own dataset**: Create `training/data/dataset.json` with your domain-specific data

### Improve the app
1. Add streaming responses (requires a self-hosted inference server like TGI or vLLM)
2. Add conversation history persistence (localStorage or a database)
3. Add a "compare models" side-by-side view
4. Add authentication (NextAuth.js)

### Go deeper into LoRA
- Read the original LoRA paper: https://arxiv.org/abs/2106.09685
- Experiment with different `target_modules` — adding more layers increases capacity
- Try DPO (Direct Preference Optimization) with `trl` for RLHF-style training
- Try full fine-tuning of a small model to compare quality vs LoRA

### For your portfolio/resume
- Add this project to your GitHub with a good README
- Record a demo video of the web app
- Write a blog post explaining what you learned about LoRA

---

*Questions? Open an issue on GitHub.*
