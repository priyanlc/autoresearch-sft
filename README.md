# Autoresearch SFT — Nemotron Reasoning Challenge

Autonomous SFT optimization using the [autoresearch](https://github.com/karpathy/autoresearch) pattern.

## Setup (RunPod A100)

```bash
# 1. Clone this directory to your RunPod instance
# 2. Copy train.csv into ./data/
mkdir -p data
cp /path/to/train.csv data/

# 3. Install PyTorch matching your CUDA version (check with: nvcc --version)
pip install torch --index-url https://download.pytorch.org/whl/cu128  # adjust cu128 to match your CUDA

# 4. Install mamba_ssm and causal_conv1d (must use --no-build-isolation to avoid CUDA mismatch)
pip install mamba_ssm --no-build-isolation
pip install causal_conv1d --no-build-isolation

# 5. Install remaining dependencies
pip install transformers accelerate peft trl datasets polars sentencepiece

# 6. Verify packages installed correctly
python check_install.py

# 7. Run one-time preparation
python prepare.py

# 8. Verify baseline works
python train.py

# 9. Initialize git (autoresearch uses git to track experiments)
git init
git add -A
git commit -m "initial baseline"
```

## Running with autoresearch

```bash
# Clone autoresearch
git clone https://github.com/karpathy/autoresearch
cd autoresearch

# Point it at our directory (or copy our files into autoresearch structure)
# The AI agent will:
#   1. Read program.md for instructions
#   2. Modify train.py
#   3. Run: python train.py
#   4. Parse METRIC: X.XXXX from output
#   5. Keep or discard the change
#   6. Repeat
```

## File structure

```
autoresearch-sft/
├── program.md      # Instructions for the AI agent (what to optimize)
├── prepare.py      # One-time setup + evaluation harness (READ-ONLY)
├── train.py        # Training script (AI agent modifies this)
├── data/
│   ├── train.csv   # Competition training data (you provide)
│   └── val_split.json  # Fixed validation split (created by prepare.py)
└── adapter/        # Output LoRA adapter (created by train.py)
```

## Manual iteration

You can also iterate manually without autoresearch:

```bash
# Edit train.py (change config, prompt format, etc.)
python train.py
# Check METRIC at the end of output
# If improved, git commit. If not, git checkout train.py
```

## Time per experiment

On A100 80GB with `SFT_SAMPLES_PER_TYPE=50` (300 total samples):
- Model loading: ~2 min
- SFT training: ~5 min
- Evaluation: ~5 min
- **Total: ~12 min per experiment**

For faster iteration, reduce `SFT_SAMPLES_PER_TYPE` to 25 or reduce `EVAL_MAX_NEW_TOKENS`.
