"""
train.py — SFT training script for autoresearch iteration.

This is the ONLY file the AI agent should modify.
Each run: loads model, trains SFT, evaluates, prints METRIC.

Usage:
    python train.py
"""

import os
import sys
import gc
import json
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import polars as pl
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# Import evaluation harness from prepare.py (read-only)
from prepare import (
    MODEL_ID, TRAIN_CSV, classify_type, evaluate_model, load_val_data,
    stratified_sample, METRIC_SUFFIX, VAL_SAMPLES_PER_TYPE,
)

# ============================================================
# === CONFIGURATION — Modify this section to improve METRIC ===
# ============================================================

# Data
SFT_SAMPLES_PER_TYPE = 50          # Keep small for fast iteration (6 * 50 = 300 samples)

# SFT hyperparameters
SFT_LR = 2e-4
SFT_EPOCHS = 1
SFT_MAX_SEQ_LEN = 1024
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0

# LoRA
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = 'all-linear'       # Options: 'all-linear', or list like ['q_proj','v_proj','o_proj']

# Training
BATCH_SIZE = 1
GRAD_ACCUM = 4

# Evaluation
EVAL_MAX_NEW_TOKENS = 384

# Output
OUTPUT_DIR = './adapter'

# ============================================================
# === PROMPT FORMAT — This is the highest-impact thing to change
# ============================================================

def build_sft_text(example, tokenizer):
    """Format a training example for SFT.

    Modify this function to change how the model learns to respond.
    The format here directly affects how the model reasons at inference time.
    """
    user_msg = example['prompt'] + METRIC_SUFFIX

    # --- Answer format: modify this to change what the model learns ---
    # Option A: Direct answer only (current)
    assistant_msg = f'\\boxed{{{example["answer"]}}}'

    # Option B: Brief reasoning + answer (uncomment to try)
    # assistant_msg = f'The answer is \\boxed{{{example["answer"]}}}'

    # Option C: Category-aware format (uncomment to try)
    # qtype = classify_type(example['prompt'])
    # if qtype == 'gravity':
    #     assistant_msg = f'Using d = 0.5*g*t^2, the answer is \\boxed{{{example["answer"]}}}'
    # elif qtype == 'numeral':
    #     assistant_msg = f'Converting to Roman numerals: \\boxed{{{example["answer"]}}}'
    # else:
    #     assistant_msg = f'\\boxed{{{example["answer"]}}}'

    messages = [
        {'role': 'user', 'content': user_msg},
        {'role': 'assistant', 'content': assistant_msg},
    ]
    for kwargs in [{'enable_thinking': True}, {}]:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs
            )
            return {'text': text}
        except Exception:
            continue
    return {'text': f'<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>'}


# ============================================================
# === TRAINING — Modify data loading/sampling strategy here ===
# ============================================================

def load_training_data(tokenizer):
    """Load and prepare training data. Modify sampling strategy here."""
    train_df = pl.read_csv(TRAIN_CSV)
    train_df = train_df.with_columns(
        pl.col('prompt').map_elements(classify_type, return_dtype=pl.Utf8).alias('qtype')
    )

    # Exclude validation data (same seed=0 as prepare.py)
    val_df = stratified_sample(train_df, VAL_SAMPLES_PER_TYPE, seed=0)
    val_ids = set(val_df['id'].to_list())
    remaining = train_df.filter(~pl.col('id').is_in(val_ids))

    # Sample training data
    sft_df = stratified_sample(remaining, SFT_SAMPLES_PER_TYPE, seed=42)

    # Build HF dataset
    sft_dataset = Dataset.from_pandas(sft_df.drop('qtype').to_pandas())
    sft_dataset = sft_dataset.map(
        lambda ex: build_sft_text(ex, tokenizer),
        remove_columns=sft_dataset.column_names,
    )

    return sft_dataset


# ============================================================
# === MAIN — Do not modify below this line ===
# ============================================================

def main():
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load tokenizer
    print(f'Loading tokenizer: {MODEL_ID}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    print('Preparing training data...')
    sft_dataset = load_training_data(tokenizer)
    val_data = load_val_data()
    print(f'SFT: {len(sft_dataset)} samples, Val: {len(val_data)} samples')

    # Load model
    print(f'Loading model: {MODEL_ID}')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Disable fast path
    for name, mod in sys.modules.items():
        if 'modeling_nemotron_h' in name:
            mod.is_fast_path_available = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Train
    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=SFT_EPOCHS,
        learning_rate=SFT_LR,
        logging_steps=10,
        bf16=True,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        warmup_ratio=WARMUP_RATIO,
        save_strategy='no',
        report_to='none',
        dataset_text_field='text',
        max_length=SFT_MAX_SEQ_LEN,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
        args=sft_args,
    )

    print(f'Training {len(sft_dataset)} samples x {SFT_EPOCHS} epoch...')
    trainer.train()

    # Save adapter
    model.save_pretrained(OUTPUT_DIR)
    print(f'Adapter saved to {OUTPUT_DIR}')

    # Cleanup trainer
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate
    print('\nEvaluating...')
    overall, by_type = evaluate_model(model, tokenizer, val_data, max_new_tokens=EVAL_MAX_NEW_TOKENS)

    # Print results
    elapsed = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'Overall accuracy: {overall:.4f}')
    for qtype in sorted(by_type.keys()):
        stats = by_type[qtype]
        acc = stats['correct'] / max(stats['total'], 1)
        print(f'  {qtype}: {acc:.4f} ({stats["correct"]}/{stats["total"]})')
    print(f'Time: {elapsed:.0f}s')
    print(f'{"="*60}')

    # === THE METRIC LINE — autoresearch parses this ===
    print(f'METRIC: {overall:.4f}')

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
