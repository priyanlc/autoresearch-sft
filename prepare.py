"""
prepare.py — One-time data preparation and evaluation harness (READ-ONLY).

Run once before starting autoresearch:
    python prepare.py

This downloads the model, prepares the data splits, and verifies everything works.
The evaluation function is also defined here and imported by train.py.
"""

import os
import sys
import re
import math
import json
import subprocess

import polars as pl
import torch
from transformers import AutoTokenizer


# === Paths ===
MODEL_ID = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
DATA_DIR = './data'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
VAL_JSON = os.path.join(DATA_DIR, 'val_split.json')
RESULTS_TSV = 'results.tsv'

# === Validation split (fixed, never changes) ===
VAL_SAMPLES_PER_TYPE = 20  # 6 * 20 = 120 samples


def classify_type(prompt_text):
    """Classify a puzzle prompt into one of 6 categories."""
    p = prompt_text.lower()
    if 'bit manipulation' in p or '8-bit binary' in p: return 'bit_ops'
    elif 'encrypt' in p or 'decrypt' in p: return 'cipher'
    elif 'gravitational' in p or 'falling distance' in p: return 'gravity'
    elif 'numeral system' in p: return 'numeral'
    elif 'transformation rules' in p: return 'symbol'
    elif 'unit conversion' in p or 'convert the following measurement' in p: return 'unit_conv'
    return 'unknown'


def extract_boxed_answer(text):
    """Extract the last non-empty \\boxed{} content from model output."""
    if text is None:
        return None
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        for m in reversed(matches):
            if m.strip():
                return m.strip()
    idx = text.rfind('\\boxed{')
    if idx == -1:
        return None
    depth = 0
    start = idx + len('\\boxed{')
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                content = text[start:i].strip()
                return content if content else None
            depth -= 1
    content = text[start:].strip()
    return content if content else None


def answers_match(pred, gold):
    """Check if prediction matches gold."""
    if pred is None:
        return False
    try:
        return math.isclose(float(pred), float(gold), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        pass
    return pred.strip().lower() == gold.strip().lower()


METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


def evaluate_model(model, tokenizer, val_data, max_new_tokens=512):
    """Evaluate on validation set with greedy decoding. Returns (overall_accuracy, per_type_dict)."""
    model.eval()
    results = {'total': 0, 'correct': 0, 'by_type': {}}

    for example in val_data:
        qtype = example['qtype']
        messages = [{'role': 'user', 'content': example['prompt'] + METRIC_SUFFIX}]

        try:
            result = tokenizer.apply_chat_template(
                messages, return_tensors='pt', add_generation_prompt=True,
                enable_thinking=True
            )
        except TypeError:
            result = tokenizer.apply_chat_template(
                messages, return_tensors='pt', add_generation_prompt=True
            )

        if hasattr(result, 'input_ids'):
            input_ids = result['input_ids'].to(model.device)
        elif isinstance(result, dict):
            input_ids = result['input_ids'].to(model.device)
        else:
            input_ids = result.to(model.device)

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_boxed_answer(response)
        gold = example['answer']
        is_correct = answers_match(pred, gold)

        results['total'] += 1
        results['correct'] += int(is_correct)

        if qtype not in results['by_type']:
            results['by_type'][qtype] = {'total': 0, 'correct': 0}
        results['by_type'][qtype]['total'] += 1
        results['by_type'][qtype]['correct'] += int(is_correct)

    overall = results['correct'] / max(results['total'], 1)
    return overall, results['by_type']


def load_val_data():
    """Load the fixed validation split."""
    with open(VAL_JSON, 'r') as f:
        return json.load(f)


def stratified_sample(df, n_per_type, seed):
    """Sample n_per_type from each category."""
    dfs = []
    for qtype in df['qtype'].unique().to_list():
        subset = df.filter(pl.col('qtype') == qtype)
        n = min(n_per_type, len(subset))
        dfs.append(subset.sample(n=n, seed=seed))
    return pl.concat(dfs)


# === One-time preparation ===
if __name__ == '__main__':
    print('=== Autoresearch SFT: Preparation ===')

    # Check data exists
    if not os.path.exists(TRAIN_CSV):
        print(f'ERROR: {TRAIN_CSV} not found.')
        print(f'Please copy train.csv to {DATA_DIR}/')
        sys.exit(1)

    # Create validation split (fixed, deterministic)
    print('Creating validation split...')
    train_df = pl.read_csv(TRAIN_CSV)
    train_df = train_df.with_columns(
        pl.col('prompt').map_elements(classify_type, return_dtype=pl.Utf8).alias('qtype')
    )

    val_df = stratified_sample(train_df, VAL_SAMPLES_PER_TYPE, seed=0)
    val_data = []
    for row in val_df.iter_rows(named=True):
        val_data.append({
            'prompt': row['prompt'],
            'answer': str(row['answer']),
            'qtype': classify_type(row['prompt']),
        })

    with open(VAL_JSON, 'w') as f:
        json.dump(val_data, f)
    print(f'Saved {len(val_data)} validation samples to {VAL_JSON}')

    # Verify model is downloadable
    print(f'\nVerifying model access: {MODEL_ID}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f'Tokenizer loaded. Vocab size: {len(tokenizer)}')

    # Initialize results.tsv
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, 'w') as f:
            f.write('commit\tmetric\tbit_ops\tcipher\tgravity\tnumeral\tsymbol\tunit_conv\tstatus\tdescription\n')
        print(f'Initialized {RESULTS_TSV}')

    print('\n=== Preparation complete ===')
    print(f'Next: run "python train.py" to verify baseline, then start autoresearch.')
