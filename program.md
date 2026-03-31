# Autoresearch: SFT for Nemotron Reasoning Challenge

## Goal
Maximize **validation accuracy** on 6 types of "Alice's Wonderland" reasoning puzzles by optimizing Supervised Fine-Tuning (SFT) of a LoRA adapter on the Nemotron-3-Nano-30B model.

## Metric
The single metric to optimize is printed at the end of `train.py`:
```
METRIC: 0.XXXX
```
This is the proportion of correctly answered puzzles on a held-out validation set (120 samples, 20 per category). Higher is better. Maximum is 1.0.

## What you can modify
You may ONLY edit `train.py`. Everything in `prepare.py` is read-only.

### Things worth trying (in rough priority order):

**Prompt format (high impact):**
- How the answer is formatted in training data (just `\boxed{answer}` vs reasoning + answer)
- System prompt variations (generic vs category-specific)
- Whether to include reasoning scaffolding in the assistant response

**Data strategy (high impact):**
- `SFT_SAMPLES_PER_TYPE`: how many samples per puzzle category (currently 50 for speed)
- Sampling strategy: uniform across categories vs weighted toward harder ones
- Data ordering: shuffle vs curriculum (easy first)

**LoRA configuration (medium impact):**
- `LORA_RANK`: adapter rank (1-32, competition max is 32)
- `LORA_ALPHA`: scaling factor (typically 1x or 2x the rank)
- `LORA_DROPOUT`: regularization (0.0 to 0.1)
- `TARGET_MODULES`: which layers to adapt ('all-linear' vs specific projections)

**Training hyperparameters (medium impact):**
- `SFT_LR`: learning rate (try 1e-5 to 5e-4)
- `SFT_EPOCHS`: number of passes (1-3)
- `WARMUP_RATIO`: LR warmup fraction
- `WEIGHT_DECAY`: L2 regularization
- `MAX_SEQ_LEN`: maximum sequence length for training

**Answer format in training (high impact):**
- Direct answer: `\boxed{answer}` only
- With reasoning: step-by-step explanation then `\boxed{answer}`
- Category-specific: different formats per puzzle type

## Constraints
- LoRA rank must be ≤ 32 (competition rule)
- The model must output answers in `\boxed{}` format
- Training must complete within the time budget
- Do not modify `prepare.py`

## The 6 puzzle types
1. **bit_ops** — deduce bit transformation from input/output examples → 8-char binary string
2. **cipher** — crack substitution cipher → lowercase words
3. **gravity** — infer gravitational constant from d=0.5gt² → decimal number
4. **numeral** — convert number to Roman numerals → Roman numeral string
5. **symbol** — figure out symbol substitution/arithmetic rules → symbol string
6. **unit_conv** — find hidden linear conversion factor → decimal number

Each type has ~20 validation samples. Check per-category accuracy to find weak spots.

## Tips
- Start by understanding the current accuracy breakdown per category
- Focus on the weakest categories first — that's where gains are easiest
- The prompt format during SFT is critical because inference uses `enable_thinking=True`
- Small data changes (even re-ordering) can have outsized effects
- If a change crashes, revert and try something smaller
