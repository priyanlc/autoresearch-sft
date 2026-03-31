"""Quick check that all dependencies are installed correctly."""

import sys

packages = [
    ('torch', 'torch'),
    ('transformers', 'transformers'),
    ('accelerate', 'accelerate'),
    ('peft', 'peft'),
    ('trl', 'trl'),
    ('datasets', 'datasets'),
    ('polars', 'polars'),
    ('mamba_ssm', 'mamba_ssm'),
    ('causal_conv1d', 'causal_conv1d'),
    ('sentencepiece', 'sentencepiece'),
]

all_ok = True
for name, module in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'ok')
        print(f'  {name}: {version}')
    except ImportError as e:
        print(f'  {name}: MISSING ({e})')
        all_ok = False

print()

import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB')

print()
if all_ok:
    print('All packages installed OK.')
else:
    print('Some packages are missing. Run: pip install -r requirements.txt')
    sys.exit(1)
