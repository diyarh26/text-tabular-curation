#!/bin/bash
# One-time setup on a Linux GPU machine, from the repo root:
#   bash scripts/setup_gpu_vm.sh
#   source .venv/bin/activate
#   python -m tt_curation run configs/toy.yaml --models light --folds 0    # GPU smoke test, includes TAR
set -e

nvidia-smi --query-gpu=name,memory.total --format=csv || { echo "❌ No NVIDIA GPU visible"; exit 1; }

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d .venv ] || uv venv -p 3.11 .venv
uv pip install --python .venv/bin/python -c requirements/constraints.txt -e ".[bench,dev]"

.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; print('✅ torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

if [ ! -f .env ]; then
    echo "HF_TOKEN=" > .env
    echo "⚠️ Put your Hugging Face token in .env (HF_TOKEN=...); TabPFN 2.5 needs it"
fi
.venv/bin/python prep/make_toy_dataset.py
echo "🎉 Ready. source .venv/bin/activate"
