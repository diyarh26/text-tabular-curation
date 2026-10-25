#!/usr/bin/env bash
# Run INSIDE tmux. A failure stops the sequence; tee must be called with pipefail by the caller.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ ! -x .venv/bin/python ]]; then
    echo 'Missing .venv/bin/python. Run bash scripts/setup_gpu_vm.sh in this checkout first.' >&2
    exit 1
fi

PYTHON=.venv/bin/python
ROOT=results/linkedin_vm_batch256
"$PYTHON" -u scripts/run_full_linkedin_vm.py --results_root "$ROOT" --prepare_only
if [[ ! -f "$ROOT/linkedin_salary/gpu_preflight.json" ]]; then
    "$PYTHON" -u scripts/run_full_linkedin_vm.py --results_root "$ROOT" --gpu_smoke
fi
"$PYTHON" -u scripts/run_full_linkedin_vm.py --results_root "$ROOT"
"$PYTHON" -u scripts/run_full_linkedin_vm.py --results_root "$ROOT" --audit_only
"$PYTHON" -u scripts/run_full_linkedin_vm.py --results_root "$ROOT" --report_only
echo 'COMPLETE: all 120 LinkedIn VM physical-batch-256 evaluations audited and reported.'
