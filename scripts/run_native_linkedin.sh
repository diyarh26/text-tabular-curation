#!/usr/bin/env bash
# Native official entry points. Separate reproduction output; never alter VM-v1 results.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
NATIVE_CHECKOUT="${NATIVE_CHECKOUT:-$PROJECT_ROOT/artifacts/multabench-native}"
export MULTABENCH_LINKEDIN_DATA_DIR="${MULTABENCH_LINKEDIN_DATA_DIR:-$PROJECT_ROOT/artifacts/native_linkedin_data}"
export MPLBACKEND=Agg
export PYTHONPATH="$NATIVE_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/artifacts/native_linkedin_evaluation}"
mkdir -p "$OUTPUT"
"$PYTHON" "$PROJECT_ROOT/scripts/setup_native_multabench.py" --checkout "$NATIVE_CHECKOUT"
"$PYTHON" -m multabench.datasets.prepare_linkedin --csv "$PROJECT_ROOT/data/processed/linkedin_salary.csv" --output "$MULTABENCH_LINKEDIN_DATA_DIR"
"$PYTHON" "$PROJECT_ROOT/scripts/verify_native_linkedin.py" --data_dir "$MULTABENCH_LINKEDIN_DATA_DIR" > "$OUTPUT/loader_parity.json"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; assert torch.cuda.device_count()==1, "Set CUDA_VISIBLE_DEVICES to ONE GPU"; from multabench.e5.constants import E5_LORA; assert E5_LORA.batch_size==256 and E5_LORA.weight_decay==0.001; print(torch.cuda.get_device_name(0), E5_LORA)'
cd "$NATIVE_CHECKOUT"
# This is the unchanged official embedding creation command (frozen once, TAR once per fold).
"$PYTHON" -m multabench.embeddings.embed --encoder e5-small --dataset_name REG_TEXT_LINKEDIN_SALARY --output_dir "$MULTABENCH_LINKEDIN_DATA_DIR" --device cuda
for fold in 0 1 2 3 4 5; do
    "$PYTHON" -m multabench.embeddings.embed --encoder e5-small-tar --dataset_name REG_TEXT_LINKEDIN_SALARY --fold "$fold" --size 10k --output_dir "$MULTABENCH_LINKEDIN_DATA_DIR" --device cuda
    for model in tabm cat light tabpfnv2 tabpfnv2p5; do
        for dataset in REG_TEXT_LINKEDIN_SALARY_STRUCTURED REG_TEXT_LINKEDIN_SALARY_TEXT REG_TEXT_LINKEDIN_SALARY; do
            "$PYTHON" benchmark.py --model "$model" --dataset_name "$dataset" --fold "$fold" --size 10k --text_encoder e5-small --device cuda --output_dir "$OUTPUT/runs"
        done
        "$PYTHON" benchmark.py --model "$model" --dataset_name REG_TEXT_LINKEDIN_SALARY --fold "$fold" --size 10k --text_encoder e5-small-tar --device cuda --output_dir "$OUTPUT/runs"
    done
    "$PYTHON" "$PROJECT_ROOT/scripts/report_native_linkedin.py" --results "$OUTPUT"
done
