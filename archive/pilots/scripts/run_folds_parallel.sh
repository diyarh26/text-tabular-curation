#!/bin/bash
# One process per GPU, each running whole folds (all 5 learners x 4 conditions, sharing the fold's TAR encoder).
#   bash scripts/run_folds_parallel.sh configs/la_airbnb_price.yaml 1,2,3,4,5
# GPU i runs the i-th fold of the list; logs go to logs/fold<k>.log. Runs are resumable, so rerun the same command
# after an interruption. Run it inside tmux.
set -e
CONFIG=${1:?config, e.g. configs/la_airbnb_price.yaml}
FOLDS=${2:?comma-separated folds, e.g. 1,2,3,4,5}
IFS=',' read -ra FOLD_LIST <<< "$FOLDS"
N_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "${#FOLD_LIST[@]}" -gt "$N_GPUS" ]; then
    echo "❌ ${#FOLD_LIST[@]} folds but only $N_GPUS GPUs"; exit 1
fi

mkdir -p logs
source .venv/bin/activate
# The CPU cores, shared among the processes (CatBoost, LightGBM and TabM use them).
export OMP_NUM_THREADS=$(( $(nproc) / ${#FOLD_LIST[@]} ))
pids=()
for i in "${!FOLD_LIST[@]}"; do
    fold=${FOLD_LIST[$i]}
    echo "🚀 fold $fold on GPU $i -> logs/fold$fold.log"
    CUDA_VISIBLE_DEVICES=$i python -m tt_curation run "$CONFIG" --folds "$fold" >> "logs/fold$fold.log" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
python -m tt_curation report "$CONFIG"
[ $status -eq 0 ] && echo "🎉 all folds done" || echo "⚠️ a fold process failed; see logs/, then rerun this command"
