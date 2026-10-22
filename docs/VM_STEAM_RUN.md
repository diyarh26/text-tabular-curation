# Full Steam evaluation on a GPU VM

The requested experiment is **Steam only, all 120 evaluations**. The Colab pilot used two learners, one fold,
and a 10-epoch cap; the VM run uses five learners, four conditions, six folds, and the official 50-epoch TAR cap
with early stopping. Do not use `colab_tar_compare.py` as the full-run entry point.

## Locate or clone the project

The initially opened folder may be empty. Inspect likely existing project locations and running jobs first.
Use an appropriate existing checkout/environment when present; otherwise clone using the already configured
Git credentials:

```bash
git clone https://github.com/diyarh26/text-tabular-curation.git
cd text-tabular-curation
```

Inspect Git status before updating an existing checkout; preserve local changes and results. Use a fast-forward
update only when appropriate. Never display tokens. The prepared CSV, config, and code are in Git, so no upload
bundle or new raw-data download is necessary.

## Preflight

- Read repository instructions, README, the Steam config, and the launcher's implementation.
- Check GPU/VRAM, RAM, disk capacity, current GPU jobs, the Python environment, and actual CUDA operation.
- Inspect installed versions against `requirements/constraints.txt`; MulTaBench is pinned to
  `ff0c9c8b40a2085a06a7771190e53a261341802f`. Reuse a healthy environment.
- If setup is missing, inspect and run `scripts/setup_gpu_vm.sh`. It installs the full benchmark extras;
  `requirements/colab_tar.txt` alone is insufficient for the five-learner run.
- Check TabPFN 2.0 and 2.5 checkpoint access before the long experiment. Inspect actual wrappers for credentials;
  they read `HF_TOKEN`. Keep any required token in the environment or ignored `.env`, never in Git or logs.
- Run the existing lightweight checks (`pytest -m "not bench"`) and appropriate small import/CUDA/model-access
  checks. Keep smoke-test output outside the final results directory. Fix routine environment issues before launch.
- All subprocesses should use `MPLBACKEND=Agg`; the dedicated launcher already sets it. Never inherit an
  unavailable notebook-only Matplotlib backend into an isolated Python environment.

Verify `data/processed/steam_recommended.csv`: exactly 15,000 rows; 13,405 `yes`, 1,595 `no`; nonempty review text;
one text and eight structured features as declared in `configs/steam_recommended.yaml`. No minimum word count
or per-game quota was imposed. The sample covers 123 games; the existing rare-category merge leaves 121 categories.

Expected SHA256 after replacing CRLF line endings with LF:

```text
c9a9594066403fda72ee765d17ebb9461ed7bdb461d82161aaa745e294e631f5
```

The launcher verifies this hash. Investigate mismatches; do not regenerate or resample the data.

## Launch and monitor

Use the existing VM resources. With one GPU, folds run sequentially. Check for an existing matching job before
starting another. Run inside tmux with persistent stdout/stderr logging, for example from the repository root:

```bash
mkdir -p logs
tmux new-session -d -s steam_full 'bash -c "set -o pipefail; .venv/bin/python -u scripts/run_full_steam.py 2>&1 | tee -a logs/steam_full.log"'
```

Adjust the Python executable if the verified environment has a different path. Confirm actual progress after
launch. Do not rent additional hardware, alter billing, change the scientific protocol, or silently omit learners.

The launcher uses:

- TabM, CatBoost, LightGBM, TabPFN-v2, TabPFN-v2p5.
- `structured`, `unstructured`, `joint_frozen`, `joint_tar`.
- Folds 0 through 5: repeated 3-fold CV, twice, with at most 10,000 train and 5,000 test rows per split.
- Frozen E5-small-v2 for text-only/joint-frozen, not TF-IDF; standard training-fitted PCA preprocessing.
- One independent target-tuned E5 per fold, shared across all five learners; no outer-test-label training.
- TAR: maximum 50 epochs, early stopping, official LR/rank/layers, full precision, and the official batch of 256
  without gradient accumulation. If CUDA runs out of memory, stop and report the error before changing settings.
  Smaller execution micro-batches can be selected explicitly only after approval; accumulation retains effective
  batch 256. That adjustment creates a different manifest, so never silently mix runs with different settings.

Output: `results/steam_full_e5/steam_recommended/`. Completed evaluations are skipped; epoch checkpoints and
completed adapters persist per fold. An interruption repeats only work after the latest usable checkpoint.
The manifest checks data, config, code, package versions, and training settings before reusing prior work.

If a model fails, investigate its JSON traceback and fix the cause before resuming. The upstream CLI can exit
successfully even when individual evaluation JSONs contain errors. The full launcher audits each completed fold
and stops on failures. Never interpret incomplete results as a final acceptance result.

## Completion

```bash
.venv/bin/python scripts/run_full_steam.py --audit_only
```

Completion requires the exact 120 expected model/condition/fold combinations with successful statuses and
matching metadata, not merely 120 JSON files. Inspect `evaluation_report.csv`, `curation_verdict.csv`, and
`summary.txt` after the final audit. A dataset passes only when at least three learners satisfy both joint signal
and TAR gain under the repository's scoring rules.

When handing back the job, report the Git commit, GPU, data checks, environment fixes, exact command,
tmux session, log path, output directory, and actual progress. Do not stop at writing a plan if checks pass.
