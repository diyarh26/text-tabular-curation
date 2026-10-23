# Text-Tabular Dataset Curation with MulTaBench

Curating a new **text-tabular** dataset (a table whose features include free text next to numerical and categorical
columns) and testing it with the [MulTaBench](https://github.com/alanarazi7/MulTaBench) curation pipeline.

Final project, NLP course 097215, Technion, Spring 2026 (Track 2: Benchmark).

## The curation pipeline

A dataset belongs in a text-tabular benchmark only if both modalities matter and adapting the text encoder to the
task helps. Each of 5 learners (TabM, CatBoost, LightGBM, TabPFN 2.0, TabPFN 2.5) is evaluated in 4 conditions:

| Condition | Features | Text encoder |
|---|---|---|
| `structured` | numerical + categorical | — |
| `unstructured` | text only | frozen E5-small-v2 |
| `joint_frozen` | all | frozen E5-small-v2 |
| `joint_tar` | all | E5-small-v2 LoRA-tuned on the target (target-aware) |

The dataset passes if, for at least 3 of the 5 learners:
1. **Joint signal:** `joint_frozen` > `structured` and `joint_frozen` > `unstructured`
2. **TAR gain:** `joint_tar` > `joint_frozen`

Scores are MulTaBench's (higher is better; error = 1 − AUROC for binary, log loss for multiclass, RMSE for
regression), averaged over its 6 outer folds (3-fold CV × 2 repeats, train/test capped at 10K/5K rows). AUROC / R² are
reported as well.

## How it works

MulTaBench's `benchmark.py` only runs the datasets it hosts. This repo calls one level below it, so a new dataset gets
**exactly the official splits, model wrappers, preprocessing, encoders and metrics** (MulTaBench pinned to commit
`ff0c9c8`). Two details:

- Columns are typed explicitly from a YAML spec, since MulTaBench embeds any `object` column as text.
- As with MulTaBench's TAR cache, E5 is LoRA-tuned once per fold on the fold's training rows and shared by all five
  learners.

```
configs/        one YAML spec per dataset (start from _template.yaml)
prep/           code that builds each curated CSV from its raw source
data/processed/ the curated CSVs (deliverable); data/raw/ stays local
src/tt_curation the pipeline: spec + typed loading, runner, criteria
results/<name>/ runs/*.json, evaluation_report.csv, curation_verdict.csv
notebooks/      run_on_colab.ipynb, for the GPU runs
paper/          the NeurIPS-format report
```

## Usage

```bash
pip install uv
uv pip install -c requirements/constraints.txt -e ".[bench,dev]"

python -m tt_curation validate configs/my_dataset.yaml                 # types, class balance, warnings
python -m tt_curation run configs/my_dataset.yaml --preset screen      # quick: LightGBM + CatBoost, 3 folds, no TAR
python -m tt_curation run configs/my_dataset.yaml                      # full: 5 learners x 4 conditions x 6 folds
python -m tt_curation report configs/my_dataset.yaml                   # rebuild the CSVs from the run JSONs
```

The full run needs a GPU (TabPFN, TabM and the LoRA tuning; `joint_tar` refuses to run on CPU): on a Linux GPU
machine run `bash scripts/setup_gpu_vm.sh`, or use `notebooks/run_on_colab.ipynb`. Verify access to the TabPFN model
weights before starting; these wrappers read `HF_TOKEN` from `.env` when authentication is needed. Runs are
resumable: rerunning skips successful runs and retries failed ones.

Smoke test on CPU with a synthetic dataset: `pytest` (`pytest -m "not bench"` skips the end-to-end run).

## Full Steam run on the VM

The prepared Steam CSV and config are included in Git. See [the VM run instructions](docs/VM_STEAM_RUN.md) for
preflight checks, data verification, and launching inside tmux. From the repository root, after setting up the
full benchmark environment:

```bash
.venv/bin/python -u scripts/run_full_steam.py
```

This launches all **5 learners x 4 conditions x 6 folds = 120 evaluations**, with frozen E5 for the text baselines
and TAR capped at the official **50 epochs with early stopping**. It adds the tested persistent checkpoint logic
to the existing runner and defaults to the official training batch of 256. Smaller execution micro-batches can be
selected explicitly after an approved memory adjustment, retaining effective batch 256 through accumulation.
All six folds run sequentially on one GPU, with one tuned encoder per fold shared across learners.

Results go to `results/steam_full_e5/steam_recommended/`. A manifest prevents reuse with different data, code,
dependencies, or training settings. Rerun the same command to resume. To audit all expected evaluations:

```bash
.venv/bin/python scripts/run_full_steam.py --audit_only
```

Checkpoint files are ignored by Git. Intermediate evaluation reports are saved after each fold; the final
curation report is generated only after all 120 expected evaluations succeed. This entry point is separate
from the smaller Colab pilot described below.

## Full LinkedIn run on Kaggle T4 x2

Import [the Kaggle notebook](notebooks/run_linkedin_full_kaggle.ipynb), enable **GPU T4 x2** and Internet,
and attach the `GH_TOKEN` Kaggle secret for read access to this private repository. See
[setup and resume instructions](docs/KAGGLE_LINKEDIN_RUN.md). GPU 0 handles folds 0/2/4 and GPU 1 handles
1/3/5, each in its own process, for **120 evaluations** with the 50-epoch cap. The approved Kaggle memory
adjustment uses micro-batch 16 and 16-step gradient accumulation (effective batch 256); set
`MICRO_BATCH = 256` for the reference physical batch on hardware where it fits.
Fold checkpoints and results are isolated; a final verdict requires all 120 expected successful evaluations.
The notebook has **no automatic time cutoff**. Rolling checkpoint ZIPs are exported every 15 minutes,
and another session can resume the same corrected revision if manually interrupted or stopped by Kaggle.
For the approved 256-to-16 change, it backs up the previous experiment and imports validated baseline
evaluations only when data, model code, dependencies and other scientific settings match. TAR starts fresh.
This adds separate launchers and does not change the active Steam VM runtime code.

**October 23 correction:** the legacy batch-16 adapter accumulated mean losses without normalizing
by the effective batch, scaling gradients by approximately 16. The LinkedIn launcher now uses summed
cross-entropy divided by the full optimizer update's example count, including short final batches.
Actual pinned-Trainer gradient, optimizer, evaluation and restart tests cover the correction.
The notebook preserves affected experiments, reuses only validated baseline evaluations, and retrains
all six TAR encoders. Old TAR optimizer checkpoints are rejected. Steam's full physical-batch-256
evaluation is unaffected. See [the correction and restart instructions](docs/KAGGLE_LINKEDIN_RUN.md).

## Steam versus LinkedIn: Colab TAR pilot

Open `notebooks/compare_steam_linkedin_tar_colab.ipynb` in Colab and upload
`artifacts/steam_linkedin_tar_bundle.zip` when prompted. Rebuild both from the current local data/code with
`python scripts/build_colab_tar_bundle.py`; the bundle includes both prepared 15K-row CSVs, so no GitHub push or
private-repo token is required. Select a GPU runtime.

The default runs **joint TAR only**, using CatBoost and LightGBM, fold 0, and a 10-epoch cap with early stopping.
This is an exploratory cap; MulTaBench's official cap is 50. Tuning uses micro-batches of 16 with gradient
accumulation to preserve an effective batch of 256. Drive stores completed runs, epoch checkpoints, and completed
adapters, allowing the same experiment to resume across sessions. Steam and LinkedIn have separate run cells.

The existing local screens used TF-IDF, so comparison against those screens changes both the encoder and tuning.
The notebook's optional frozen-E5 joint control is off by default; enable it to isolate TAR gain. Its comparison
table always matches the folds available for both conditions. The pilot does not establish full curation acceptance.
The legacy Colab micro-batch helper shares the loss-normalization bug described above; its TAR pilot
scores are exploratory and should not be used as results from the corrected training protocol.

## Results

_To be added once the dataset is curated._

## References

- A. Arazi et al. *MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image.* arXiv:2605.10616, 2026.
- A. Arazi, E. Shapira, R. Reichart. *TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields.* NeurIPS 2025.
