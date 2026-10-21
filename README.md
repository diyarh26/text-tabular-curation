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

The full run needs a GPU (TabPFN, TabM and the LoRA tuning): use `notebooks/run_on_colab.ipynb`. TabPFN 2.5's weights
are gated: accept their license on Hugging Face and set `HF_TOKEN` (in `.env` locally, or as a Colab secret). Runs are
resumable: an interrupted run is picked up where it stopped.

Smoke test on CPU with a synthetic dataset: `pytest` (`pytest -m "not bench"` skips the end-to-end run).

## Results

_To be added once the dataset is curated._

## References

- A. Arazi et al. *MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image.* arXiv:2605.10616, 2026.
- A. Arazi, E. Shapira, R. Reichart. *TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields.* NeurIPS 2025.
