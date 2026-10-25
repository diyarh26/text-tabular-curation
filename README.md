# LinkedIn Salary: Text-Tabular Dataset Curation

Predict advertised annual salary from **job titles, descriptions, and structured job/company metadata**, and test whether the two modalities provide complementary information.

This project curates 15,000 LinkedIn job postings and evaluates five tabular learners under four conditions using [MulTaBench](https://github.com/alanarazi7/MulTaBench). E5-small-v2 supplies frozen or target-aware LoRA representations. Matching salary expressions are masked before evaluation.

NLP course 097215, Technion, Spring 2026, Track 2. Author: Diyar Husayyan.

**Status:** native dataset integration and local parity checks are complete. The latest available physical-batch-256 experiment contains **95/120 successful evaluations**. Final six-fold results and the separately written paper are pending.

## Dataset

| Property | Value |
|---|---|
| Rows | 15,000, fixed random sample, seed 42 |
| Text inputs | `title`, `description` |
| Structured inputs | Six categorical + six numerical/flag columns |
| Target | `log_salary`: natural log of annualized USD advertised salary |
| Source | [arshkon / LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings), CC BY-SA 4.0 |
| Data preparation | [prep/linkedin.py](prep/linkedin.py) |
| Curated CSV | [data/processed/linkedin_salary.csv](data/processed/linkedin_salary.csv) |

The cleaning procedure masks recognized monetary/contact expressions, removes empty and duplicate descriptions, limits descriptions to 500 words, and selects a fixed sample. E5 independently truncates at 512 **tokens**. Some structured values remain missing and are handled by the native learners. See the [dataset card](docs/LINKEDIN_DATASET_CARD.md) for every column, missingness, source files, transformations and limitations.

## Evaluation

Five learners: **TabM, CatBoost, LightGBM, TabPFN-v2, TabPFN-v2.5**.

| Condition | Structured inputs | Text representations |
|---|---|---|
| Structured | Included | None |
| Text only | Excluded | Frozen E5-small-v2 |
| Joint frozen | Included | Frozen E5-small-v2 |
| Joint TAR | Included | E5-small-v2 adapted to the fold's target using LoRA |

The official six splits are three-fold CV repeated twice, with 10,000 training and 5,000 test rows per split. Each text column is encoded and projected through the learner's native preprocessing. TAR trains one encoder per outer fold and shares it across learners.

Source-default TAR settings: physical train/evaluation batch **256**, accumulation **1**, float32, learning rate `1e-4`, weight decay `0.001`, LoRA rank 16 on the last three layers, and a 50-epoch cap with early-stopping patience 3.

The course criterion uses raw R-squared fold means:

- **Joint Signal:** joint frozen minus the better unimodal mean is positive for at least 3/5 learners.
- **TAR Gain:** joint TAR minus joint frozen is positive for at least 3/5 learners.

The criteria are counted independently. Native RMSE diagnostics are retained.

## Preliminary results

These R-squared means include **only completed folds 0-3** (80 evaluations). Another 15 baseline records exist on fold 4. The scores below are from the original VM-v1 batch-256 experiment, not a newly rerun native benchmark.

| Learner | Structured | Text only | Joint frozen | Joint TAR | Joint delta | TAR delta |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.495086 | 0.655175 | 0.694021 | 0.743217 | +0.038846 | +0.049196 |
| LightGBM | 0.484813 | 0.624861 | 0.663917 | 0.738696 | +0.039056 | +0.074778 |
| TabM | 0.472411 | 0.691113 | 0.721669 | 0.750158 | +0.030556 | +0.028489 |
| TabPFN-v2 | 0.499556 | 0.671406 | 0.697989 | 0.737967 | +0.026583 | +0.039977 |
| TabPFN-v2.5 | 0.507069 | 0.698593 | 0.724456 | 0.746758 | +0.025863 | +0.022302 |

Mean Joint Signal delta: **+0.032181**. Mean TAR Gain delta: **+0.042948**. Both are positive for 5/5 learners on these completed folds. **Final acceptance awaits all six folds.**

Raw records, manifest, per-fold CSV and preliminary summary are under [results/linkedin_batch256/linkedin_salary/](results/linkedin_batch256/linkedin_salary/). Regenerate reports with:

```bash
python scripts/report_linkedin.py
```

## Setup and native MulTaBench support

Use Python 3.11. Install the lightweight data/reporting tools:

```bash
python -m pip install -c requirements/constraints.txt -e ".[dev]"
python -m tt_curation validate configs/linkedin_salary.yaml
```

For GPU dependencies on Linux, run `bash scripts/setup_gpu_vm.sh`. The full environment uses the exact source/dependency pins in `pyproject.toml` and `requirements/constraints.txt`.

LinkedIn is registered inside an isolated official MulTaBench checkout using a small dataset-only patch. The three registered IDs are full, structured-only and text-only views of one dataset. The original **`benchmark.py`, evaluation functions, model classes, preprocessing, splitting and LoRA implementation are unchanged**. See [native setup, commands and verification](docs/NATIVE_MULTABENCH.md).

```bash
python scripts/setup_native_multabench.py
# Preparation and validation commands are in docs/NATIVE_MULTABENCH.md.
# For a separate full native reproduction on sufficient GPU memory:
CUDA_VISIBLE_DEVICES=0 bash scripts/run_native_linkedin.sh
```

The original saved experiment used `tt_curation` around official components and persistent checkpoints, with zero data-loader workers. The new native reproduction uses the source default of four workers. The [source audit](docs/MULTABENCH_COMPLIANCE_AUDIT.md) records this distinction. Existing results keep their original provenance; integration does not relabel or replace them.

## Continue the existing VM run

Use [resume_linkedin_vm256_kaggle.ipynb](notebooks/resume_linkedin_vm256_kaggle.ipynb) with the prepared private checkpoint bundle. It checks out the exact original runtime, preserves the completed results, and keeps physical batch 256. See [the continuation instructions](docs/KAGGLE_VM256_RESUME.md).

The original source is preserved in [reproduction/original_vm256/](reproduction/original_vm256/), with hashes and a recovery Git tag `original-linkedin-vm256-bfe0ba3`. Do not replace the running environment with the reorganized checkout.

## Repository guide

| Location | Purpose |
|---|---|
| `prep/`, `configs/`, `data/processed/` | Selected dataset creation, types and CSV |
| `integration/multabench/` | Reviewable native dataset patch and recipe |
| `scripts/`, `notebooks/` | Native reproduction, existing-run continuation, reporting and packaging |
| `src/tt_curation/` | Preserved official-component runner and data/report helpers |
| `results/linkedin_batch256/` | Original physical-batch-256 results and provenance |
| `reproduction/` | Unmodified original runtime snapshot |
| `paper/`, `submission/` | Separate-chat paper handoff and delivery instructions |
| `archive/` | Previous dataset candidates, pilots, alternate batch runs and old report |

## Checks and submission

```bash
pytest -m "not bench"
# After native preparation, including the real structured CLI parity test:
pytest tests/test_native_linkedin.py
```

The submission builder requires all 120 matching evaluations and a final paper PDF:

```bash
python scripts/build_submission.py --paper paper/report.pdf
```

See [submission instructions](submission/README.md) and the [assignment checklist](docs/SUBMISSION_REQUIREMENTS.md). No final paper or final submission ZIP has been produced yet.

## Attribution

Code: MIT, see [LICENSE](LICENSE). The LinkedIn dataset remains subject to its source's CC BY-SA 4.0 terms; the code license does not relicense the data.

Model and evaluation implementations are from [MulTaBench](https://github.com/alanarazi7/MulTaBench), pinned to `ff0c9c8b40a2085a06a7771190e53a261341802f`. Relevant methods include [E5](https://arxiv.org/abs/2212.03533) and [LoRA](https://arxiv.org/abs/2106.09685); the paper will provide the full learner and benchmark references.
