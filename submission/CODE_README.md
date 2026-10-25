# Historical multi-file packaging notes

Superseded by `track2_submission/README.md`. The current delivery has exactly
two Python files, `src/preprocess.py` and `src/evaluate.py`. These earlier notes
are retained in the repository only; they are not included in the delivery ZIP.

This bundle contains the curated 15,000-row CSV, the code used to create and
evaluate it, and its native MulTaBench integration. It excludes notebooks,
transfer tools, tests, previous candidates and duplicate source snapshots.

| Step | Entry point |
|---|---|
| Create the dataset from the raw download | `prep/linkedin.py` |
| Evaluate using the original batch-256 implementation | `scripts/run_full_linkedin_vm.py` (launched by `scripts/run_linkedin_vm256.sh`) |
| Aggregate the submitted experiment and calculate its deltas | `scripts/report_linkedin.py` |

The small `src/tt_curation/` package and two imported helper scripts
(`run_full_linkedin.py`, `colab_tar_compare.py`) are required by the original
evaluation entry point. Their historical filenames do not mean a Colab notebook
or a pilot must be run. The runtime, preparation script, configuration and
dependency pins are unmodified files from experiment revision
`bfe0ba3b846cb8bd719d43a586dc4d25b36088cd`, copied into runnable paths once.
`package_manifest.json` records their original SHA-256 hashes.

## Setup

Use Linux, Python 3.11, Git and a CUDA GPU with enough memory for physical batch
256. From the extracted bundle root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -c requirements/constraints.txt -e '.[bench]' \
  autogluon-features autogluon-core autogluon-common
export MPLBACKEND=Agg
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

MulTaBench is installed from its official source at
`ff0c9c8b40a2085a06a7771190e53a261341802f`, a pinned commit in master history.
Model downloads require internet and any access required by the model provider.

## Dataset creation

The prepared CSV is already at `data/processed/linkedin_salary.csv`; evaluating
it does not require the raw download. To recreate it, obtain the source download
and arrange its files under `data/raw/linkedin/` as listed in
`docs/LINKEDIN_DATASET_CARD.md`, then run:

```bash
.venv/bin/python prep/linkedin.py
```

This performs the recorded joins, salary filtering, money/contact masking,
description truncation and deduplication, then the fixed seed-42 sample. The
source data license is CC BY-SA 4.0; the source and attribution are in the card.

## Original evaluation and result calculation

The submitted experiment used `tt_curation` around official MulTaBench model,
preprocessing, splitting and E5/LoRA components. It used persistent training
checkpoints and zero data-loader workers. It was not executed through the native
`benchmark.py` entry point; the native integration below is supplied separately.

For a fresh rerun of this exact implementation, first record the extracted source
in a local Git repository because the original runner requires a source revision:

```bash
git init
git add .gitignore README.md LICENSE pyproject.toml src prep configs requirements scripts integration reproduction docs
git -c user.name='Submission reproduction' -c user.email='reproduction@example.invalid' commit -m 'Record supplied source'
CUDA_VISIBLE_DEVICES=0 bash scripts/run_linkedin_vm256.sh
```

This records a new source revision and creates a separate fresh experiment under
`results/linkedin_vm_batch256/linkedin_salary/`. It prepares the manifest, checks
physical batch 256 with a two-update GPU smoke test, discards the smoke weights,
then runs all 120 evaluations from scratch. Each completed fold prints a results
table. There is no automatic batch reduction or custom training time limit.

For that fresh experiment, calculate its reports again with:

```bash
.venv/bin/python scripts/run_full_linkedin_vm.py --results_root results/linkedin_vm_batch256 --report_only
```

For the separately supplied **original** result records, use:

```bash
.venv/bin/python scripts/report_linkedin.py
```

That command checks the original manifest and saved result hashes before writing
the performance CSV, per-fold CSV, deltas and summary. It refuses mixed results
and only issues a final verdict after all 120 original evaluations are present.
The code/data-only bundle does not contain those results or the paper.

## Run inside native MulTaBench

To evaluate the dataset through the official entry points instead:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_native_linkedin.sh
```

This prepares an isolated pinned upstream checkout, registers three feature
views of this one dataset, verifies the CSV/target and all six splits, creates
frozen/TAR embeddings with the official CLI, and runs unchanged `benchmark.py`.
`integration/multabench/` contains the reviewable dataset registration and local
data/cache routing patch. The upstream learners, preprocessing, evaluation,
splitting and LoRA implementation are unchanged. Native reproduction uses the
upstream four data-loader workers and writes separate results under
`artifacts/native_linkedin_evaluation/`; it does not relabel the original run.
`scripts/report_native_linkedin.py` calculates its course R-squared summaries
from native RMSE and the test-target variance without modifying raw results.

## Protocol and verdict

Both evaluation paths use five learners, four conditions and six official
splits (120 evaluations): structured, frozen text only, joint frozen and joint
TAR. TAR uses E5-small-v2, physical train/evaluation batch 256, accumulation 1,
float32, learning rate 0.0001, weight decay 0.001, rank-16 LoRA on the last three
layers, a 50-epoch cap and early-stopping patience 3. One TAR encoder is shared
by all five learners within each outer fold.

The course verdict uses raw six-fold mean R-squared on log salary:

- Joint Signal delta = joint frozen minus the better unimodal mean.
- TAR Gain delta = joint TAR minus joint frozen.
- Each delta must be strictly positive for at least 3/5 learners, independently.

Native negative RMSE is also retained. Original results and a fresh native rerun
are distinct experiments; no equality of their future numerical results is claimed.
