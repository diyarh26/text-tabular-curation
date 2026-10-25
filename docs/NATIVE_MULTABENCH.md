# Native MulTaBench integration

LinkedIn is registered **inside an isolated copy of MulTaBench**, with a dataset-only patch against official master-history commit `ff0c9c8b40a2085a06a7771190e53a261341802f`.

The patch changes dataset registration, candidate-tier metadata, and local data/cache lookup. It leaves `benchmark.py`, native evaluation functions, embedding computation, learner classes, preprocessing, train/test splitting, E5 and LoRA training code unchanged. The complete patch is in `integration/multabench/linkedin_salary.patch`; the new native recipe files are in its `overlay/` directory. No global installed package is overwritten by the setup script.

## Prepare and validate

Install the project and benchmark dependencies using Python 3.11 and the constraints file. On a fresh Linux GPU machine, `bash scripts/setup_gpu_vm.sh` performs the environment setup.

From the project root:

```bash
python scripts/setup_native_multabench.py
export PYTHONPATH="$PWD/artifacts/multabench-native${PYTHONPATH:+:$PYTHONPATH}"
export MULTABENCH_LINKEDIN_DATA_DIR="$PWD/artifacts/native_linkedin_data"
python -m multabench.datasets.prepare_linkedin \
  --csv data/processed/linkedin_salary.csv \
  --output "$MULTABENCH_LINKEDIN_DATA_DIR"
python scripts/verify_native_linkedin.py --data_dir "$MULTABENCH_LINKEDIN_DATA_DIR"
```

The native preparation command rejects a different CSV and exports exactly the same 15,000 ordered rows and target into typed Parquet plus metadata. It creates three feature views of **one dataset**, rather than three independent dataset contributions:

| Native ID | Inputs | Condition |
|---|---|---|
| `REG_TEXT_LINKEDIN_SALARY_STRUCTURED` | 12 numeric/categorical columns | Structured |
| `REG_TEXT_LINKEDIN_SALARY_TEXT` | Title and description | Text only, frozen E5 |
| `REG_TEXT_LINKEDIN_SALARY` | All 14 inputs | Joint frozen or Joint TAR, selected by encoder |

The candidate IDs are local and do not claim acceptance into MulTaBench's published core tier or a hosted Hugging Face dataset. Category types are explicit so industry/state/etc. are never accidentally embedded as text.

## Actual official commands

After preparation, the official command recognizes the new IDs:

```bash
python artifacts/multabench-native/benchmark.py --model light \
  --dataset_name REG_TEXT_LINKEDIN_SALARY_STRUCTURED --fold 0 --device cpu \
  --output_dir artifacts/native_structured_check
```

For text conditions, create the raw embeddings using the unchanged native embedding CLI. Run module commands with the prepared `PYTHONPATH` and data-directory environment variable:

```bash
python -m multabench.embeddings.embed --encoder e5-small \
  --dataset_name REG_TEXT_LINKEDIN_SALARY \
  --output_dir "$MULTABENCH_LINKEDIN_DATA_DIR" --device cuda
python -m multabench.embeddings.embed --encoder e5-small-tar \
  --dataset_name REG_TEXT_LINKEDIN_SALARY --fold 0 --size 10k \
  --output_dir "$MULTABENCH_LINKEDIN_DATA_DIR" --device cuda
```

The frozen cache is shared by the full/text-only views because they have the exact same text values. The cache holds raw E5 outputs; each learner still fits its own native projection on its training rows. TAR caches remain per fold and are trained only on that fold's 10,000 outer training rows. Missing local caches fail explicitly; no request is made for a nonexistent published LinkedIn cache.

For a **separate fresh native reproduction**, on Linux with one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_native_linkedin.sh
```

This shell script only sequences the official embedding CLI and `benchmark.py`: five learners, four conditions, six folds, 120 evaluation JSONs. It does not substitute an external evaluation implementation. It retains physical train/eval batch 256, accumulation 1, float32, upstream weight decay 0.001, 50-epoch cap, patience 3 and four data-loader workers. There is no automatic batch/precision change or custom time cutoff. It writes to a separate native reproduction directory, not the active VM-v1 results.

The official TAR implementation uses temporary training checkpoints. Completed native caches and evaluation JSONs can be reused; an interrupted **unfinished native TAR fit** restarts that fit. The active VM-v1 continuation retains its separate persistent checkpoint implementation.

Do not use `--official`: that flag also requires the upstream leaderboard's clean master checkout and designated hardware. Our dataset patch and a T4 do not meet those leaderboard execution requirements.

## Reporting and the existing results

Native outputs retain official negative RMSE. `scripts/report_native_linkedin.py` additionally reports course R-squared as `1 - RMSE^2 / variance(y_test)`; upstream regression target preprocessing is identity, and the target is log salary. This is mathematically equivalent to R-squared from the same predictions. It does not change training or overwrite native JSONs.

The current experiment's results continue to be reported by `scripts/report_linkedin.py`. They keep their original `bfe0ba3` source revision and `linkedin_vm_physical256_v1` identity. Integration does not relabel those results as a native CLI run.

Validation completed locally:

- All three native views match the CSV runner exactly in row order, every value, dtypes and target.
- All six splits in every view match the original VM split hashes.
- The unchanged native evaluation function receives the correct inputs for all four conditions (contract test).
- Native frozen/TAR cache lookup, text-view aliasing, and missing-fold rejection work through the actual upstream cache reader (synthetic test embeddings only).
- A real LightGBM structured fold-0 evaluation through the unchanged official CLI matches our runner's RMSE within `1e-12`.
- Dataset tampering and a different source CSV are rejected.

These checks validate native integration and the relationship to the original data path. They do **not** claim a new full native 120-run GPU evaluation, a GPU text/TAR parity experiment, or proof of batch-256 fit on every T4 runtime. Source identity of the model/training files and the exact data/split checks support retaining the existing experiment with its execution differences disclosed.
