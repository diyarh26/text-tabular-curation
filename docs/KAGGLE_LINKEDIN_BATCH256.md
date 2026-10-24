# LinkedIn on Kaggle: official-source batch 256

Import `notebooks/run_linkedin_full_kaggle_batch256.ipynb`. Select GPU T4 x2, enable Internet,
keep the notebook private, and enable the existing `GH_TOKEN` secret for this notebook.
The notebook fetches all prepared data/code from the private GitHub repository at its
embedded immutable source revision. It installs Python 3.11 and the pinned full environment.
Existing optional `HF_TOKEN`/`TABPFN_TOKEN` secrets are used if available; access is checked.

This is a separate fresh experiment, using the same 15,000 salary rows, log_salary target,
two text columns, twelve structured columns and six official regression splits.
There are 120 evaluations: five learners x four conditions x six splits.

## Source verification

Checked the official repository on 2026-10-24. Master was
`fe9adafbe0cd415ae5804de2afc4277eeaea8212`; our reproducible installation remains pinned at
`ff0c9c8b40a2085a06a7771190e53a261341802f` to match the previous experiments.
The following files are byte-identical after LF normalization between those versions:

- `multabench/finetune/train_args.py` and `multabench/finetune/lora.py`.
- `multabench/e5/constants.py` and `multabench/e5/e5_finetune.py`.
- `multabench/benchmark/splits.py` and `multabench/benchmark/metrics.py`.
- `multabench/baselines/catboost.py`, `lgbm.py`, `tabm.py`, and `tabpfn.py`.

`baselines/abstract_model.py` differs only in two TODO comments, without executable changes.

| Setting | Official source and this notebook |
|---|---|
| Physical TAR train/evaluation batch | 256 |
| Gradient accumulation | 1 |
| Precision | float32; no fp16/bf16 |
| Learning rate / weight decay | 0.0001 / 0.001 |
| Maximum epochs / early-stopping patience | 50 / 3 |
| LoRA rank / alpha / dropout / layers | 16 / 32 / 0.1 / last 3 |
| E5 input length | 512 tokens |
| DataLoader workers | 4 |
| Frozen embedding inference batch | Native default 32 |
| Outer evaluation | Seed-42 repeated 3-fold split, two repeats; 10K train / 5K test |

Native loss, optimizer and scheduler remain unchanged. Checkpoint locations, restoration
of callback state, parking an idle frozen backbone on CPU, and reporting are operational additions.
The existing CSV loader/runner calls MulTaBench's actual implementations; it does not replace
the repository's hosted-dataset registry or become its `benchmark.py` entry point.

Primary sources: [LoRA defaults](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/multabench/finetune/train_args.py),
[native Trainer](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/multabench/finetune/lora.py),
[six splits](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/multabench/benchmark/splits.py).

## Execution and output

Each GPU gets one fold at a time from a shared queue. When it finishes, it takes the next
unstarted fold. Running folds are never divided across devices. If only one fold remains,
only its GPU performs that fold, preserving official single-GPU training.

Before the full run, each device must pass a real batch-256 TAR smoke test using fold-0
training inputs, two optimizer updates and validation. Temporary smoke adapters are discarded.
An OOM stops with a traceback; no batch, precision, sequence-length or hyperparameter fallback
is performed. Local CPU tests cannot establish actual T4 memory fit.

All outputs are under `/kaggle/working/linkedin_full_batch256/`. The supervisor prints a
per-fold R2 table in the running notebook cell and a cumulative completed-fold comparison.
It records timing and sampled GPU memory, and refreshes these ZIPs every 15 minutes:

- `linkedin_full_batch256_results.zip`: small reports, provenance and run JSONs.
- `linkedin_full_batch256_checkpoint.zip`: also includes adapters and native Trainer checkpoints.

There is no notebook-imposed runtime cutoff. Kaggle's own platform/session/quota limits apply.
To resume, attach this experiment's saved output (ZIP or unpacked folder) and rerun the notebook.
The source/package/data/protocol manifest must match. Older 16x16 LinkedIn outputs are rejected.
The `linkedin_native_physical256_v2` protocol also differs from the earlier VM v1 launcher,
which used zero DataLoader workers; do not merge those manifests or outputs.

R2 is used for the course verdict; original RMSE/negative-RMSE diagnostics remain unchanged.
Joint delta is joint frozen minus the better unimodal mean. TAR delta is joint TAR minus
joint frozen. Raw six-fold means need strictly positive gains for >=3/5 learners independently
on each criterion. A final verdict is produced only after all 120 expected identities pass audit.
