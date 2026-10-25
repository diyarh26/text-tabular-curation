# FDA Device and Food on Kaggle: physical batch 256

Use two separate private notebooks:

- `notebooks/run_device_full_kaggle_batch256.ipynb`
- `notebooks/run_food_full_kaggle_batch256.ipynb`

Each evaluates the existing prepared CSV with five learners, four conditions and
six official splits: **120 evaluations**. No further cleaning or sampling occurs.
GPU 0 owns folds 0/2/4 and GPU 1 owns folds 1/3/5. Each worker uses one GPU; GPU
memory is not pooled. Completed fold tables print directly in the running cell,
followed by cumulative means over fully completed folds only.

## Explicit protocol choices

The user chose the current official MulTaBench source defaults at commit
`ff0c9c8b40a2085a06a7771190e53a261341802f`, including its six-fold scheme, rather
than the paper's five-seed experiment. The official model/preprocessing/split code
is called through `tt_curation` with the prepared CSV and declared feature types;
this does not claim that the datasets are registered in native `benchmark.py`.

| Setting | These notebooks |
|---|---|
| Encoder | `intfloat/e5-small-v2`, separately tuned per outer fold |
| Physical training/evaluation batch | 256 |
| Gradient accumulation | 1 |
| E5 learning rate | 0.0001 |
| Weight decay | **0.001**, the official source default explicitly selected by the user |
| LoRA | Rank 16, alpha 32, dropout 0.1, last three layers |
| Token limit | 512 tokens, official tokenizer truncation |
| TAR budget | Up to 50 epochs, early stopping patience 3, best validation-loss checkpoint |
| Precision | float32; no mixed-precision override |
| Primary curation metric | Macro one-vs-rest AUROC |
| Diagnostic metric | Pinned repository's negative log loss |

The paper states weight decay 0.01. These notebooks preserve 0.001 deliberately;
they must not be described as an exact reproduction of every paper setting.
The manifest records all actual E5 TAR parameters, dataset/source/package hashes,
outer split hashes, physical batch and accumulation, and both acceptance rules.

The course verdict uses raw mean AUROC improvements >0 in at least three learners
per criterion, counted independently, following the supplied instructor replies.
The paper-style verdict rounds each condition mean to three decimals, then needs
both improvements >=0.001 in the same three learners. That rule is applied to
these six course folds and is labelled accordingly. Final verdicts require all
120 evaluations to pass their audit; single-fold results remain preliminary.

## Run

1. Import the desired notebook and select **GPU T4 x2**, **Internet On**.
2. Add the Kaggle secret `GH_TOKEN`, grant Contents read access to the private repo
   `diyarh26/text-tabular-curation`, and enable it for this notebook. Do not put the
   token in a cell. Enable optional `TABPFN_TOKEN`/`HF_TOKEN` if required for access.
3. Use **Save Version > Save & Run All**. The notebook verifies CUDA, model access,
   prepared data, splits and the pinned environment before starting the workers.
4. Download the final results ZIP. Keep the checkpoint ZIP for interruption recovery.
   Once outputs are saved, stop any unused interactive GPU session to preserve quota.

| Output | Device | Food |
|---|---|---|
| Working folder | `fda_device_full_kaggle` | `fda_food_full_batch256` |
| Results ZIP | `fda_device_full_kaggle_results.zip` | `fda_food_full_batch256_results.zip` |
| Checkpoint ZIP | `fda_device_full_kaggle_checkpoint.zip` | `fda_food_full_batch256_checkpoint.zip` |

Folders are under `/kaggle/working`. The new Food folder/ZIP name isolates it from
the earlier batch-64 experiment. Attach only checkpoints from this exact notebook
revision and protocol. A manifest mismatch stops restoration; settings are never
migrated silently. Exact source commits are embedded in each generated notebook.

Both raw metrics remain in per-run JSONs and `evaluation_report.csv`. Final
`curation_verdict.csv` uses AUROC and the course rule; `paper_verdict.csv` uses the
paper-style rule. `curation_summary.json` records the two separate decisions.

An out-of-memory error stops the run and preserves available results and logs.
There is no automatic reduction in batch, token limit, epoch budget or precision.
Batch 256 must fit one T4; local CPU tests cannot establish Kaggle VRAM capacity.
There is no added notebook wall-time cutoff; Kaggle's session/quota limits apply.
Rolling ZIP snapshots refresh every 15 minutes and include only completed Trainer
checkpoints. Keep a saved notebook output/checkpoint to resume across sessions.
