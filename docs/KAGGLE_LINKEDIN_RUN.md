# LinkedIn full evaluation on Kaggle T4 x2

## Resume the corrected 115-result checkpoint (October 24)

Use `notebooks/resume_linkedin_remaining_kaggle.ipynb` with the downloaded
`linkedin_full_checkpoint.zip`. This archive already has the corrected loss
normalization and must **not** undergo the historical migration described below.

Upload the ZIP as a **private Kaggle Dataset**, attach it with **Add Input**, and
select GPU T4 x2, Internet On, and enabled GH_TOKEN (plus optional TABPFN_TOKEN /
HF_TOKEN if needed). The notebook accepts a retained ZIP or Kaggle's unpacked
dataset contents. Use Save Version > Save & Run All. If the original run remains
active, let it finish instead of launching another copy.

Initial validation must retain 115 successful evaluations and identify fold 5's
epoch-30 checkpoint (step 2130). The exact original source revision remains
`b75bdbb835d2ec0da13e076fce5249933879127b`, micro-batch remains 16, accumulation 16,
effective batch 256, float32, cap 50 and original early stopping. A later compatible
checkpoint may contain more progress. Missing/incompatible input stops execution;
there is no fallback that starts a fresh full experiment.

Only unfinished work runs. One worker finds all its folds complete and exits;
the other skips completed folds and resumes fold 5. One idle GPU is expected.
The previous 20-case toy smoke is not repeated. At the end, the notebook prints
the correct unrounded **R²** comparison and independent criterion counts, writes
`summary_r2.txt`, `curation_verdict_r2.csv`, and `curation_summary_r2.json`, then
re-exports both ZIPs including those reports. Original run JSONs are preserved.

The downloaded archive's 115 result records and fold-5 model, optimizer, scheduler,
RNG and Trainer-state files were checked locally, including ZIP CRC validation.
Notebook guards/restoration/reporting have CPU tests. The real GPU resume occurs
on Kaggle; its strict manifest check retains the original pinned GPU environment.

## Corrected restart after the October 23 overnight run

Use the newly generated notebook from this revision, not a copy downloaded before
the correction. Keep the old output ZIPs. Import the new notebook into Kaggle,
enable GPU T4 x2, Internet, GH_TOKEN and TABPFN_TOKEN, then Save & Run All.
Optionally attach the old notebook output with Add Input: the notebook will
validate and reuse the 90 unaffected baseline evaluations. Without an old input,
it runs all 120 evaluations from scratch. In both cases, **all six TAR encoders
train fresh**, with the corrected loss. No affected TAR results or optimizer
checkpoints enter the corrected experiment.

The original micro-batch adapter accumulated mean cross-entropy losses without
the required normalization. Under the pinned Trainer this scaled gradients by
approximately 16. This affects legacy batch-16 TAR runs, including the earlier
Colab pilot. Steam's full physical-batch-256 run and LinkedIn's three baseline
conditions are unaffected. This was an implementation bug, not merely a logging
difference. Old TAR scores are retained only as results of that affected protocol.

The correction computes each micro-batch's summed cross-entropy divided by the
actual example count for the entire optimizer update. Tests exercise the pinned
Trainer and upstream classification head: full-batch versus accumulated loss and
gradients, short final batches, optimizer/scheduler updates, validation loss,
and an interrupted checkpoint restart versus uninterrupted training. The notebook
runs those CPU checks before its 20-case GPU smoke test and the full experiment.
Dropout and GPU floating-point execution can still produce numerical differences
from a physical batch of 256; effective batch size does not imply identical scores.

The manifest records `accumulation_loss=ce_sum_over_effective_batch_items_v1`.
Only the specifically identified legacy TAR worker change can be migrated; data,
configuration, other model code and package versions must still match. The old
experiment is kept in a sibling `linkedin_full_legacy_tar_backup_<timestamp>`.
All imported baseline files have hashes recorded in `baseline_reuse.json`.

Import `notebooks/run_linkedin_full_kaggle.ipynb` into a **private** Kaggle notebook.
Enable Internet, choose **GPU T4 x2**, and enable a Kaggle secret named `GH_TOKEN`
with read access to `diyarh26/text-tabular-curation` (fine-grained token: repository
Contents read permission). Do not put tokens in cells, outputs, or Git.
For TabPFN-v2.5, accept the model license at [Prior Labs](https://ux.priorlabs.ai)
and add the account API key as a `TABPFN_TOKEN` Kaggle secret. It is a separate
credential from `HF_TOKEN`. The pinned TabPFN library requires this license check
when downloading gated weights; availability on another machine does not establish
fresh-download access. An optional `HF_TOKEN` secret is also read if present.

Use **Save Version > Save & Run All** for a background run with saved outputs.
Do not also launch an interactive full run. Check the account's actual quota.
Two devices do not pool VRAM: each worker has its own 16 GB T4.

The notebook downloads an exact GitHub source revision, installs a Python 3.11
environment with the pinned full dependencies, verifies both GPUs, tests all 20
toy learner/condition combinations on GPU 0, and checks both classification and
regression checkpoints before TAR training.
It then starts two processes:

| Visible physical GPU | Folds | Evaluations |
|---|---|---|
| 0 | 0, 2, 4 | 60 |
| 1 | 1, 3, 5 | 60 |

Each process sees **one GPU only**, before importing PyTorch/MulTaBench.
Both use the same prepared 15,000-row LinkedIn sample, five learners, four
conditions, six official splits, frozen E5-small-v2 and independent TAR adapters
per fold. Maximum 50 epochs, early stopping, float32. The approved Kaggle configuration
uses micro-batch 16, accumulation 16, effective batch 256. It does not
change parameters automatically on out-of-memory or other failures.
Before tuning a separate E5 encoder, the launcher moves the unused frozen E5
weights to CPU and releases unused CUDA allocations. Frozen embedding vectors and
weights are retained; normal encoding moves that model back to the requested
device when needed. This does not change batch size, precision or training parameters.

The notebook's settings cell now defaults to the explicitly approved `MICRO_BATCH = 16`.
Both smoke and full runs use micro-batches of 16
and 16-step gradient accumulation for effective batch 256, still in float32 with
the same 50-epoch cap. This changes batch execution and can cause numerical
differences; it is recorded in the manifest. Set `MICRO_BATCH = 256` for the reference
physical batch on hardware where it fits; the command-line worker still defaults to 256.
An existing experiment with a different micro-batch cannot be silently mixed.

When resuming the failed batch-256 attempt with the new batch-16 notebook, the
notebook verifies data/configuration, model code, dependencies and all other
scientific settings. It preserves the old directory as a sibling named
`linkedin_full_batch256_backup_<timestamp>`, creates a new batch-16 manifest and
copies only successful, metadata-validated structured/text/joint-frozen JSONs.
TAR results, adapters and training checkpoints stay in the backup. A
`baseline_reuse.json` records source provenance and imported file hashes.

Fold outputs and training checkpoints have separate directories to avoid races.
Both workers are stopped on a saved model error or failed process. Completed
evaluations remain resumable. Every minute the supervisor prints progress and GPU
utilization/memory; detailed output is in `linkedin_full/logs/gpu0.log` / `gpu1.log`.
Report aggregation happens only after both writers stop.

There is **no notebook-imposed time cutoff**. The supervisor keeps running until
completion, an error or an external/manual interruption. The official 50-epoch
training cap and early stopping remain in place. Kaggle controls its own session
and quota limits. Every 15 minutes the supervisor atomically refreshes rolling
ZIPs without stopping training. Live snapshots exclude incomplete checkpoints and
retain the previous ZIP if a selected file changes during copying. Final export
includes logs after both workers stop. TAR resumes from a saved epoch checkpoint,
not the current batch; work since the last successful snapshot may be lost.

Outputs under `/kaggle/working`:

- `linkedin_full_results.zip`: results, manifests and logs for review.
- `linkedin_full_checkpoint.zip`: those files plus TAR adapters/optimizer/epoch
  checkpoints for resumption. Keep this ZIP or the saved notebook output.
- `linkedin_full/linkedin_salary/evaluation_report.csv`: available successful runs;
  inspect `n_folds` if incomplete.
- `summary.txt` and `curation_verdict.csv`: final verdict only after all 120
  expected identities succeed and pass metadata/finite-regression-metric checks.
  Incomplete runs receive an explicit incomplete summary without a verdict CSV.

To resume, attach the previous version's notebook output with **Add Input**, then
rerun. A single `linkedin_full_checkpoint.zip` under `/kaggle/input` is found
automatically; if several exist, set `RESUME_ZIP` explicitly. The source commit
stored in that checkpoint is downloaded again, except that legacy runs fetch the
corrected source and undergo baseline-only migration. Data, configuration, runtime code,
package versions and scientific settings must match the saved manifest. Existing
successful evaluations are skipped. Checkpoints/results from the 10-epoch Colab
pilot are deliberately not reused.

The environment, model-download caches and toy results live in scratch, outside
the saved working output. They are reinstalled/downloaded in a new session.
The currently running Steam VM files are unchanged by these additions.

Official references: [Notebook runtime/output limits and saved outputs](https://www.kaggle.com/docs/notebooks),
[GPU quota guidance](https://www.kaggle.com/docs/efficient-gpu-usage),
[two-T4 announcement](https://www.kaggle.com/discussions/product-announcements/735239).
