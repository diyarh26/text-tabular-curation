# LinkedIn full evaluation on Kaggle T4 x2

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

The notebook budgets ten hours from its first cell, including setup and smoke
tests, then interrupts workers and exports their saved files. This reserves time
within Kaggle's documented 12-hour GPU session limit for export; multiple sessions
may be needed. Platform termination before export can still lose current-session
changes. TAR can resume only from saved epoch checkpoints, not the current batch.
No fixed runtime or quota sufficiency is promised.

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
stored in that checkpoint is downloaded again. Data, configuration, runtime code,
package versions and scientific settings must match the saved manifest. Existing
successful evaluations are skipped. Checkpoints/results from the 10-epoch Colab
pilot are deliberately not reused.

The environment, model-download caches and toy results live in scratch, outside
the saved working output. They are reinstalled/downloaded in a new session.
The currently running Steam VM files are unchanged by these additions.

Official references: [Notebook runtime/output limits and saved outputs](https://www.kaggle.com/docs/notebooks),
[GPU quota guidance](https://www.kaggle.com/docs/efficient-gpu-usage),
[two-T4 announcement](https://www.kaggle.com/discussions/product-announcements/735239).
