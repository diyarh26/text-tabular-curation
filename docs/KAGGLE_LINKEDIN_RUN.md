# LinkedIn full evaluation on Kaggle T4 x2

Import `notebooks/run_linkedin_full_kaggle.ipynb` into a **private** Kaggle notebook.
Enable Internet, choose **GPU T4 x2**, and enable a Kaggle secret named `GH_TOKEN`
with read access to `diyarh26/text-tabular-curation` (fine-grained token: repository
Contents read permission). Do not put tokens in cells, outputs, or Git.
An optional `HF_TOKEN` secret is read if present; public checkpoint access is attempted otherwise.

Use **Save Version > Save & Run All** for a background run with saved outputs.
Do not also launch an interactive full run. Check the account's actual quota.
Two devices do not pool VRAM: each worker has its own 16 GB T4.

The notebook downloads an exact GitHub source revision, installs a Python 3.11
environment with the pinned full dependencies, verifies both GPUs, tests all 20
toy learner/condition combinations on GPU 0, and checks the two regression checkpoints.
It then starts two processes:

| Visible physical GPU | Folds | Evaluations |
|---|---|---|
| 0 | 0, 2, 4 | 60 |
| 1 | 1, 3, 5 | 60 |

Each process sees **one GPU only**, before importing PyTorch/MulTaBench.
Both use the same prepared 15,000-row LinkedIn sample, five learners, four
conditions, six official splits, frozen E5-small-v2 and independent TAR adapters
per fold. Maximum 50 epochs, early stopping, batch 256, float32. It does not
change parameters automatically on out-of-memory or other failures.

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
