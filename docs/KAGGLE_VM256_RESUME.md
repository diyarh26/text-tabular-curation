# Finish the original LinkedIn VM experiment on Kaggle

Use these two local deliverables together:

- `resume_linkedin_vm256_kaggle.ipynb`
- `linkedin_vm256_resume_checkpoint.zip`

Create a **private Kaggle Dataset** from the checkpoint ZIP and attach it to the notebook.
Kaggle may expose the ZIP or its extracted folder; both are supported. Do not attach the
old 16x16 LinkedIn checkpoint or the failed fresh native-v2 Kaggle experiment instead.
Enable Internet, select T4 x2, enable the existing GH_TOKEN secret for this notebook,
and use Save Version > Save & Run All once. One NVIDIA GPU also works, sequentially.
Optional existing HF_TOKEN/TABPFN_TOKEN secrets are used when available.

## Preserved experiment

The notebook continues the original VM source commit
`bfe0ba3b846cb8bd719d43a586dc4d25b36088cd` and protocol
`linkedin_vm_physical256_v1`. It fetches a real immutable Git checkout, installs the
original pinned environment, and compares data/config hashes, all 13 source hashes,
all six split hashes, package versions and all training parameters with the saved manifest.
It does not use the newer native-v2 launcher or migrate the old manifest.

The existing 95 run JSONs and original manifest are protected by byte-level SHA256 checks:

| Fold | Already saved | Work remaining |
|---|---:|---|
| 0–3 | 80 evaluations | None |
| 4 | 15 baselines | Resume TAR after epoch 1 / update 71; five downstream evaluations |
| 5 | None | Fifteen baselines, one shared TAR training, five TAR evaluations |

The bundle contains the 95 results and full fold-4 Trainer checkpoint, including model,
optimizer, scheduler, RNG, training arguments and early-stopping callback state.
Unused model weights from completed folds 0–3 are omitted from the upload; those original
backups stay untouched. The supplement is merged into a NEW bundle without editing either input.

Physical train/evaluation batch stays 256, accumulation 1, float32, learning rate 0.0001,
weight decay 0.001, native LoRA settings, maximum 50 total epochs, and patience 3.
The original VM-v1 loader behavior (zero workers in its persistence helper) is retained.
The cap is not reset to 50 additional epochs when resuming fold 4.

Best-checkpoint paths originally point into `/home/student/...`. The notebook changes only
that location field in a Kaggle working copy and preserves the original Trainer JSON under
`transfer_metadata/original_trainer_states/`. Model/optimizer/scheduler/RNG/callback state
and the experiment manifest are not rewritten. Missing or inconsistent input stops the run;
there is no fresh-experiment fallback and no automatic deletion of incomplete checkpoints.

## GPU execution

On two GPUs, fold-4 TAR starts in a clean process while fold-5 baselines run on the other GPU.
After its baselines finish, fold-5 TAR starts in another clean process. No baseline model
or access-probe estimator is fitted inside a TAR worker process. Each process sees one GPU.
This avoids carrying baseline allocations into TAR; it does **not** guarantee batch 256 fits.

There is no short two-step training smoke test. The actual resumed/remaining TAR training
is monitored. On OOM, settings stay unchanged, workers stop, and saved work is exported.
No automatic batch, precision, sequence-length, model, or fold adjustment is made.
Local CPU validation cannot establish actual Kaggle T4 memory fit.

## Results and resumption

Output root: `/kaggle/working/linkedin_vm256_resume/`.
Live R2 tables appear when folds complete, using only fully completed folds in cumulative means.
Final acceptance uses raw six-fold R2 means and independent >=3/5 criterion counts.
The original raw RMSE diagnostics remain in the run JSONs. New timing metadata describes the
Kaggle phase/session portion; it does not claim to include all elapsed VM training time.

ZIP snapshots refresh every 15 minutes without a notebook-imposed wall-time cutoff:

- `linkedin_vm256_resume_results.zip`: small results, reports and provenance.
- `linkedin_vm256_resume_checkpoint.zip`: also contains adapters/native Trainer state.

Attach the latest checkpoint output to resume again after a platform/manual interruption.
Kaggle's own session/quota limits still apply. The final experiment has mixed VM/Kaggle
hardware execution; disclose this provenance when reporting the final six-fold evaluation.
Do not publish checkpoint weights or private inputs to GitHub.
