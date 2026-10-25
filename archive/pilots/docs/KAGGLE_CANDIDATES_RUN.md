# Full LendingClub and FDA Food evaluations on Kaggle

For the new physical-batch-256 Food and Device experiments, use
[KAGGLE_FDA_BATCH256.md](KAGGLE_FDA_BATCH256.md). The previously published batch-64
notebook files remain pinned to their original source revision. Their log-loss
verdict labels predate the correction below; their saved AUROC values can be
reanalyzed without retraining.

Import these as **two separate private notebooks**:

- `notebooks/run_lendingclub_full_kaggle.ipynb`.
- `notebooks/run_food_full_kaggle.ipynb`.

Each notebook runs all five learners, four conditions and six official folds: 120 evaluations.
The prepared CSVs, configs and provenance are in the private GitHub repository. No raw-data
download or new sampling is needed. Each notebook downloads an exact published source commit;
on resumption it uses the exact commit stored with its experiment.
Preparation summaries retain their original local Windows paths and byte checksums. Published
provenance under `data/provenance/<dataset>/` contains the same trace/edit content with Git's
LF text line endings; an original CRLF JSON checksum can therefore differ.

## Start

1. Set **GPU T4 x2** and **Internet On** in each notebook.
2. Add the Kaggle secret `GH_TOKEN`, with Contents read access to
   `diyarh26/text-tabular-curation`, and **enable it for each notebook**. A secret saved in
   the account but not enabled for this notebook is insufficient.
3. For TabPFN-v2.5, accept the model license at https://ux.priorlabs.ai and enable
   `TABPFN_TOKEN` if needed. `HF_TOKEN` is a separate optional credential. Both checkpoint
   versions are tested on this notebook's classification task before the long run.
4. Use **Save Version > Save & Run All** for each background run. Check the actual remaining
   account quota and available concurrency; do not also start interactive full runs.

An isolated Python 3.11 environment installs the pinned full benchmark dependencies.
Both GPUs must pass actual CUDA computation and all five learners must import. Short CPU
checks validate launcher/report/resume behavior and corrected Trainer accumulation. There
is no additional dataset screen, 20-evaluation GPU toy grid or TAR pilot.

The approved Kaggle memory adjustment is physical micro-batch 64 with 4-step accumulation,
effective batch 256, float32. The tested summed-loss normalization handles the whole optimizer
update's example count, including short final batches. Official learning rate, LoRA settings,
50-epoch cap and early stopping remain unchanged. Numerical differences from a physical
batch of 256 remain possible; the variant is recorded in the manifest. No automatic memory
fallback or arbitrary wall-time cutoff is applied.

GPU 0 owns folds 0/2/4; GPU 1 owns folds 1/3/5. Each worker sees one GPU only. Two T4s do not
pool VRAM. Frozen text/joint conditions use E5-small-v2. TAR is independently tuned per fold
and shared across all five learners. Outer test labels are never used for training.
The actual pinned MulTaBench wrappers, preprocessing, splits and metric are called through
`tt_curation`; this is not a claim of native dataset registration in `benchmark.py`.

## Fixed samples and metrics

| | LendingClub | FDA Food |
|---|---|---|
| Config basename | lendingclub_longdesc_no_purpose | fda_food_severity |
| Rows | 15,000 | 7,918 |
| Classes | Fully Paid 12,822; Charged Off 2,178 | I 3,182; II 4,065; III 671 |
| Text/structured inputs | 3 / 9 | 3 / 8 |
| Primary curation metric | AUROC | Macro one-vs-rest AUROC |
| Diagnostic repository metric | Same AUROC | Log loss, stored as negative log loss |

LendingClub uses complete selected inputs and descriptions with strictly more than 50 content
words, sampled from the full eligible pool. `purpose` is absent; `title` remains text. Food uses
one product per recall event, retains the agreed missing ancillary values and masks the audited
explicit recall-severity reference. Do not merge classes, rebalance, filter or resample.

The Food folds use about 5,279 training and 2,639 test rows, with one-row differences. The code
records the exact official indices, hashes and sizes; it must not impose LendingClub's 10K/5K
counts on Food. All six split definitions retain every class on both sides.

## Live fold tables

**After each fold completes all 20 evaluations and passes its audit, its full five-learner table
prints directly in the running notebook cell.** It does not wait for ZIP export or for both GPU
workers to finish. The supervisor checks for a newly published fold table every five seconds.
Because two workers run independently, fold 1 may print before fold 0.

Each table includes structured, text, joint frozen, joint TAR, Joint Signal delta and TAR delta.
It prints the independent learner counts passing each criterion. A second table shows cumulative
means over **fully completed folds only**; partial next-fold baselines never enter that comparison.
The updated runtime prints Food's macro-AUROC curation table first, followed by
the repository's negative-log-loss diagnostic table. Original pinned notebooks
retain their original reporting order until explicitly regenerated.

These early tables are marked preliminary. They inform the user without triggering a
performance-based early exit. Course acceptance uses all six folds and unrounded AUROC
means: Joint Signal >0 in at least 3/5 learners and TAR Gain >0 in at least 3/5 learners, counted
independently. The qualifying learners may differ. The paper rule is separately
reported with rounded condition means and both deltas >=0.001 on the same >=3/5
learners. Do not equate AUROC and log-loss delta scales.

## Output and resumption

| | LendingClub | FDA Food |
|---|---|---|
| Working root | /kaggle/working/lendingclub_full | /kaggle/working/fda_food_full |
| Result ZIP | lendingclub_full_results.zip | fda_food_full_results.zip |
| Checkpoint ZIP | lendingclub_full_checkpoint.zip | fda_food_full_checkpoint.zip |

Within the working root, `<dataset>/foldN_report.txt` holds each live table,
`<dataset>/progress_report.txt` holds the latest completed-fold comparison, and each fold
has separate run JSONs, checkpoints and timing information. Worker logs are `logs/gpu0.log`
and `logs/gpu1.log`. The smaller results ZIP contains reports, manifests, run JSONs and logs;
the checkpoint ZIP additionally contains adapters and Trainer checkpoints.

Rolling ZIPs refresh every 15 minutes without stopping training. Live copies exclude incomplete
Trainer checkpoints; changed-file detection retains the previous checkpoint ZIP if a copy races
with a writer. Final exports run after both workers stop. Kaggle still enforces its own external
session and quota limits; preserving a saved notebook output/checkpoint is essential.

To resume, attach the previous saved notebook output through **Add Input**, then rerun the same
notebook. Exactly one matching checkpoint ZIP is auto-detected. Set `RESUME_ZIP` if there are
several. Other datasets' ZIP names are ignored. Restoration validates paths, dataset, protocol
and source revision before extracting; it never overwrites nonempty working output.

The manifest rejects changes to data, config, runtime code, packages, split definitions,
micro-batch or other scientific settings. Successful evaluations are skipped; TAR resumes from
the latest saved epoch checkpoint. Failed attempts remain archived for diagnosis and are retried.
No legacy LinkedIn TAR, pilot or TF-IDF-screen results are imported.

The current notebooks start fresh batch-64 experiments. Do not attach earlier batch-16
candidate checkpoints to these files: the manifest rejects that change. Preserve the old
notebook/output if an earlier run has already started. No automatic protocol migration occurs.

Only a successful audit of the exact 120 combinations creates a final `curation_verdict.csv`.
Partial results have an explicit incomplete summary, with no stale final verdict. Keep all run
JSONs, manifests and reports. Share the results ZIP for analysis; retain the checkpoint ZIP to resume.
