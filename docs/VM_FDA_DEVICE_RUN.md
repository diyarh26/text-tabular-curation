# Full FDA Device evaluation on the existing GPU VM

Run **FDA Device only**, all five learners, four conditions and six folds: 120 evaluations.
The user has approved the full experiment. Do not run another candidate screen or TAR pilot.
Basic environment, data and launcher checks remain necessary before the long job.

## Existing machine and checkout

The Azure VM previously ran Steam successfully using a Tesla T4 (15,360 MiB), a checkout
at `~/text-tabular-curation`, a project `.venv`, configured GitHub access and tmux. Recheck
the actual machine; these observations are historical, not a guarantee of current state.

1. Inspect repository instructions, Git status, running tmux sessions and GPU processes.
   Preserve existing work, results and jobs. Do not start a second GPU job on an occupied T4.
2. Reuse healthy Git authentication and the existing checkout. Fetch and fast-forward safely;
   do not reset local work or display credentials. Verify that this document and the Device
   config/data are available. Clone the private `diyarh26/text-tabular-curation` repo only if needed.
3. Check GPU/driver, actual PyTorch CUDA computation, RAM and free disk. If no working NVIDIA
   GPU is available, stop and report it. Never rent, resize or stop a VM without permission.
4. Verify the pinned environment against `requirements/constraints.txt`. MulTaBench must be
   at `ff0c9c8b40a2085a06a7771190e53a261341802f`. Reuse a healthy environment; if missing,
   inspect `scripts/setup_gpu_vm.sh` and install the full benchmark extras. TAR-only Colab
   requirements do not install everything needed by all five learners.
5. Set `MPLBACKEND=Agg` before imports and for subprocesses. Check both TabPFN versions'
   checkpoint access. Previously both worked without new authentication; verify instead of
   requesting new tokens unnecessarily. Never put secrets in Git or logs.

## Fixed prepared data

- Config: `configs/fda_device_severity.yaml`.
- CSV: `data/processed/fda_device_severity.csv`.
- Preparation summary: `data/processed/fda_device_severity.summary.json`.
- Published provenance: `data/provenance/fda_device_severity/selected_rows.csv` and
  `text_changes.json`. The summary retains the original local preparation paths; these
  published copies contain the original trace/change-log content. Git normalizes text
  line endings to LF. The summary's original Windows change-log checksum therefore differs
  from the published LF checksum; this does not indicate a change to the audited edits.

Published provenance SHA256, after CRLF-to-LF normalization:

- `selected_rows.csv`: `834509665ae2d960e4b7ebac6da8891c39236e64f52d20776e64314a4bdebc10`.
- `text_changes.json`: `8bb70c197895b322e6c07f3c74c2a930ddeac45866e6e2f303fe1bf8775fbf8a`.

Expected CSV SHA256 **after replacing CRLF with LF**:

```text
b005b5ecf1bd1026c649f5989861cc6b8d3338ac1d09e67fefdc59c3793a6ec7
```

Config SHA256 with the same LF normalization:
`34442d58c3d3f2803612aa9810718aa229e63bc3ec4f25ec51278b7d144eb590`.

Require exactly 15,000 rows; target `classification`, with Class I = 988, Class II = 13,426,
and Class III = 586. These are recall severity classes, not device regulatory classes.

Text inputs: `product_description`, `reason_for_recall`, `distribution_pattern`.
Categorical inputs: `recalling_firm`, `city`, `state`, `country`, `voluntary_mandated`,
`initial_firm_notification`. Numerical inputs: `recall_year`, `recall_month`.
The target is not a feature. Source IDs belong only to provenance, never the model matrix.

The sample contains one randomly selected product per eligible recall event, followed by
a seed-42 random sample of 15,000 events. The published selection trace must have 15,000
unique event IDs and labels matching the model CSV in row order. No balancing or minimum
word count was used. Do not download, regenerate, resample, merge classes or change columns.

Missing inputs are intentional and remain for the existing preprocessing: reason 10,
distribution 1, city 2, state 1,689, country 3, voluntary/mandated 8, initial notification 360.
There are no missing targets. Do not impose complete-case filtering.

Twelve rows had text edits: eleven explicit recall-severity references were masked with
the same label-independent `[recall severity]` placeholder, and one pasted administrative
template was removed. Keep these edits and legitimate device/laser/clinical class wording.
No further text cleaning or manual truncation is authorized.

## Prepare a resumable Device launcher

The existing Steam and LinkedIn launchers are dataset-specific. **Neither may be run
unchanged on Device.** Prepare a dedicated `scripts/run_full_device.py`, reusing the tested
infrastructure and keeping changes small and separate from earlier experiments.

- Use `tt_curation.runner.run_all` and its actual pinned MulTaBench model, split,
  preprocessing, E5 and metric implementations. This is the same evaluation route as our
  earlier runs; it is not proof of native registration in MulTaBench's `benchmark.py`.
- Reuse the corrected `full_tar` helper from `scripts/run_full_linkedin.py`, including
  moving the idle frozen E5 off the GPU before TAR, persistent adapters/epoch checkpoints,
  and restoration of early-stopping state. Include all imported helper code in the manifest.
  Do not copy the legacy micro-batch path from Steam or the Colab pilot alone.
- Default to the official physical batch 256, effective batch 256, full precision,
  maximum 50 epochs and official early stopping. Preserve all other pinned hyperparameters.
  If memory fails, report the traceback and the proposed execution adjustment before
  changing settings. Smaller micro-batches require correct loss normalization, effective
  batch 256, a recorded protocol variant, and isolated caches/results; do not silently mix.
- No pilot epoch cap, arbitrary elapsed-time limit, performance-based early exit or silent
  learner omission. The official 50-epoch ceiling and early stopping still apply.
- Use `results/fda_device_full_e5/fda_device_severity/` with separate fold subdirectories,
  following the existing LinkedIn worker's layout. Run folds sequentially on one T4.
- Manifest: data/config/code hashes, installed package versions, MulTaBench commit,
  source Git commit, all six folds, all five models, all four conditions, E5 encoders,
  training settings and checkpoint/loss-normalization version. Reject incompatible reuse.
- Validate negative log loss correctly: the official multiclass `test_score` is
  **negative log loss**, higher is better; `test_error` is nonnegative log loss.
  Negative scores are valid and must not be rejected by Steam's AUROC-range check.
  Require finite values and `test_score == -test_error` within numerical tolerance.
  Our additional `auroc` is macro one-vs-rest AUROC and must lie in [0, 1].
- Check task = multiclass, dataset/model/condition/fold identities, E5 encoder, size = 10k,
  n_train = 10,000 and n_test = 5,000. Audit each fold's exact 20 required evaluations,
  not merely file counts. Stop on an error; a successful process exit is insufficient.
- Confirm that the launcher audit accepts a valid negative multiclass score and rejects
  missing/failed/wrong-dataset/TF-IDF results. Reuse the existing relevant lightweight
  checks; do not run another full smoke grid or pilot in place of the real experiment.

Commit and push launcher/check fixes separately before launching. Pull before editing if
the remote changed, preserving other work. Do not edit the Steam/LinkedIn experiment files
or course report. Record the final launch commit.

## Full experiment and monitoring

Learners: `tabm`, `cat`, `light`, `tabpfnv2`, `tabpfnv2p5`.
Conditions: `structured`, `unstructured`, `joint_frozen`, `joint_tar`.
Folds: 0 through 5, using the existing official repeated three-fold split implementation
and `size=10k`. Text-only and joint-frozen use `intfloat/e5-small-v2`, not TF-IDF.
TAR is independently fine-tuned once per fold on training data and shared across learners;
retain the official handling of all three text columns and internal validation split.
Fit preprocessing only on the appropriate training data; never use outer-test labels.

Initialize the verified launcher's manifest and run all folds inside a distinct tmux session,
for example `fda_device_full`, logging unbuffered stdout/stderr to `logs/fda_device_full.log`.
Use pipefail so failures are preserved. Confirm the actual process and new results before
handing back the session. Do not stop after writing a launch plan if checks pass.

Report the launch commit, GPU/environment checks, data verification, exact command, tmux
name, log/output paths and actual progress. When asked for status, read current files/logs.
After fold 0, save a 5-by-4 score table, both deltas, TAR duration/epochs, total fold time,
sampled peak GPU memory and a remaining-time estimate. Clearly label this preliminary.
Do not promise an unsolicited future chat message if the client cannot send one.

## Final audit and reporting

Require all **120 exact combinations**, status `ok`, valid metadata and finite metrics,
with six successful folds per model/condition. Generate `evaluation_report.csv`,
`curation_verdict.csv` and `summary.txt` from these results only. Partial reports must be
marked incomplete and must not retain a stale final verdict. Preserve all original run JSONs.

Official-score deltas use unrounded six-fold means:

- Joint Signal = joint_frozen - max(structured, unstructured).
- TAR Gain = joint_tar - joint_frozen.
- Strictly positive Joint Signal in at least 3/5 learners, and strictly positive TAR Gain
  in at least 3/5 learners, **counted independently**; the groups may differ.

Show log loss/official-score deltas and supplementary macro-AUROC tables distinctly.
Do not substitute macro AUROC for the pinned official multiclass verdict without an
explicit course requirement and discussion. The user's preferred +0.01 margin is an
aspiration, not an instruction to filter data or alter acceptance rules.

Only after a successful final audit, commit and push this experiment's manifests, run
JSONs, timing/audit summaries and reports. Do not commit environments, tokens, downloaded
model weights or large training checkpoints. Preserve checkpoints locally for resumption.
Explain the final independent criterion counts and any protocol deviations honestly.
