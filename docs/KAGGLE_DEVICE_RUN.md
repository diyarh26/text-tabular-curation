# FDA Device full evaluation on Kaggle

Import `notebooks/run_device_full_kaggle.ipynb` into a private Kaggle notebook.
Select GPU T4 x2, enable Internet, and enable a `GH_TOKEN` secret with Contents
read access to `diyarh26/text-tabular-curation`. Never paste secrets into cells.
The notebook also reads optional `HF_TOKEN` and `TABPFN_TOKEN` secrets. If model
access requires it, accept the TabPFN license and enable its account token.
Both classifier versions are checked before full training.

Use Save Version > Save & Run All and check your remaining GPU quota. This is a
fresh, separate Kaggle experiment: it neither stops nor resumes the Azure run.
No Azure results are imported. GPU 0 runs folds 0/2/4 and GPU 1 runs folds 1/3/5.
Each fold independently tunes one encoder shared by all five learners.

The protocol is the fixed 15,000-row FDA Device sample, five learners, four
conditions, six official folds, size 10k (10,000 train / 5,000 test), frozen
E5-small-v2, physical/effective batch 256, no gradient accumulation, float32,
and 50 maximum TAR epochs with official early stopping. Provenance hashes,
class counts, approved text edits and columns are verified. Missing inputs
are retained. No additional cleaning, sampling, pilot or screening is performed.
An out-of-memory error stops the run and preserves its traceback; the notebook
never switches to a smaller batch or precision automatically. Discuss any such
adjustment before preparing a separate protocol variant.

Detailed logs are in `/kaggle/working/fda_device_full_kaggle/logs/gpu0.log` and
`gpu1.log`. The running cell prints status every minute and full score tables
with both deltas after each audited fold. Official scores are NEGATIVE LOG LOSS;
macro one-vs-rest AUROC is shown separately. Single-fold and cumulative partial
reports are preliminary. Only all 120 exact, metadata-validated evaluations
produce a final verdict. Joint Signal and TAR Gain use unrounded six-fold means
and independently require positive deltas in at least three of five learners.

Download `/kaggle/working/fda_device_full_kaggle_results.zip` for analysis and
keep `fda_device_full_kaggle_checkpoint.zip` for resumption. Rolling snapshots
refresh every 15 minutes; incomplete checkpoints are excluded. A platform
termination can lose work since the last preserved snapshot. There is no
notebook wall-time cutoff; Kaggle controls session and account limits.

To resume, attach the previous saved notebook output via Add Input. A single
matching checkpoint ZIP is detected automatically; set RESUME_ZIP if multiple
versions are attached. Dataset, exact source commit, code, packages, splits and
training settings must match. Restore refuses to overwrite existing output.
Do not attach Azure, LinkedIn, Food or other protocol checkpoints.

Source: [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).
