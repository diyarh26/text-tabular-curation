# LinkedIn Salary: July native MulTaBench

Technion NLP 097215, Track 2. Official source: [MulTaBench](https://github.com/alanarazi7/MulTaBench/tree/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca), detached commit `599bd6a5631c96f8aef297cc4cb6e4c197ae0dca` (26 July 2026, 21:26:40 +03:00). Install its complete `requirements.txt` with Python 3.11; no newer installed MulTaBench modules are used.

**Status:** all 100 native evaluations are complete, backed up and verified. Both Joint Signal and TAR Gain are positive for 5/5 learners using unrounded scores, meeting the staff requirement of at least 3/5 independently. Only the separately written `final_report.pdf` remains to be supplied. The original six-A40 Pod exhausted its prepaid balance during the final evaluation; its 99 completed results were preserved unchanged. Only TabPFN-v2.5 joint TAR, fold ID 4, was rerun on one A40 with identical native settings and a new W&B ID. Native early stopping ended that TAR fit at epoch 28. The interrupted attempt is excluded. Both temporary Pods were stopped and removed after verified external backups.

Mean native R^2 across five repetitions (display rounded; CSV retains full precision):

| Learner | Structured | Text only | Joint frozen | Joint TAR | Joint gain | TAR gain |
|---|---:|---:|---:|---:|---:|---:|
| TabM | 0.452062 | 0.685794 | 0.712217 | 0.746061 | +0.026423 | +0.033845 |
| CatBoost | 0.484192 | 0.654352 | 0.686998 | 0.736704 | +0.032645 | +0.049706 |
| LightGBM | 0.470307 | 0.626776 | 0.662384 | 0.729198 | +0.035608 | +0.066814 |
| TabPFN-v2 | 0.487042 | 0.667419 | 0.690777 | 0.731955 | +0.023358 | +0.041178 |
| TabPFN-v2.5 | 0.484076 | 0.692584 | 0.717093 | 0.740511 | +0.024509 | +0.023418 |

The 15,000-row CSV is unchanged. SHA-256 after LF normalization: `8880f84563535c60eda01be5b52010dea6f80e318f228ade00166d7cb66d57fa`. It contains mixed-language descriptions; no language filtering or translation is performed. Inputs are `title`, `description`, and the twelve supplied structured columns; target is natural-log USD salary, `log_salary`. Source: [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings).

`preprocess.py` reads the raw relational CSVs, preserves the confirmed cleaning/masking order, deduplicates cleaned 500-word descriptions, samples without replacement with seed 42, and groups categories occurring fewer than 20 times after sampling. It preserves the specified missingness. It refuses to write if exact row/value parity with the approved CSV fails. `evaluate.py prepare` adds only a native `UrlDatasetID` entry, its local CSV mapping and a curation annotation in the external official checkout. The actual `benchmark.py` entry point performs all learning, detection, preprocessing, splitting and metrics.

The July analysis uses five repetitions, native fold IDs **0-4**: **100 evaluations** across `tabm`, `cat`, `light`, `tabpfnv2` and `tabpfnv2p5`. These are repeated native random splits, not disjoint CV folds. Each repetition samples 11,500 rows, then trains on 10,000 and tests on 1,500. TabM/CatBoost/LightGBM further use 9,000 fit + 1,000 validation rows; TabPFNs retain all 10,000 fit rows.

| Condition | Native CLI controls |
|---|---|
| Structured | `--multimodal_state no_text --tune_e5 no` |
| Frozen text | `--multimodal_state text_only --tune_e5 no` |
| Joint frozen | `--multimodal_state all --tune_e5 no` |
| Joint TAR | `--multimodal_state all --tune_e5 yes` |

Production retains E5-small-v2, native PCA 30, physical batch 256, at most 50 epochs, learning rate 0.0001, weight decay 0.001, LoRA rank 16, three text layers and patience 3. TAR is fitted independently inside each learner's preprocessing: **25 TAR fits** total. There is no added shared TAR or embedding cache. The native regression score is R^2 directly; the CSV preserves native R^2/MSE/RMSE and full-precision W&B exports. Joint delta is mean joint-frozen R^2 minus the better unimodal mean; TAR delta is mean joint-TAR minus mean joint-frozen. Per staff clarification, each delta must be strictly positive for at least 3/5 learners independently, using unrounded scores.

Run from this directory on the approved GPU Pod. Keep `/workspace/july-native` on a verified persistent mount, outside this submission. The optional `/opt/july-native-venv` installation uses container SSD for dependencies; source, logs and results remain on the persistent mount. Put `WANDB_API_KEY`, `WANDB_ENTITY`, `TABPFN_TOKEN` and optional `WANDB_PROJECT=linkedin-salary-july-native` in an external secret file. The LinkedIn loader uses the staged local CSV and does not download from Kaggle. Use the existing PriorLabs account to accept the TabPFN 2.5 license and obtain `TABPFN_TOKEN`. The checked E5 and TabPFN checkpoint files currently allow anonymous Hugging Face downloads; `HF_TOKEN` can remain blank. Run `auth-check` before renting; it checks credentials/license only and never loads or fits a model.

```bash
python3.11 src/evaluate.py --runtime /workspace/july-native prepare
python3.11 src/evaluate.py --runtime /workspace/july-native install --python python3.11 --system-tools --venv-dir /opt/july-native-venv
PY=/workspace/july-native/venv/bin/python
$PY src/preprocess.py --raw /workspace/raw/linkedin --audit /workspace/july-native/preparation-audit.json --output /workspace/july-native/recreated.csv
$PY src/evaluate.py --runtime /workspace/july-native inspect
$PY src/evaluate.py --runtime /workspace/july-native --secrets /workspace/credentials.env auth-check
$PY src/evaluate.py --runtime /workspace/july-native --secrets /workspace/credentials.env preflight --gpus 0,1,2,3,4,5 --persistent-mount /workspace
$PY src/evaluate.py --runtime /workspace/july-native --secrets /workspace/credentials.env smoke --gpus 0 --persistent-mount /workspace
$PY src/evaluate.py --runtime /workspace/july-native --secrets /workspace/credentials.env start --gpus 0,1,2,3,4,5 --persistent-mount /workspace
$PY src/evaluate.py --runtime /workspace/july-native status
$PY src/evaluate.py --runtime /workspace/july-native collect
```

Preflight tests real CUDA computation on every allocated UUID and both regression checkpoints on an isolated GPU. No local CPU training is performed. The isolated native smoke jobs use **512 training examples and one E5 epoch**; their results are excluded from production and the submission CSV. Production commands use historical defaults without these overrides. `start` uses tmux; each worker sees one GPU through `CUDA_VISIBLE_DEVICES` set before imports and uses logical `cuda:0`. Per-job native working directories prevent output collisions. Failed jobs are retained while healthy jobs continue; incomplete score groups receive no final pass verdict. `reconcile` can recover existing finished W&B runs after interruption without rerunning models. Never remove a coordinator lock until its recorded process has been inspected.

Back up before stopping the Pod. From the local computer, substituting the Pod's SSH connection details:

```bash
python src/evaluate.py --runtime LOCAL_BACKUP_DIRECTORY backup --ssh-host root@POD_HOST --ssh-port POD_PORT --identity SSH_PRIVATE_KEY --remote-runtime /workspace/july-native
```

The backup verifies the expected result grid and native summaries. This code never terminates a Pod. Native source, dependencies, logs, caches, checkpoints and JSON exports remain outside the six-file submission layout. `self-test` checks job uniqueness, default commands, raw-score arithmetic, precision, incomplete reporting and coordinator exclusion without fitting a model.

The optional `monitor` command runs locally without fitting models. It snapshots native summaries and logs through SSH, verifies the 100-job grid and full-precision metrics, and refreshes the local CSV. With explicit `--stop-when-complete`, it requests a stop only after all 100 results are complete, backed up and verified, the GPUs have no active compute processes, and the exact task Pod identity matches. It does not terminate/delete Pods. Run `python src/evaluate.py monitor --help` for arguments. The active production used launcher commit `e3c28241b8764c12225b161277b248d2be3ce2fb`; the later supervisor addition was not copied over its executing code. Official MulTaBench remains pinned to the July commit throughout.
