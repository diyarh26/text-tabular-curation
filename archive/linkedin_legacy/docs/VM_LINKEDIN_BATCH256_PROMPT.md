Continue my Technion NLP final project (097215, Track 2). Run a NEW, full LinkedIn salary evaluation on this Azure GPU VM, starting from scratch with physical TAR batch 256. The private repository is `diyarh26/text-tabular-curation`.

The code, data and checks are already published. Use this exact source commit:

`bfe0ba3b846cb8bd719d43a586dc4d25b36088cd`

It is available on `origin/kaggle-fda-batch256-auroc`. Do not substitute main's older launcher or use the Kaggle resume notebook.

Important working preference: keep me updated briefly. Discuss scientific changes and important decisions with me BEFORE making them. Setup, read-only checks, the isolated GPU preflight below, and launching this approved experiment are already authorized. Do not stop just to ask whether to begin. If a check fails, show the actual error/traceback and stop before changing the experiment.

The approved experiment:

- LinkedIn ONLY: `configs/linkedin_salary.yaml`, `data/processed/linkedin_salary.csv`.
- The existing 15,000 rows; target `log_salary`; regression. Do not regenerate, resample, rebalance, clean again or change columns/missing-value rules.
- Text: `title`, `description`. Structured: `experience_level`, `work_type`, `state`, `application_type`, `company_industry`, `job_industry`, `remote_allowed`, `company_size`, `company_employees`, `company_followers`, `n_skills`, `n_benefits`.
- Five learners: TabM, CatBoost, LightGBM, TabPFN-v2, TabPFN-v2.5. Four conditions: structured, text only with frozen E5, joint frozen E5, joint TAR. Six official folds (0 through 5), 10,000 train / 5,000 test per fold: exactly 120 evaluations.
- Actual pinned MulTaBench source: `ff0c9c8b40a2085a06a7771190e53a261341802f`, through our existing `tt_curation` runner. Keep its models, preprocessing, splits, seeds and hyperparameters.
- TAR: physical train AND validation batch **256**, accumulation **1**, full precision, E5-small-v2, 512 tokens, LoRA rank 16 / alpha 32 / dropout 0.1 / last 3 layers, learning rate 0.0001, official-source weight decay **0.001**, maximum 50 epochs and native early stopping patience 3. One tuned E5 per fold, shared by all five learners. No 10-epoch pilot cap and no wall-clock cutoff.
- All 120 evaluations must be newly produced by this experiment. Import NO Kaggle results, checkpoints, adapters, or baselines. Keep previous experiments intact. Once this new run starts, its OWN manifest-verified checkpoints/results may be resumed after an interruption.

1. Inspect repository instructions, OS, `nvidia-smi`, running GPU processes/tmux sessions, RAM and free disk. Historically this VM had one Tesla T4, but verify current hardware and CUDA computation. Require an idle, working NVIDIA GPU. Do not kill another job, resize/rent a machine, or silently share an occupied GPU. Check existing GitHub authentication without exposing credentials; reuse it if healthy. Do not ask for a token unnecessarily.

2. Preserve the existing checkout, uncommitted work and results. Fetch the branch and make a separate detached worktree at the exact source commit. For a healthy existing checkout at `~/text-tabular-curation`, use:

```bash
BASE_REPO="$HOME/text-tabular-curation"
RUN_DIR="$HOME/text-tabular-linkedin-vm256"
SOURCE_COMMIT=bfe0ba3b846cb8bd719d43a586dc4d25b36088cd
git -C "$BASE_REPO" fetch origin kaggle-fda-batch256-auroc
git -C "$BASE_REPO" worktree add --detach "$RUN_DIR" "$SOURCE_COMMIT"
cd "$RUN_DIR"
git rev-parse HEAD
```

If the base checkout is absent, clone the private repository using existing authentication first. If the proposed run directory exists, inspect it; do not overwrite/delete it. Use a new empty sibling directory if needed. Keep the experiment worktree at the pinned commit while running and resuming.

3. Read `README.md`, `scripts/run_full_linkedin_vm.py`, `scripts/run_linkedin_vm256.sh`, and `scripts/setup_gpu_vm.sh`. Create this worktree's OWN `.venv` so an editable installation cannot point to the old checkout:

```bash
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
bash scripts/setup_gpu_vm.sh
uv pip check --python .venv/bin/python
.venv/bin/python -c "import torch, tt_curation; print('Project:', tt_curation.REPO_ROOT); print('Torch:', torch.__version__); assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0)); print((torch.ones(2, device='cuda') * 2).cpu())"
.venv/bin/python -m pytest tests/test_linkedin_vm.py tests/test_full_linkedin.py -q
```

Verify the imported project is THIS worktree. The setup installs the pinned benchmark dependencies. Never print `.env` or tokens. If model access actually requires an existing secret, handle it securely; do not assume new authentication is required. Local validation passed 37 tests on CPU; actual GPU execution must still pass here.

4. Confirm `results/linkedin_vm_batch256/` has no old experiment outputs. Start the prepared shell launcher in a NEW tmux session, with persistent logging and pipeline failure propagation:

```bash
mkdir -p logs
tmux new-session -d -s linkedin_vm256 -c "$RUN_DIR" \
  "bash -o pipefail -c 'bash scripts/run_linkedin_vm256.sh 2>&1 | tee -a logs/linkedin_vm256.log'"
```

If that tmux session already exists, inspect it before launching anything else. The shell script verifies the CSV/config checksums, official split hashes, exact upstream defaults, source/package manifest, and empty experiment initialization. It then checks all five learner imports, both TabPFN REGRESSION checkpoints through CUDA predictions, and **two real TAR optimizer updates plus validation on fold-0 LinkedIn training examples at physical batch 256**. This is an isolated memory/access smoke test: its adapter/checkpoints are discarded and never used by production. The production 50-epoch schedule is unchanged. After preflight succeeds, the shell script starts the full six-fold run automatically, then audits and reports it. No GPU-memory fallback is implemented.

5. Watch initial execution long enough to verify preflight passes and full evaluation starts. If physical 256 runs out of memory, stop, preserve logs and saved work, and show me the traceback. Do not switch precision, shorten sequences, reduce batch, add accumulation, skip a learner/fold, or alter hyperparameters without discussing it with me.

6. The full-run log prints a result table immediately after EACH completed fold and a cumulative table using ONLY fully completed folds. Report R2, both deltas, the two independent criterion counts, TAR epochs/time, fold/per-run timings and sampled GPU memory. After fold 0, use actual timing to estimate the remaining time. These are preliminary until all six folds finish. No performance-based early exit.

The acceptance calculation uses raw, unrounded six-fold **R2** means:

- Joint Signal delta = joint frozen minus the better of structured/text-only means.
- TAR Gain delta = joint TAR minus joint frozen.
- Each criterion needs a strictly positive delta in at least 3/5 learners, counted independently; they need not be the same three.

Native RMSE/negative-RMSE diagnostics stay in the original JSONs. Do not use those as the primary course verdict. Do not claim the run reproduces the paper's five-seed design: this is our agreed six-fold course evaluation using official-source training defaults.

7. Main output: `results/linkedin_vm_batch256/linkedin_salary/`. Main log: `logs/linkedin_vm256.log`. Tmux: `linkedin_vm256`. The manifest prevents mixing different code/packages/data/settings. A normal interruption can be resumed with the SAME shell command at the SAME source revision. Never delete results just to bypass an audit failure; inspect it and discuss any necessary repair first.

8. At completion, confirm both commands succeed and show me `summary.txt` and the full six-fold R2/delta table:

```bash
.venv/bin/python -u scripts/run_full_linkedin_vm.py --audit_only
.venv/bin/python -u scripts/run_full_linkedin_vm.py --report_only
```

Only after all 120 records pass the audit, publish the small result JSONs, manifest, preflight, timing/memory records and reports in a separate results branch/worktree and push it. Preserve the pinned experiment worktree so its resume/audit command keeps working. Do not commit `.venv`, `.env`, secrets, HF model weights, or anything under `checkpoints/`. Do not overwrite older LinkedIn/Kaggle results or merge into main. Tell me the result branch and commit.

At the end of your current turn, state the actual status, GPU, source commit, preflight result, number of completed evaluations, tmux session, log and output paths. Do not describe the experiment as finished until it actually is. I will ask for progress as it runs.
