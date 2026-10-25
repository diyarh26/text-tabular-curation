"""Fresh LinkedIn VM evaluation: physical batch 256, six folds, live R2 reports.

Prepare an EMPTY output with --prepare_only, test CUDA with --gpu_smoke, then run
without a mode flag. Resume only this experiment by repeating the same command.
No Kaggle results/adapters are imported. No automatic memory fallback or time cap.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
from importlib.metadata import distribution, version
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

from colab_tar_compare import write_json
from run_full_linkedin import CONDITIONS, DATA_SHA256_LF, MODELS, fold_dir, full_tar, text_digest

PROTOCOL = "linkedin_vm_physical256_v1"
CONFIG_SHA256_LF = "859863d245ed81b761d27682bf4b1657e5687b8a60f87d22dbe26e87c8cbc2ce"
TAR = dict(learning_rate=0.0001, epochs=50, lora_rank=16, layers=3,
           patience=3, weight_decay=0.001, batch_size=256)
TRAINING = dict(per_device_train_batch_size=256, per_device_eval_batch_size=256,
                gradient_accumulation_steps=1, num_train_epochs=50, learning_rate=0.0001,
                weight_decay=0.001, fp16=False, bf16=False, max_steps=-1,
                load_best_model_at_end=True, restore_callback_states_from_checkpoint=True)
PACKAGES = ["multabench", "torch", "transformers", "accelerate", "peft", "catboost",
            "lightgbm", "pytabkit", "tabpfn", "scikit-learn", "pandas", "numpy", "skrub"]


def verified_data():
    from tt_curation import REPO_ROOT
    from tt_curation.data import load_dataset
    from tt_curation.spec import load_spec
    config = REPO_ROOT / "configs/linkedin_salary.yaml"
    spec = load_spec(config)
    if text_digest(config) != CONFIG_SHA256_LF or text_digest(spec.csv) != DATA_SHA256_LF:
        raise ValueError("Prepared LinkedIn config/CSV changed. Do not resample or change features.")
    data = load_dataset(spec)
    if (spec.name != "linkedin_salary" or spec.task != "regression" or len(data.y) != 15_000
            or data.text_columns != ["title", "description"] or len(data.structured_columns) != 12):
        raise ValueError("Expected the agreed 15K salary regression sample, 2 text + 12 structured inputs")
    return data, config


def official_splits(data):
    import numpy as np
    from multabench.benchmark.splits import get_split
    result = {}
    for fold in range(6):
        train, test = get_split(data.y.to_numpy(), is_cls=False, split=fold, size="10k")
        if (len(train) != 10_000 or len(test) != 5_000 or set(train) & set(test)
                or set(train) | set(test) != set(range(15_000))):
            raise ValueError(f"Unexpected official regression split {fold}")
        result[str(fold)] = dict(n_train=len(train), n_test=len(test),
            train_indices_sha256=hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest(),
            test_indices_sha256=hashlib.sha256(np.asarray(test, dtype="<i8").tobytes()).hexdigest())
    return result


def experiment_settings(data, config):
    from tt_curation import MULTABENCH_COMMIT, REPO_ROOT
    from tt_curation.conditions import Condition, MODELS as names
    from tt_curation.runner import _git_commit
    from multabench.e5.constants import E5_LORA
    direct = distribution("multabench").read_text("direct_url.json")
    if not direct or json.loads(direct).get("vcs_info", {}).get("commit_id") != MULTABENCH_COMMIT:
        raise ValueError("Installed MulTaBench must be installed from the exact pinned Git commit")
    if asdict(E5_LORA) != TAR or set(MODELS) != set(names) or CONDITIONS != [str(c) for c in Condition]:
        raise ValueError("Official TAR defaults or evaluation grid differ from the approved protocol")
    revision = _git_commit()
    if not revision:
        raise ValueError("Run from a Git checkout so source provenance can be recorded")
    paths = [*sorted((REPO_ROOT / "src/tt_curation").glob("*.py")),
             *(REPO_ROOT / "scripts" / p for p in
               ("run_full_linkedin_vm.py", "run_full_linkedin.py", "colab_tar_compare.py", "run_linkedin_vm256.sh")),
             REPO_ROOT / "requirements/constraints.txt", REPO_ROOT / "pyproject.toml"]
    return dict(protocol=PROTOCOL, dataset="linkedin_salary", task="regression", rows=15_000,
        csv_sha256_lf=DATA_SHA256_LF, config_sha256_lf=text_digest(config),
        text_columns=data.text_columns, structured_columns=data.structured_columns,
        multabench_commit=MULTABENCH_COMMIT, source_revision=revision,
        models=MODELS, conditions=CONDITIONS, folds=list(range(6)), expected_evaluations=120,
        size="10k", frozen_encoder="e5-small", curation_metric="r2",
        native_diagnostic_metric="root_mean_squared_error", tar_parameters=TAR,
        training_arguments=TRAINING, precision="float32", max_tokens=512,
        accumulation_loss="reference_single_batch_mean", restore_early_stopping_state=True,
        course_rule=dict(round_scores=False, delta_strictly_greater_than=0,
                         independent_counts=True, min_learners=3),
        splits=official_splits(data),
        code={p.relative_to(REPO_ROOT).as_posix(): text_digest(p) for p in paths},
        packages={p: version(p) for p in PACKAGES})


def initialize_manifest(output, settings, *, prepare=False):
    path = Path(output) / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != settings:
            raise ValueError("Manifest differs. Do not mix revisions, packages, batches, results or checkpoints.")
    elif prepare and (not Path(output).exists() or not any(Path(output).iterdir())):
        write_json(path, settings)
    else:
        raise ValueError("Initialize an EMPTY new output with --prepare_only. Do not import old results.")


def expected_paths(output, folds=range(6)):
    for fold in folds:
        for model in MODELS:
            for condition in CONDITIONS:
                yield fold, model, condition, fold_dir(output, fold) / "runs" / f"{model}_{condition}_10k_fold{fold}.json"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def audit_runs(output, folds=range(6), *, allow_missing=False):
    from tt_curation import MULTABENCH_COMMIT
    from tt_curation.conditions import MODELS as names
    output, folds = Path(output), list(folds)
    problems = []
    manifest = output / "manifest.json"
    source = json.loads(manifest.read_text(encoding="utf-8"))["source_revision"] if manifest.exists() else None
    expected = list(expected_paths(output, folds))
    for fold, model, condition, path in expected:
        label = str(path.relative_to(output))
        if not path.exists():
            if not allow_missing:
                problems.append(f"{label}: missing")
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status") != "ok":
                problems.append(f"{label}: {row.get('error', row.get('status'))}\n{row.get('traceback', '')}")
                continue
            encoder = "e5-small-tar" if condition == "joint_tar" else None if condition == "structured" else "e5-small"
            identity = dict(protocol=PROTOCOL, dataset="linkedin_salary", model=names[model], short_name=model,
                condition=condition, fold=fold, size="10k", task="regression", n_train=10_000, n_test=5_000,
                n_features=12 if condition == "structured" else 2 if condition == "unstructured" else 14,
                text_encoder=encoder, multabench_commit=MULTABENCH_COMMIT, git=source,
                metric="root_mean_squared_error")
            valid = all(row.get(k) == v for k, v in identity.items())
            valid &= all(finite(row.get(k)) for k in ("r2", "test_error", "test_score", "runtime_s"))
            if valid:
                valid &= (row["r2"] <= 1 + 1e-8 and row["test_error"] >= 0 and row["runtime_s"] >= 0
                          and math.isclose(row["test_score"], -row["test_error"], abs_tol=1e-8)
                          and str(row.get("device", "")).startswith("cuda"))
            if condition == "joint_tar":
                stats = row.get("tar_finetune") or {}
                valid &= (row.get("tar_parameters") == TAR and row.get("training_arguments") == TRAINING
                          and stats.get("rows") == 10_000 and finite(stats.get("epochs"))
                          and 0 < stats["epochs"] <= 50 and finite(stats.get("best_val_loss"))
                          and finite(stats.get("finetune_seconds")) and stats["finetune_seconds"] >= 0)
            if not valid:
                problems.append(f"{label}: unexpected provenance, feature counts, TAR settings or regression metrics")
        except (ValueError, OSError, TypeError, AttributeError) as exc:
            problems.append(f"{label}: unreadable ({type(exc).__name__}: {exc})")
    allowed = {p for *_, p in expected_paths(output)}
    problems.extend(f"Unexpected run file: {p.relative_to(output)}"
                    for p in sorted(output.glob("folds/*/runs/*.json")) if p not in allowed)
    return problems


def collect_runs(output, folds):
    import pandas as pd
    return pd.DataFrame([json.loads(p.read_text(encoding="utf-8"))
                         for *_, p in expected_paths(output, folds)])


def write_reports(output, *, just_finished=None):
    """Use fully audited folds only; always compute the course verdict from R2."""
    from tt_curation.criteria import curation_verdict, evaluation_report
    output = Path(output)
    complete = [f for f in range(6) if not audit_runs(output, [f])]
    problems = audit_runs(output)
    if not complete:
        (output / "curation_verdict.csv").unlink(missing_ok=True)
        (output / "summary.txt").write_text("INCOMPLETE: no fully audited fold. No final verdict.\n", encoding="utf-8")
        return False
    runs = collect_runs(output, complete)
    lines = [f"{'COMPLETE' if not problems else 'PRELIMINARY'}: {len(complete)*20}/120 evaluations; folds {complete}.",
             "Primary metric: R2 (higher is better). Raw RMSE/negative RMSE remain in run JSONs.",
             "Joint delta = joint frozen - better unimodal mean; TAR delta = joint TAR - joint frozen."]

    def table(selected):
        # criteria.py expects higher-is-better test_score; map a COPY to R2, preserving the raw records.
        verdict = curation_verdict(selected.assign(test_score=selected.r2))
        columns = {"score_structured": "Structured", "score_unstructured": "Text only",
                   "score_joint_frozen": "Joint frozen", "score_joint_tar": "Joint TAR",
                   "delta_frozen_vs_best_unimodal": "Joint delta", "delta_tar_vs_frozen": "TAR delta"}
        display = verdict.set_index("model")[list(columns)].rename(columns=columns)
        joint, tar = int(verdict.joint_signal.sum()), int(verdict.tar_gain.sum())
        text = display.to_string(float_format=lambda x: f"{x:.6f}")
        text += f"\nJoint Signal: {joint}/5 | TAR Gain: {tar}/5; need >=3/5 independently on six-fold means."
        return verdict, text

    if just_finished is not None:
        verdict, text = table(runs[runs.fold == just_finished])
        folder = fold_dir(output, just_finished)
        verdict.to_csv(folder / "r2_verdict_preliminary.csv", index=False)
        (folder / "r2_summary.txt").write_text(text + "\nPRELIMINARY: one fold only.\n", encoding="utf-8")
        print(f"\n===== FOLD {just_finished} COMPLETE: ALL 20 EVALUATIONS AUDITED =====\n{text}", flush=True)
        stats = runs[(runs.fold == just_finished) & (runs.condition == "joint_tar")].iloc[0].tar_finetune
        print("Shared TAR training: " + json.dumps(stats), flush=True)
        timing = folder / "timing.json"
        if timing.exists():
            print("Fold timing/memory: " + timing.read_text(encoding="utf-8"), flush=True)
        runs[runs.fold == just_finished][["model", "condition", "runtime_s"]].to_csv(folder / "run_times.csv", index=False)
    verdict, text = table(runs)
    evaluation_report(runs).to_csv(output / "evaluation_report.csv", index=False)
    verdict.to_csv(output / "r2_cumulative_preliminary.csv", index=False)
    lines += [text]
    if problems:
        (output / "curation_verdict.csv").unlink(missing_ok=True)
        lines += ["No final verdict until all 120 expected evaluations pass the audit."]
    else:
        verdict.to_csv(output / "curation_verdict.csv", index=False)
        passed = verdict.joint_signal.sum() >= 3 and verdict.tar_gain.sum() >= 3
        lines += [f"Final six-fold course verdict: {'PASS' if passed else 'FAIL'}."]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n===== CUMULATIVE RESULTS: FULLY COMPLETED FOLDS ONLY =====\n" + "\n".join(lines), flush=True)
    return not problems


@contextlib.contextmanager
def guarded_tar(cache, *, smoke_steps=None):
    """Native single-batch loss; only checkpoint persistence/validation surrounds upstream TAR."""
    from multabench.finetune import lora
    from transformers import TrainerCallback
    with full_tar(cache, micro_batch=256):
        original = lora.Trainer

        class SmokeStop(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= smoke_steps:
                    control.should_training_stop = True
                return control

        class GuardedTrainer(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                actual = {k: getattr(self.args, k) for k in TRAINING}
                if actual != TRAINING or self.compute_loss_func is not None:
                    raise ValueError(f"Refusing changed native batch/loss/precision/training settings: {actual}")
                if self.args.eval_strategy.value != "epoch" or self.args.save_strategy.value != "epoch":
                    raise ValueError("Official epoch validation/checkpoint schedule changed")
                callbacks = self.callback_handler.callbacks
                early = [c for c in callbacks if c.__class__.__name__ == "EarlyStoppingCallback"]
                if len(early) != 1 or early[0].early_stopping_patience != 3:
                    raise ValueError("Official early stopping patience must remain 3")
                if self.train_dataset[0]["input_ids"].numel() != 512:
                    raise ValueError("Expected the upstream 512-token padded training examples")
                write_json(Path(cache) / "actual_training_arguments.json", actual)
                if smoke_steps is not None:
                    self.add_callback(SmokeStop())

        lora.Trainer = GuardedTrainer
        try:
            yield
        finally:
            lora.Trainer = original


@contextlib.contextmanager
def fail_fast(destination):
    """Persist the runner's failure JSON and traceback, then stop before another evaluation starts."""
    from tt_curation import runner
    original = runner.run_one

    def checked(*args, **kwargs):
        row = original(*args, **kwargs)
        row["protocol"] = PROTOCOL
        if row["condition"] == "joint_tar":
            row.update(tar_parameters=TAR, training_arguments=TRAINING)
        if row["status"] != "ok":
            path = Path(destination) / "runs" / f"{row['short_name']}_{row['condition']}_10k_fold{row['fold']}.json"
            write_json(path, row)
            raise RuntimeError("Evaluation failed; saved work preserved. No memory fallback.\n"
                               + row.get("traceback", row.get("error", "Unknown error")))
        return row

    runner.run_one = checked
    try:
        yield
    finally:
        runner.run_one = original


class GPUMonitor:
    """Sample nvidia-smi every two seconds; label this sampled device usage, not an exact peak."""
    def __init__(self, path):
        self.path, self.peak, self.error = Path(path), 0, None
        self.stop = threading.Event()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        def sample():
            device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            with self.path.open("a", encoding="utf-8") as stream:
                while not self.stop.is_set():
                    try:
                        result = subprocess.run(["nvidia-smi", "-i", device,
                            "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
                        stream.write(result.stdout)
                        stream.flush()
                        self.peak = max(self.peak, int(result.stdout.strip().split(",")[-2]))
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        self.error = str(exc)
                    self.stop.wait(2)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=12)


def require_gpu():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA required, with exactly ONE visible GPU. Set CUDA_VISIBLE_DEVICES=0 before Python.")
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name(0)}; torch {torch.__version__}; physical batch 256, accumulation 1.", flush=True)


def gpu_smoke(data, output):
    """Two REAL optimizer updates + validation; isolated temporary adapter is never used by production."""
    import gc
    import numpy as np
    import torch
    from multabench.benchmark.load import split_dataset
    from multabench.datasets.objects import MultimodalDataset, SupervisedTask
    from tt_curation.runner import _model_class, shared_tar_encoder
    require_gpu()
    for model in MODELS:
        _model_class(model)
    x = np.random.default_rng(42).normal(size=(32, 4)).astype(np.float32)
    y = x[:, 0] + x[:, 1] * 0.2
    for model in ("tabpfnv2", "tabpfnv2p5"):
        wrapper = _model_class(model)(problem_type=SupervisedTask.REGRESSION, device=torch.device("cuda"),
                                     dataset=None, text_encoder=None, fold=0, size="10k")
        probe = wrapper.get_model_instance(cat_feature=[])
        probe.set_params(device="cuda")
        probe.fit(x, y)
        assert np.isfinite(probe.predict(x[:3])).all()
        del probe, wrapper
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{model}: regression checkpoint and CUDA prediction verified", flush=True)
    dataset = MultimodalDataset(x=data.x, y=data.y, task_type=SupervisedTask.REGRESSION, dataset_id=None)
    split = split_dataset(dataset, fold=0, size="10k")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="linkedin256_smoke_") as scratch:
        with guarded_tar(Path(scratch), smoke_steps=2):
            with shared_tar_encoder(split, data.text_columns, False, torch.device("cuda"), enabled=True) as stats:
                record = dict(protocol=PROTOCOL, passed=True, smoke_only_optimizer_updates=2,
                    production_epoch_cap_unchanged=50, training_arguments=TRAINING, tar_stats=stats,
                    seconds=time.monotonic()-started,
                    peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                    gpu=torch.cuda.get_device_name(0))
    write_json(output / "gpu_preflight.json", record)
    print("GPU PREFLIGHT PASSED. Two-update smoke output discarded; production still starts from scratch.\n"
          + json.dumps(record, indent=2), flush=True)


def main():
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results/linkedin_vm_batch256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare_only", action="store_true")
    mode.add_argument("--gpu_smoke", action="store_true")
    mode.add_argument("--report_only", action="store_true")
    mode.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()
    from dotenv import load_dotenv
    from tt_curation import REPO_ROOT
    load_dotenv(REPO_ROOT / ".env")
    output = Path(args.results_root).resolve() / "linkedin_salary"
    data, config = verified_data()
    settings = experiment_settings(data, config)
    initialize_manifest(output, settings, prepare=args.prepare_only)
    if args.prepare_only:
        from tt_curation.data import describe
        print("\n".join(describe(data)), flush=True)
        print("Manifest, official six splits and physical-256 settings verified. No training started.", flush=True)
        return
    if args.audit_only:
        problems = audit_runs(output)
        if problems:
            raise SystemExit("\n".join(problems))
        print("PASS: all 120 fresh LinkedIn physical-256 evaluations audited.")
        return
    if args.report_only:
        if not write_reports(output):
            raise SystemExit("INCOMPLETE: no final verdict")
        return
    require_gpu()
    if args.gpu_smoke:
        gpu_smoke(data, output)
        return
    preflight = output / "gpu_preflight.json"
    if not preflight.exists() or not json.loads(preflight.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("Run --gpu_smoke first. Physical batch 256 must be tested on this VM.")
    problems = audit_runs(output, allow_missing=True)
    if problems:
        raise RuntimeError("Existing files need inspection before resuming; no files were removed.\n" + "\n".join(problems))
    import torch
    from tt_curation.conditions import Condition
    from tt_curation.runner import run_all
    for fold in range(6):
        if not audit_runs(output, [fold]):
            print(f"Fold {fold}: already complete in THIS experiment; audited and preserved.", flush=True)
            continue
        destination = fold_dir(output, fold)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        succeeded = False
        with GPUMonitor(destination / "gpu_memory.csv") as monitor:
            try:
                with guarded_tar(destination / "checkpoints"), fail_fast(destination):
                    run_all(data, destination, models=MODELS, conditions=list(Condition), folds=[fold],
                            size="10k", device="cuda", frozen_encoder="e5-small")
                problems = audit_runs(output, [fold])
                if problems:
                    raise RuntimeError("\n".join(problems))
                succeeded = True
            finally:
                timing_path = destination / "timing.json"
                attempts = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else []
                attempts.append(dict(seconds=time.monotonic()-started, completed=succeeded,
                    peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                    sampled_device_peak_mib=monitor.peak, sampling_error=monitor.error,
                    note="This session attempt; a resumed attempt can reuse saved work."))
                write_json(timing_path, attempts)
        write_reports(output, just_finished=fold)
    if not write_reports(output):
        raise RuntimeError("Full evaluation did not pass the 120-record audit")


if __name__ == "__main__":
    main()
