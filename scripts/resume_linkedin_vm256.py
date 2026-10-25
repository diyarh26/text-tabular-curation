"""Resume the original VM experiment unchanged; executed OUTSIDE its pinned source checkout."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import subprocess
import sys
import time
import zipfile

SOURCE = "bfe0ba3b846cb8bd719d43a586dc4d25b36088cd"
PROTOCOL = "linkedin_vm_physical256_v1"
RUN_NAME = "linkedin_vm256_resume"
MODELS = {"tabm": "TabM", "cat": "CatBoost", "light": "LightGBM",
          "tabpfnv2": "TabPFN-v2", "tabpfnv2p5": "TabPFN-v2p5"}
CONDITIONS = ["structured", "unstructured", "joint_frozen", "joint_tar"]
GRID = {(m, c, f) for m in MODELS for c in CONDITIONS for f in range(6)}
MINIMUM = {(m, c, f) for m, c, f in GRID if f < 4 or (f == 4 and c != "joint_tar")}
CHECKPOINT_FILES = ["model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                    "training_args.bin", "trainer_state.json"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def restore_input(source, root):
    """Require the audited VM-specific bundle. Never overwrite working output or start fresh."""
    import shutil
    source, root = Path(source), Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to overwrite working output")
    root.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("Unexpected input symlink")
        if not (source / "transfer_index.json").is_file():
            raise ValueError("Select the prepared VM resume bundle root")
        shutil.copytree(source, root, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(source) as archive:
            names = set()
            for info in archive.infolist():
                p = PurePosixPath(info.filename)
                if (info.filename in names or p.is_absolute() or ".." in p.parts or "\\" in info.filename
                        or stat.S_ISLNK(info.external_attr >> 16)
                        or not (root.parent / info.filename).resolve().is_relative_to(root)):
                    raise ValueError("Unsafe or wrong experiment ZIP member: " + info.filename)
                names.add(info.filename)
            if root.name + "/transfer_index.json" not in names:
                raise ValueError("This is not the prepared VM checkpoint bundle")
            archive.extractall(root.parent)
    index = json.loads((root / "transfer_index.json").read_text())
    verified = root / "transfer_metadata/input_verified.json"
    if verified.exists():
        if json.loads(verified.read_text()).get("index_sha256") != sha(root / "transfer_index.json"):
            raise ValueError("Input verification record/index differs")
    else:
        for name, digest in index["initial_checkpoint_sha256"].items():
            path = root / name
            if not path.resolve().is_relative_to(root) or sha(path) != digest:
                raise ValueError("Transferred checkpoint file differs: " + name)
        write_json(verified, dict(index_sha256=sha(root / "transfer_index.json"), checkpoint_integrity_verified=True))
    validate_resume(root)


def run_path(output, identity):
    model, condition, fold = identity
    return output / "folds" / f"fold{fold}" / "runs" / f"{model}_{condition}_10k_fold{fold}.json"


def validate_resume(root):
    root = Path(root)
    output = root / "linkedin_salary"
    index = json.loads((root / "transfer_index.json").read_text())
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (index.get("source_revision") != SOURCE or manifest.get("source_revision") != SOURCE
            or manifest.get("protocol") != PROTOCOL or manifest.get("dataset") != "linkedin_salary"
            or (root / "source_revision.txt").read_text().strip() != SOURCE
            or sha(manifest_path) != index["manifest_sha256"]):
        raise ValueError("Original VM source/protocol/manifest differs; no migration or fresh fallback")
    args = manifest["training_arguments"]
    for key, expected in dict(per_device_train_batch_size=256, per_device_eval_batch_size=256,
            gradient_accumulation_steps=1, num_train_epochs=50, learning_rate=.0001,
            weight_decay=.001, fp16=False, bf16=False, restore_callback_states_from_checkpoint=True).items():
        if args.get(key) != expected:
            raise ValueError("Original training settings differ: " + key)
    if len(index["original_run_sha256"]) != 95:
        raise ValueError("The audited 95-result inventory is required")
    for name, digest in index["original_run_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or sha(path) != digest:
            raise ValueError("A previously completed evaluation changed or is missing: " + name)
    actual = set()
    for path in output.glob("folds/fold*/runs/*.json"):
        row = json.loads(path.read_text())
        identity = (row.get("short_name"), row.get("condition"), row.get("fold"))
        if identity not in GRID or identity in actual or path != run_path(output, identity):
            raise ValueError("Unexpected/duplicate result identity: " + str(path))
        actual.add(identity)
        model, condition, fold = identity
        expected = dict(status="ok", dataset="linkedin_salary", model=MODELS[model], task="regression",
            size="10k", protocol=PROTOCOL, git=SOURCE, multabench_commit=manifest["multabench_commit"],
            n_train=10000, n_test=5000, n_features=12 if condition == "structured" else 2 if condition == "unstructured" else 14,
            text_encoder=None if condition == "structured" else "e5-small-tar" if condition == "joint_tar" else "e5-small")
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("Saved evaluation failed identity/provenance/status validation: " + str(path))
        if (not all(isinstance(row.get(k), (float, int)) and not isinstance(row[k], bool) and math.isfinite(row[k])
                    for k in ("test_score", "test_error", "r2")) or row["r2"] > 1 + 1e-8
                or row["test_error"] < 0 or not math.isclose(row["test_score"], -row["test_error"], abs_tol=1e-8)):
            raise ValueError("Invalid regression metrics: " + str(path))
        if condition == "joint_tar" and (row.get("tar_parameters") != manifest["tar_parameters"]
                                        or row.get("training_arguments") != args):
            raise ValueError("Saved TAR parameters differ: " + str(path))
    if not MINIMUM <= actual:
        raise ValueError("Previously completed VM evaluations are missing")
    for fold in (4, 5):
        cache = output / f"folds/fold{fold}/checkpoints"
        for directory in (cache / "trainer").glob("checkpoint-*"):
            if directory.is_dir() and any(not (directory / name).is_file() or (directory / name).stat().st_size == 0
                                         for name in CHECKPOINT_FILES):
                raise ValueError("Incomplete checkpoint directory; preserve and inspect it: " + str(directory))
        if (cache / "complete.json").exists():
            required = ["adapter/adapter_model.safetensors", "adapter/adapter_config.json", "tokenizer/tokenizer_config.json"]
            if any(not (cache / name).is_file() for name in required):
                raise ValueError("Completed TAR adapter/tokenizer is incomplete")
    if any((m, "joint_tar", 4) not in actual for m in MODELS):
        cache = output / "folds/fold4/checkpoints"
        if not (cache / "complete.json").exists():
            checkpoints = sorted((cache / "trainer").glob("checkpoint-*/trainer_state.json"),
                                 key=lambda p: int(p.parent.name.split("-")[-1]))
            if not checkpoints:
                raise ValueError("Fold 4 checkpoint required; refusing to restart its TAR from scratch")
            latest = checkpoints[-1]
            state = json.loads(latest.read_text())
            if (state.get("global_step", 0) < 71 or not 1 <= state.get("epoch", 0) <= 50
                    or latest.parent.name != "checkpoint-" + str(state.get("global_step"))):
                raise ValueError("Fold 4 checkpoint predates the transferred step 71")
            for name in CHECKPOINT_FILES:
                p = latest.parent / name
                if not p.is_file() or p.stat().st_size == 0:
                    raise ValueError("Incomplete fold 4 checkpoint: " + str(p))
    return dict(saved_evaluations=len(actual), remaining=120-len(actual), original_results_preserved=95,
                source_revision=SOURCE, physical_batch=256, accumulation=1)


def relocate_checkpoint_paths(root):
    """Rewrite only best-checkpoint LOCATION in the working copy; keep all training state intact."""
    root = Path(root).resolve()
    changed = []
    for state_path in root.glob("linkedin_salary/folds/fold*/checkpoints/trainer/checkpoint-*/trainer_state.json"):
        state = json.loads(state_path.read_text())
        best = state.get("best_model_checkpoint")
        if not best:
            continue
        name = PurePosixPath(best.replace("\\", "/")).name
        if not name.startswith("checkpoint-") or not name[len("checkpoint-"):].isdigit():
            raise ValueError("Unexpected best checkpoint path")
        destination = state_path.parent.parent / name
        if not (destination / "model.safetensors").is_file():
            raise ValueError("The saved best checkpoint is missing: " + best)
        if str(destination) == best:
            continue
        backup = root / "transfer_metadata/original_trainer_states" / state_path.relative_to(root)
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(state_path.read_bytes())
        state["best_model_checkpoint"] = str(destination)
        write_json(state_path, state)
        changed.append(str(state_path.relative_to(root)))
    return changed


def original_runtime(project):
    project = Path(project).resolve()
    sys.path.insert(0, str(project / "scripts"))
    spec = importlib.util.spec_from_file_location("original_vm256", project / "scripts/run_full_linkedin_vm.py")
    vm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vm)
    from tt_curation import REPO_ROOT
    if REPO_ROOT.resolve() != project:
        raise ValueError("Editable installation points to another project")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip() != SOURCE:
        raise ValueError("Original immutable Git checkout required")
    return vm


def verify_environment(root, project):
    validate_resume(root)
    vm = original_runtime(project)
    data, config = vm.verified_data()
    # Includes exact original package versions, all 13 code hashes, data and six splits.
    vm.initialize_manifest(Path(root) / "linkedin_salary", vm.experiment_settings(data, config))
    return vm, data


def worker(root, project, fold, phase):
    import torch
    vm, data = verify_environment(root, project)
    vm.require_gpu()
    output = Path(root) / "linkedin_salary"
    destination = output / "folds" / f"fold{fold}"
    from tt_curation.conditions import Condition
    from tt_curation.runner import run_all
    selected = [Condition.JOINT_TAR] if phase == "tar" else list(Condition)[:3]
    expected = [(model, str(condition), fold) for model in MODELS for condition in selected]
    if all(run_path(output, identity).exists() for identity in expected):
        print(f"Fold {fold} {phase}: already complete; preserved.", flush=True)
        return
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    succeeded = False
    with vm.GPUMonitor(destination / f"gpu_memory_{phase}_kaggle.csv") as monitor:
        try:
            with vm.guarded_tar(destination / "checkpoints"), vm.fail_fast(destination):
                # Fresh process for every phase. No baseline estimator or probe is fitted in the TAR process.
                run_all(data, destination, models=list(MODELS), conditions=selected, folds=[fold],
                        size="10k", device="cuda", frozen_encoder="e5-small")
            validate_resume(root)
            if not all(run_path(output, identity).exists() for identity in expected):
                raise RuntimeError("Phase did not produce all required evaluations")
            succeeded = True
        finally:
            record = dict(phase=phase, seconds=time.monotonic()-started, completed=succeeded,
                peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                sampled_device_peak_mib=monitor.peak, sampling_error=monitor.error,
                note="This Kaggle phase attempt; may include reused runs/checkpoints.")
            timing_path = destination / "timing.json"
            records = json.loads(timing_path.read_text()) if timing_path.exists() else []
            write_json(timing_path, records + [record])


def export_archives(root, *, live=False):
    root = Path(root).resolve()
    paths = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    if live:
        ready = set()
        for p in root.glob("**/trainer/checkpoint-*/trainer_state.json"):
            try:
                if p.parent.name == "checkpoint-" + str(json.loads(p.read_text())["global_step"]):
                    ready.add(p.parent)
            except (OSError, ValueError, KeyError):
                pass
        paths = [p for p in paths if p.suffix not in (".tmp", ".log") and
                 all(q in ready for q in p.parents if q.name.startswith("checkpoint-") and q.parent.name == "trainer")]
    for suffix, include_weights in [("results", False), ("checkpoint", True)]:
        selected = [p for p in paths if include_weights or "checkpoints" not in p.relative_to(root).parts]
        destination = root.parent / f"{RUN_NAME}_{suffix}.zip"
        temporary = destination.with_suffix(".zip.tmp")
        try:
            before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in selected} if live else {}
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for p in selected:
                    archive.write(p, (Path(RUN_NAME) / p.relative_to(root)).as_posix())
            if live and any((p.stat().st_size, p.stat().st_mtime_ns) != before[p] for p in selected):
                raise OSError("Checkpoint changed during autosave")
        except OSError:
            temporary.unlink(missing_ok=True)
            if not live:
                raise
            print("Snapshot changed; retained the previous ZIP.", flush=True)
            return False
        temporary.replace(destination)
        print(f"Saved {destination.name} ({destination.stat().st_size/2**20:.1f} MiB)", flush=True)
    return True


def report_completed(vm, output, announced):
    for fold in range(6):
        if fold not in announced and not vm.audit_runs(output, [fold]):
            vm.write_reports(output, just_finished=fold)
            announced.add(fold)


def stop_workers(processes):
    for p in processes:
        if p.poll() is None:
            try: os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError: pass
    deadline = time.monotonic()+40
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(1)
    for p in processes:
        if p.poll() is None:
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        p.wait()


def supervise(root, project, gpus=(0, 1)):
    vm, _ = verify_environment(root, project)
    root = Path(root).resolve()
    output = root / "linkedin_salary"
    vm.write_reports(output)
    announced = {f for f in range(6) if not vm.audit_runs(output, [f])}
    if len(announced) == 6:
        export_archives(root)
        return
    active, processes, handles = {}, [], []
    jobs = []
    for fold in (4, 5):
        if fold in announced: continue
        baselines = [(m, c, fold) for m in MODELS for c in CONDITIONS[:3]]
        jobs.append((fold, "tar" if all(run_path(output, i).exists() for i in baselines) else "baselines"))
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    started = time.monotonic()
    last_update, last_export = -60, started
    complete = False
    failed = None
    try:
        while jobs or active:
            for gpu, (p, handle, fold, phase) in list(active.items()):
                code = p.poll()
                if code is None: continue
                handle.close()
                del active[gpu]
                if code:
                    failed = f"GPU {gpu}, fold {fold}, {phase}: exit {code}. No settings were changed."
                    break
                if phase == "baselines": jobs.append((fold, "tar"))
            if failed: break
            report_completed(vm, output, announced)
            for gpu in gpus:
                if gpu in active or not jobs: continue
                fold, phase = jobs.pop(0)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
                handle = (logs / f"fold{fold}_{phase}_gpu{gpu}.log").open("a", encoding="utf-8")
                handles.append(handle)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "worker",
                           "--root", str(root), "--project", str(project), "--fold", str(fold), "--phase", phase]
                print(f"GPU {gpu}: fold {fold} {phase}; new process, physical batch 256.", flush=True)
                p = subprocess.Popen(command, env=env, cwd=project, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                processes.append(p)
                active[gpu] = (p, handle, fold, phase)
            elapsed = time.monotonic()-started
            if elapsed-last_update >= 60:
                count = len(list(output.glob("folds/fold*/runs/*.json")))
                print(f"{count}/120 saved; active GPU:(fold,phase) "
                      f"{ {g:(v[2],v[3]) for g,v in active.items()} }; elapsed {elapsed/3600:.2f} h", flush=True)
                hardware = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
                print(hardware.stdout.strip(), flush=True)
                # Show real TAR progress instead of claiming a short smoke test establishes memory fit.
                for path in sorted(logs.glob("*tar*.log")):
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    if lines: print(path.name + ": " + lines[-1], flush=True)
                last_update = elapsed
            if time.monotonic()-last_export >= 900:
                saved = export_archives(root, live=True)
                last_export = time.monotonic() if saved else time.monotonic()-840
            if jobs or active: time.sleep(5)
    except KeyboardInterrupt:
        failed = "Interrupted; saved work retained."
    finally:
        stop_workers(processes)
        for h in handles: h.close()
        if failed: print(failed, flush=True)
        try:
            validate_resume(root)
            report_completed(vm, output, announced)
            complete = vm.write_reports(output)
        finally:
            export_archives(root)
            for path in sorted(logs.glob("*.log")):
                print("\n--- " + path.name + " ---\n" + "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-45:]), flush=True)
    if not complete:
        raise SystemExit("INCOMPLETE: saved work retained. Inspect logs before resuming; no automatic memory fallback.")
    print("COMPLETE: all 120 original-VM evaluations audited.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["validate", "prepare", "supervise", "worker", "export"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--project")
    parser.add_argument("--fold", type=int, choices=[4, 5])
    parser.add_argument("--phase", choices=["baselines", "tar"])
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    if args.mode == "export":
        if Path(args.root).name != RUN_NAME or not (Path(args.root) / "transfer_index.json").is_file():
            parser.error("Expected the experiment-specific resume root")
        export_archives(args.root)
        return
    print("RESUME CHECK: " + json.dumps(validate_resume(args.root)), flush=True)
    if args.mode == "validate": return
    if not args.project: parser.error("--project is required")
    if args.mode == "prepare":
        verify_environment(args.root, args.project)
        changed = relocate_checkpoint_paths(args.root)
        print("Only relocated checkpoint path fields in working copies:", changed, flush=True)
    elif args.mode == "supervise":
        try: gpus = tuple(int(x) for x in args.gpus.split(","))
        except ValueError: parser.error("Invalid GPU indices")
        if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus): parser.error("Invalid GPU indices")
        supervise(args.root, args.project, gpus)
    else:
        if args.fold is None or args.phase is None: parser.error("worker requires --fold and --phase")
        worker(args.root, args.project, args.fold, args.phase)


if __name__ == "__main__": main()
