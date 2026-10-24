"""Schedule official one-GPU LinkedIn folds on two devices; native batch 256, live R2 tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from kaggle_linkedin import stop_workers, tail, worker_environment
from run_full_linkedin_vm import PROTOCOL, audit_runs, fold_dir, write_reports

RUN_NAME = "linkedin_full_batch256"


def restore_input(source, root, expected_revision):
    """Restore only THIS batch-256 protocol into an empty root; also embedded in notebook bootstrap."""
    import json, shutil, stat, tempfile, zipfile
    from pathlib import Path, PurePosixPath
    root, source = Path(root).resolve(), Path(source)
    protocol = "linkedin_native_physical256_v2"
    if root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to overwrite existing working output")

    def validate(revision, manifest):
        if (revision != expected_revision or manifest.get("source_revision") != revision
                or manifest.get("protocol") != protocol or manifest.get("dataset") != "linkedin_salary"
                or manifest.get("training_arguments", {}).get("per_device_train_batch_size") != 256
                or manifest.get("training_arguments", {}).get("gradient_accumulation_steps") != 1):
            raise ValueError("Checkpoint source/protocol differs. Old 16x16 results cannot enter this experiment.")

    root.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("Unexpected checkpoint symlink")
        validate((source / "source_revision.txt").read_text().strip(),
                 json.loads((source / "linkedin_salary/manifest.json").read_text()))
        shutil.copytree(source, root, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(source) as archive:
            names = set()
            for info in archive.infolist():
                name = info.filename
                if (name in names or "\\" in name or ".." in PurePosixPath(name).parts
                        or stat.S_ISLNK(info.external_attr >> 16)
                        or not (root.parent / name).resolve().is_relative_to(root)):
                    raise ValueError("Unsafe checkpoint archive member: " + name)
                names.add(name)
            validate(archive.read(root.name + "/source_revision.txt").decode().strip(),
                     json.loads(archive.read(root.name + "/linkedin_salary/manifest.json")))
            with tempfile.TemporaryDirectory(prefix="linkedin256_restore_", dir=root.parent) as scratch:
                archive.extractall(scratch)
                if root.exists():
                    root.rmdir()  # Verified empty above.
                (Path(scratch) / root.name).replace(root)


def export_archives(root, *, live=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    if live:
        ready = set()
        for state_path in root.glob("**/trainer/checkpoint-*/trainer_state.json"):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.parent.name == f"checkpoint-{state['global_step']}":
                    ready.add(state_path.parent)
            except (OSError, ValueError, KeyError):
                pass
        paths = [p for p in paths if p.suffix not in (".tmp", ".log") and
                 all(parent in ready for parent in p.parents
                     if parent.name.startswith("checkpoint-") and parent.parent.name == "trainer")]
    for suffix, checkpoints in [("results", False), ("checkpoint", True)]:
        destination = root.parent / f"{root.name}_{suffix}.zip"
        temporary = destination.with_suffix(".zip.tmp")
        selected = [p for p in paths if checkpoints or "checkpoints" not in p.relative_to(root).parts]
        try:
            before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in selected} if live else {}
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for path in selected:
                    archive.write(path, (Path(root.name) / path.relative_to(root)).as_posix())
            if live and any((p.stat().st_size, p.stat().st_mtime_ns) != before[p] for p in selected):
                raise OSError("Files changed during the live snapshot")
        except OSError:
            temporary.unlink(missing_ok=True)
            if not live:
                raise
            print("Checkpoint changed during autosave; retained the previous ZIP.", flush=True)
            return False
        temporary.replace(destination)
        print(f"Saved {destination.name} ({destination.stat().st_size/2**20:.1f} MiB)", flush=True)
    return True


def emit_reports(output, announced):
    for fold in range(6):
        report = fold_dir(output, fold) / "r2_summary.txt"
        if fold in announced or not report.exists() or audit_runs(output, [fold]):
            continue
        print(f"\n===== FOLD {fold} COMPLETE: ALL 20 EVALUATIONS AUDITED =====\n"
              + report.read_text(encoding="utf-8"), flush=True)
        timing = fold_dir(output, fold) / "timing.json"
        if timing.exists():
            print("Fold timing/memory: " + timing.read_text(encoding="utf-8"), flush=True)
        path = fold_dir(output, fold) / "runs" / f"cat_joint_tar_10k_fold{fold}.json"
        print("Shared TAR training: " + json.dumps(json.loads(path.read_text())["tar_finetune"]), flush=True)
        announced.add(fold)
        write_reports(output)


def progress(output):
    ok, failures = 0, []
    for path in Path(output).glob("folds/fold*/runs/*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if row.get("status") == "ok":
            ok += 1
        else:
            failures.append(f"{path.name}: {row.get('error', row.get('status'))}")
    return ok, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    args = parser.parse_args()
    root = Path(args.results_root).resolve()
    if root.name != RUN_NAME:
        raise ValueError("Use the separate linkedin_full_batch256 root. Do not mix old LinkedIn runs.")
    worker = Path(__file__).with_name("run_full_linkedin_vm.py")
    common = [sys.executable, "-u", str(worker), "--results_root", str(root)]
    subprocess.run(common + ["--prepare_only"], env=worker_environment(0), check=True)
    output = root / "linkedin_salary"
    manifest = json.loads((output / "manifest.json").read_text())
    revision = (root / "source_revision.txt").read_text().strip()
    if manifest["protocol"] != PROTOCOL or manifest["source_revision"] != revision:
        raise ValueError("Manifest/source mismatch")
    for gpu in range(2):
        record = json.loads((output / f"gpu_preflight_gpu{gpu}.json").read_text())
        if not record.get("passed") or record.get("training_arguments") != manifest["training_arguments"]:
            raise ValueError(f"GPU {gpu} native-batch preflight did not pass")
    problems = audit_runs(output, allow_missing=True)
    if problems:
        raise ValueError("Existing saved results need inspection before resume.\n" + "\n".join(problems))
    announced = set()
    emit_reports(output, announced)
    if not audit_runs(output):
        write_reports(output)
        export_archives(root)
        return
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    processes, handles, active = [], [], {}
    pending = [f for f in range(6) if audit_runs(output, [f])]
    started = time.monotonic()
    reason = "Workers finished"
    complete = False
    try:
        last_update, last_export = -60, started
        while pending or active:
            failed = False
            for gpu, (process, handle, fold) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[gpu]
                if code != 0 or audit_runs(output, [fold]):
                    reason = f"ERROR: GPU {gpu} fold {fold} stopped (exit {code}); settings unchanged."
                    failed = True
                    break
            if failed:
                break
            # One central queue: no fold is assigned twice, and an idle GPU immediately
            # gets the next unstarted fold. A running fold is never split across devices.
            for gpu in range(2):
                if gpu in active or not pending:
                    continue
                fold = pending.pop(0)
                handle = (logs / f"gpu{gpu}.log").open("a", encoding="utf-8")
                handles.append(handle)
                print(f"Assigning fold {fold} to GPU {gpu} (one GPU, physical batch 256).", flush=True)
                process = subprocess.Popen(common + ["--folds", str(fold), "--fold_worker"],
                    env=worker_environment(gpu), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                active[gpu] = (process, handle, fold)
            elapsed = time.monotonic() - started
            emit_reports(output, announced)
            count, failures = progress(output)
            if elapsed - last_update >= 60:
                assignment = {gpu: item[2] for gpu, item in active.items()}
                print(f"{count}/120 evaluations saved; active GPU:fold {assignment}; queued folds {pending}; "
                      f"elapsed {elapsed/3600:.2f} h", flush=True)
                hardware = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
                print(hardware.stdout.strip(), flush=True)
                last_update = elapsed
            if failures or any(p.poll() not in (None, 0) for p, _, _ in active.values()):
                reason = "ERROR: worker failed; batch, precision and all training settings remain unchanged."
                print("\n".join(failures), flush=True)
                break
            if time.monotonic() - last_export >= 15*60:
                saved = export_archives(root, live=True)
                last_export = time.monotonic() if saved else time.monotonic() - 14*60
            if pending or active:
                time.sleep(5)
    except KeyboardInterrupt:
        reason = "INTERRUPTED: preserving saved work for resumption."
    finally:
        stop_workers(processes)
        for handle in handles:
            handle.close()
        print(reason, flush=True)
        emit_reports(output, announced)
        try:
            complete = write_reports(output)
        finally:
            export_archives(root)
        for gpu in range(2):
            tail(logs / f"gpu{gpu}.log")
    print("COMPLETE: all 120 evaluations audited." if complete else "INCOMPLETE: retain checkpoint ZIP and inspect logs.", flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
