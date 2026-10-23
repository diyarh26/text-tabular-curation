"""Two-T4 candidate supervisor: live fold tables, no elapsed-time cutoff, resumable ZIPs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

from kaggle_linkedin import stop_workers, tail, worker_environment
from run_full_candidate import CANDIDATES, audit_runs, experiment_run_name, make_report, progress_text


def restore_archive(archive_path, root, dataset):
    """Validate a dataset-specific checkpoint completely before replacing an empty run root.

    Also embedded in notebook bootstrap, before repository code can be downloaded.
    """
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to restore over existing output; keep it or choose a fresh run root")
    with zipfile.ZipFile(archive_path) as archive:
        names = set()
        for info in archive.infolist():
            name = info.filename
            destination = (root.parent / name).resolve()
            if ("\\" in name or name.startswith("/") or not destination.is_relative_to(root) or
                    stat.S_ISLNK(info.external_attr >> 16) or name in names):
                raise ValueError("Unsafe or unexpected checkpoint member: " + name)
            names.add(name)
        try:
            source = archive.read(root.name + "/source_revision.txt").decode().strip()
            manifest = json.loads(archive.read(root.name + "/" + dataset + "/manifest.json"))
        except (KeyError, UnicodeDecodeError, ValueError):
            raise ValueError("Checkpoint metadata missing or invalid") from None
        if (not re.fullmatch(r"[0-9a-f]{40}", source) or manifest.get("source_revision") != source or
                manifest.get("dataset") != dataset or manifest.get("protocol_version") != "candidate_full_v1"):
            raise ValueError("Checkpoint dataset/source/protocol does not match this notebook")
        with tempfile.TemporaryDirectory(prefix="candidate_restore_", dir=root.parent) as temporary:
            archive.extractall(temporary)
            restored = Path(temporary) / root.name
            if root.exists():
                root.rmdir()  # Only an empty, explicitly verified root can be replaced.
            restored.replace(root)
    return source


def progress(root, candidate):
    ok, failures = 0, []
    for path in (Path(root) / candidate.name / "folds").glob("fold*/runs/*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # A writer may still be completing this JSON.
        if row.get("status") == "ok":
            ok += 1
        else:
            failures.append(f"{path.name}: {row.get('error', row.get('status'))}")
    return ok, failures


def emit_fold_reports(root, candidate, announced):
    """Forward completed worker tables into the RUNNING notebook cell, not only worker logs."""
    output = Path(root) / candidate.name
    for fold in range(6):
        report = output / f"fold{fold}_report.txt"
        if fold in announced or not report.exists() or audit_runs(output, candidate, [fold]):
            continue
        print("\n" + report.read_text(encoding="utf-8"), flush=True)
        announced.add(fold)
        text = "===== CUMULATIVE RESULTS: FULLY COMPLETED FOLDS ONLY =====\n"
        text += progress_text(output, candidate, sorted(announced))
        destination = output / "progress_report.txt"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(destination)
        print(text, flush=True)


def export_archives(root, *, live=False):
    """Keep previous snapshots if checkpoint files change during a live copy."""
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
                continue
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
                raise OSError("Files changed while copying the live snapshot")
        except OSError:
            temporary.unlink(missing_ok=True)
            if not live:
                raise
            print("Checkpoint changed during autosave; kept the previous ZIP and will retry.", flush=True)
            return False
        temporary.replace(destination)
        print(f"Saved {destination} ({destination.stat().st_size/2**20:.1f} MiB)", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=CANDIDATES, required=True)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--micro_batch", type=int, choices=[16, 32, 64, 256], default=256)
    args = parser.parse_args()
    candidate = CANDIDATES[args.dataset]
    root = Path(args.results_root).resolve()
    if root.name != experiment_run_name(candidate, args.micro_batch):
        raise ValueError("Use the candidate-specific run root to isolate output/checkpoint archives")
    root.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("run_full_candidate.py")
    common = [sys.executable, "-u", str(worker), "--dataset", candidate.key,
              "--results_root", str(root), "--micro_batch", str(args.micro_batch)]
    env = worker_environment(0)
    subprocess.run(common + ["--prepare_only"], env=env, check=True)
    output = root / candidate.name
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    revision = root / "source_revision.txt"
    if revision.exists() and revision.read_text().strip() != manifest["source_revision"]:
        raise ValueError("Output source revision differs from the verified manifest")
    revision.write_text(manifest["source_revision"] + "\n", encoding="utf-8")
    announced = set()
    emit_fold_reports(root, candidate, announced)
    if not audit_runs(output, candidate):
        make_report(output, candidate)
        export_archives(root)
        return
    # Retain failed attempts for diagnosis; let workers retry them without false new-error detection.
    for path in output.glob("folds/fold*/runs/*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            row = {"status": "interrupted_json_write"}
        if row.get("status") != "ok":
            target = path.parent.parent / "failed_attempts" / f"{time.time_ns()}_{path.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    processes, handles = [], []
    started = time.monotonic()
    reason = "Workers finished"
    complete = False
    try:
        for gpu, folds in [(0, "0,2,4"), (1, "1,3,5")]:
            handle = (logs / f"gpu{gpu}.log").open("a", encoding="utf-8")
            handles.append(handle)
            processes.append(subprocess.Popen(common + ["--folds", folds], env=worker_environment(gpu),
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        last_update, last_export = -60, started
        while any(p.poll() is None for p in processes):
            elapsed = time.monotonic()-started
            emit_fold_reports(root, candidate, announced)
            count, failures = progress(root, candidate)
            if elapsed-last_update >= 60:
                print(f"{count}/120 evaluations saved; workers {[p.poll() for p in processes]}; "
                      f"elapsed {elapsed/3600:.2f} h", flush=True)
                hardware = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
                print(hardware.stdout.strip(), flush=True)
                last_update = elapsed
            if failures or any(p.poll() not in (None, 0) for p in processes):
                reason = "ERROR: a worker failed. Settings unchanged; inspect GPU logs."
                print("\n".join(failures), flush=True)
                break
            if time.monotonic()-last_export >= 15*60:
                saved = export_archives(root, live=True)
                last_export = time.monotonic() if saved else time.monotonic()-14*60
            time.sleep(5)
    except KeyboardInterrupt:
        reason = "INTERRUPTED by user; exporting saved work."
    finally:
        stop_workers(processes)
        for handle in handles:
            handle.close()
        print(reason, flush=True)
        emit_fold_reports(root, candidate, announced)
        try:
            complete = make_report(output, candidate)
        finally:
            export_archives(root)
        for gpu in range(2):
            tail(logs / f"gpu{gpu}.log")
    print("COMPLETE: all 120 evaluations audited." if complete else
          "INCOMPLETE: keep the checkpoint ZIP and resume the same experiment.", flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
