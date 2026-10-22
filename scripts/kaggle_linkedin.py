"""Supervise two independent fold workers, stop on errors/time budget, export resumable outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

from run_full_linkedin import audit_runs


def worker_environment(gpu):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}


def progress(root):
    ok, failures = 0, []
    for path in (Path(root) / "linkedin_salary/folds").glob("fold*/runs/*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # A worker may be in the middle of writing this file.
            continue
        if row.get("status") == "ok":
            ok += 1
        else:
            failures.append(f"{path.name}: {row.get('error', row.get('status'))}")
    return ok, failures


def export_archives(root):
    """Archive only experiment files, never the environment, repository, caches or secrets."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    for name, checkpoints in [("linkedin_full_results.zip", False), ("linkedin_full_checkpoint.zip", True)]:
        destination = root.parent / name
        temporary = destination.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in paths:
                if not checkpoints and "checkpoints" in path.relative_to(root).parts:
                    continue
                archive.write(path, (Path("linkedin_full") / path.relative_to(root)).as_posix())
        temporary.replace(destination)
        print(f"Saved {destination} ({destination.stat().st_size / 2**20:.1f} MiB)", flush=True)


def stop_workers(processes):
    # Linux only: process groups include any child trainers; no orphan GPU processes.
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 40
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(1)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def tail(path, lines=45):
    if Path(path).exists():
        print(f"\n--- {path.name} ---\n" + "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]),
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--max_seconds", type=float, default=10 * 3600)
    parser.add_argument("--micro_batch", type=int, choices=[16, 32, 256], default=256)
    args = parser.parse_args()
    if args.max_seconds < 60:
        raise SystemExit("Less than one minute remains in the notebook budget; save outputs and start a new session.")
    root = Path(args.results_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("run_full_linkedin.py")
    common = [sys.executable, "-u", str(worker), "--results_root", str(root),
              "--micro_batch", str(args.micro_batch)]
    env = worker_environment(0)
    subprocess.run(common + ["--prepare_only"], env=env, check=True)
    output = root / "linkedin_salary"
    if not audit_runs(output):
        subprocess.run(common + ["--report_only"], env=env, check=True)
        export_archives(root)
        return
    # Only retry failed JSONs in a new run; ignore old failures until the worker overwrites them.
    # Preserve failed attempts outside the active runs directory for diagnosis.
    for path in output.glob("folds/fold*/runs/*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            row = {"status": "interrupted_json_write"}
        if row.get("status") != "ok":
            archived = path.parent.parent / "failed_attempts" / f"{time.time_ns()}_{path.name}"
            archived.parent.mkdir(parents=True, exist_ok=True)
            path.replace(archived)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    processes, handles = [], []
    start = time.monotonic()
    reason = "workers finished"
    try:
        for gpu, folds in [(0, "0,2,4"), (1, "1,3,5")]:
            handle = (logs / f"gpu{gpu}.log").open("a", encoding="utf-8")
            handles.append(handle)
            processes.append(subprocess.Popen(common + ["--folds", folds], env=worker_environment(gpu),
                                              stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        last_update = -60
        while any(p.poll() is None for p in processes):
            elapsed = time.monotonic() - start
            count, failures = progress(root)
            if elapsed - last_update >= 60:
                print(f"{count}/120 evaluations saved; workers {[p.poll() for p in processes]}; "
                      f"elapsed {elapsed / 3600:.2f} h", flush=True)
                hardware = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                           "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
                print(hardware.stdout.strip(), flush=True)
                last_update = elapsed
            if failures or any(p.poll() not in (None, 0) for p in processes):
                reason = "ERROR: a worker failed; settings have not been adjusted. Inspect GPU logs."
                print("\n".join(failures), flush=True)
                break
            if elapsed >= args.max_seconds:
                reason = "PAUSED at the notebook time budget; resume saved checkpoints in the next session."
                break
            time.sleep(5)
    except KeyboardInterrupt:
        reason = "PAUSED by user; exporting saved work."
    finally:
        stop_workers(processes)
        for handle in handles:
            handle.close()
        print(reason, flush=True)
        # This runs only after both writers stop. A partial report never gives a final verdict.
        try:
            subprocess.run(common + ["--report_only"], env=env, check=True)
        finally:
            export_archives(root)
        for gpu in range(2):
            tail(logs / f"gpu{gpu}.log")
    complete = not audit_runs(output)
    print("COMPLETE: all 120 evaluations audited." if complete else "INCOMPLETE: keep the checkpoint ZIP to resume.",
          flush=True)
    if not complete and (reason.startswith("ERROR") or any(p.returncode not in (0, -signal.SIGINT, -signal.SIGKILL)
                                                         for p in processes) and not reason.startswith("PAUSED")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
