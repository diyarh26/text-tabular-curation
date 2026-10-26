"""Run LinkedIn Salary through the unmodified July MulTaBench benchmark entry point.

Only dataset registration/annotation is added to the external official checkout.
This file owns installation, GPU isolation, scheduling and official W&B export.
Run --help for commands. Nothing here creates or terminates a rented Pod.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import tarfile
import time
import urllib.request
import uuid

PIN = "599bd6a5631c96f8aef297cc4cb6e4c197ae0dca"
OFFICIAL = "https://github.com/alanarazi7/MulTaBench.git"
DATASET = "REG_TEXT_LINKEDIN_SALARY"
CSV_SHA256 = "8880f84563535c60eda01be5b52010dea6f80e318f228ade00166d7cb66d57fa"
FOLDS = tuple(range(5))  # July leaderboard/analysis/pass_matrix.py: _FOLDS = range(5).
MODELS = ("tabm", "cat", "light", "tabpfnv2", "tabpfnv2p5")
CONDITIONS = {
    "structured": ("no_text", "no"),
    "text_frozen": ("text_only", "no"),
    "joint_frozen": ("all", "no"),
    "joint_tar": ("all", "yes"),
}
TEXT_COLUMNS = {"title", "description"}
SECRET_KEYS = (
    "RUNPOD_API_KEY",
    "WANDB_API_KEY",
    "TABPFN_TOKEN",
    "HF_TOKEN",
    "KAGGLE_KEY",
)
ANNOTATION = '''"""LinkedIn Salary: externally prepared, fixed 15,000-row CSV.
No transformations beyond the native curation and feature detection are added.
"""
from multabench.datasets.curation_objects import CuratedFeature, CuratedTarget
from multabench.datasets.objects import FeatureType, SupervisedTask

CONTEXT = "LinkedIn job postings; predict natural log normalized USD salary"
TARGET = CuratedTarget(raw_name="log_salary", task_type=SupervisedTask.REGRESSION)
FEATURES = [CuratedFeature(raw_name=c, feat_type=FeatureType.TEXT)
            for c in ("title", "description")]
COLS_TO_DROP = []
'''


def digest(path: Path, lf: bool = False) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_error(error) -> str:
    message = str(error)
    for key in SECRET_KEYS:
        if os.environ.get(key):
            message = message.replace(os.environ[key], "[REDACTED]")
    return message


def load_secrets(path: Path | None) -> None:
    # Public HF downloads work without a token. Explicitly supply the empty
    # optional environment value expected by the historical TabPFN wrapper.
    os.environ.setdefault("HF_TOKEN", "")
    if path is None:
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError("Invalid credential file line; expected KEY=VALUE")
        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        if value:
            os.environ[key] = value


@contextlib.contextmanager
def lock(path: Path):
    """One coordinator/preparer; stale locks require inspection, never silent takeover."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Lock exists: {path}. Inspect its PID before removing it.")
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def command(arguments, *, cwd=None, capture=False, env=None):
    return subprocess.run(
        [str(a) for a in arguments],
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def git(source: Path, *arguments) -> str:
    return command(["git", "-C", source, *arguments], capture=True).stdout.strip()


def original_file(source: Path, name: str) -> str:
    return command(["git", "-C", source, "show", f"{PIN}:{name}"], capture=True).stdout


def integration_files(source: Path) -> dict[str, str]:
    registry = original_file(source, "multabench/datasets/all_datasets.py")
    anchor = "class UrlDatasetID(Enum):\n"
    assert registry.count(anchor) == 1
    registry = registry.replace(
        anchor,
        anchor
        + '    REG_TEXT_LINKEDIN_SALARY = "https://www.kaggle.com/datasets/arshkon/linkedin-job-postings"\n',
    )
    loader = original_file(source, "multabench/datasets/downloading.py")
    anchor = "    dataset2local = {\n"
    assert loader.count(anchor) == 1
    loader = loader.replace(
        anchor,
        anchor
        + '        UrlDatasetID.REG_TEXT_LINKEDIN_SALARY: "linkedin_salary.csv",\n',
    )
    return {
        "multabench/datasets/all_datasets.py": registry,
        "multabench/datasets/downloading.py": loader,
        f"multabench/datasets/annotated/{DATASET}.py": ANNOTATION,
    }


def verify_source(source: Path, integrated: bool = True) -> None:
    if git(source, "remote", "get-url", "origin").removesuffix(
        ".git"
    ) != OFFICIAL.removesuffix(".git"):
        raise RuntimeError("Official checkout has the wrong origin")
    if git(source, "rev-parse", "HEAD") != PIN:
        raise RuntimeError("Official checkout is not at the required July commit")
    attached = (
        subprocess.run(
            ["git", "-C", str(source), "symbolic-ref", "-q", "HEAD"],
            stdout=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if attached:
        raise RuntimeError("Official source must be checked out detached")
    first_parents = git(
        source, "rev-list", "--first-parent", "origin/master"
    ).splitlines()
    if PIN not in first_parents:
        raise RuntimeError("Pin is not on official master's first-parent history")
    cutoff = git(
        source,
        "log",
        "origin/master",
        "--first-parent",
        "--until=2026-07-26T23:59:59+03:00",
        "-1",
        "--format=%H",
    )
    if cutoff != PIN:
        raise RuntimeError("Pin is not the last master first-parent commit on July 26")
    additions = integration_files(source)
    changed = set(git(source, "diff", "--name-only", "HEAD").splitlines())
    if changed - additions.keys():
        raise RuntimeError(
            f"Upstream code changed outside integration: {sorted(changed)}"
        )
    untracked = set(
        git(source, "ls-files", "--others", "--exclude-standard").splitlines()
    )
    permitted = set(additions) | {
        "multabench/datasets/url_datasets/linkedin_salary.csv"
    }
    if untracked - permitted:
        raise RuntimeError(
            f"Unexpected untracked official source files: {sorted(untracked - permitted)}"
        )
    if integrated:
        for name, expected in additions.items():
            if (source / name).read_text(encoding="utf-8") != expected:
                raise RuntimeError(
                    f"Integration differs from the declared addition: {name}"
                )
        staged = source / "multabench/datasets/url_datasets/linkedin_salary.csv"
        if digest(staged, lf=True) != CSV_SHA256:
            raise RuntimeError("Staged dataset checksum changed")


def prepare(args) -> None:
    args.runtime.mkdir(parents=True, exist_ok=True)
    with lock(args.runtime / "prepare.lock"):
        if not args.source.exists():
            command(["git", "clone", "--no-checkout", OFFICIAL, args.source])
            command(["git", "-C", args.source, "checkout", "--detach", PIN])
        verify_source(args.source, integrated=False)
        if digest(args.dataset, lf=True) != CSV_SHA256:
            raise RuntimeError("Input CSV is not the approved unchanged sample")
        for name, expected in integration_files(args.source).items():
            destination = args.source / name
            if destination.exists():
                actual = destination.read_text(encoding="utf-8")
                if actual != expected and actual != original_file(args.source, name):
                    raise RuntimeError(
                        f"Refusing to overwrite unexpected source contents: {name}"
                    )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(expected, encoding="utf-8", newline="\n")
        staged = args.source / "multabench/datasets/url_datasets/linkedin_salary.csv"
        staged.parent.mkdir(parents=True, exist_ok=True)
        if staged.exists() and digest(staged, lf=True) != CSV_SHA256:
            raise RuntimeError("Refusing to overwrite a different staged dataset")
        shutil.copyfile(args.dataset, staged)
        verify_source(args.source)
        atomic_json(
            args.runtime / "integration.json",
            {
                "pin": PIN,
                "dataset": DATASET,
                "csv_sha256": CSV_SHA256,
                "files": {
                    k: digest(args.source / k) for k in integration_files(args.source)
                },
                "learning_code_modified": False,
            },
        )
    print(f"Registered {DATASET} through the native URL/local-file curation path.")


def install(args) -> None:
    if sys.platform != "linux":
        raise RuntimeError(
            "Install the official Linux requirements in Linux (Python 3.11)"
        )
    verify_source(args.source)
    if args.system_tools:
        if os.geteuid() != 0:
            raise RuntimeError("Install tmux/git as root or omit --system-tools")
        command(["apt-get", "update"])
        command(["apt-get", "install", "-y", "tmux", "git", "ca-certificates"])
    version = command(
        [args.python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        capture=True,
    ).stdout.strip()
    if version != "3.11":
        raise RuntimeError("Historical setup requires Python 3.11")
    venv = (args.venv_dir or args.runtime / "venv").resolve()
    if venv.is_relative_to(Path(__file__).resolve().parents[1]):
        raise RuntimeError("The environment must stay outside the final submission")
    if not venv.exists():
        command([args.python, "-m", "venv", venv])
    alias = args.runtime / "venv"
    if venv != alias.resolve():
        if alias.exists() or alias.is_symlink():
            raise RuntimeError("Runtime venv already points at a different environment")
        alias.symlink_to(venv, target_is_directory=True)
    python = venv / "bin/python"
    command([python, "-m", "pip", "install", "uv"])
    # The checked-in complete lock is the source of truth, including its git pin.
    command(
        [
            python,
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "-r",
            args.source / "requirements.txt",
        ]
    )
    print(f"Installed historical environment: {python}")


def verify_dependencies(source: Path) -> dict:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Run with the historical Python 3.11 environment")
    mismatches = []
    for line in (source / "requirements.txt").read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ ;]+)", line)
        if match:
            name, expected = match.groups()
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual = "MISSING"
            if actual != expected:
                mismatches.append(f"{name}: {actual} (expected {expected})")
    direct = importlib.metadata.distribution("sap-rpt-oss").read_text("direct_url.json")
    required_commit = "4310453f3bbb9c9c403b029e966565e7fe652d97"
    if (
        not direct
        or json.loads(direct).get("vcs_info", {}).get("commit_id") != required_commit
    ):
        mismatches.append("sap-rpt-oss git revision mismatch")
    if mismatches:
        raise RuntimeError("Historical environment mismatch:\n" + "\n".join(mismatches))
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "tabstar",
            "tabpfn",
            "pandas",
            "numpy",
            "scikit-learn",
            "wandb",
        )
    }


def native_imports(source: Path) -> None:
    sys.path.insert(0, str(source))
    from multabench.datasets import all_datasets

    if not Path(all_datasets.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Imported MulTaBench from a different installed version")


def inspect_dataset(args) -> None:
    verify_source(args.source)
    versions = verify_dependencies(args.source)
    native_imports(args.source)
    import pandas as pd
    from tabstar.preprocessing.splits import split_to_test, split_to_val
    from multabench.datasets.utils import dataset_from_name
    from multabench.datasets.downloading import download_dataset
    from multabench.datasets.multimodal import MultimodalState
    from multabench.baselines.preprocessing.sampling import subsample_dataset
    from multabench.preprocessing.feat_types import detect_text_features
    from multabench.preprocessing.discretize import discretize_numerical

    dataset_id = dataset_from_name(DATASET)
    full = download_dataset(dataset_id, multimodal_state=MultimodalState.ALL)
    reference = pd.read_csv(args.dataset)
    from tabstar.preprocessing.texts import normalize_col_name

    reference = reference.rename(columns={c: normalize_col_name(c) for c in reference})
    pd.testing.assert_frame_equal(full.x, reference.drop(columns=[full.y.name]))
    pd.testing.assert_series_equal(full.y, reference[full.y.name])
    if set(detect_text_features(full.x)) != TEXT_COLUMNS or len(full.y) != 15000:
        raise RuntimeError(
            "Native data roles or row count differ from the approved dataset"
        )
    conditions = {}
    for condition, (state, _) in CONDITIONS.items():
        enum = {
            "no_text": MultimodalState.NO_TEXT,
            "text_only": MultimodalState.TEXT_ONLY,
            "all": MultimodalState.ALL,
        }[state]
        dataset = download_dataset(dataset_id, multimodal_state=enum)
        expected = (
            TEXT_COLUMNS
            if state == "text_only"
            else (set(full.x) - TEXT_COLUMNS if state == "no_text" else set(full.x))
        )
        if set(dataset.x) != expected or not dataset.y.equals(full.y):
            raise RuntimeError(f"Wrong native condition features/target: {condition}")
        conditions[condition] = list(dataset.x)
    splits = []
    for fold in FOLDS:
        x, y = subsample_dataset(full.x, full.y, False, 10000, fold)
        train, test, y_train, y_test = split_to_test(x, y, False, fold, 10000)
        fit, val, y_fit, y_val = split_to_val(train, y_train, False)
        if (len(y), len(y_train), len(y_test), len(y_fit), len(y_val)) != (
            11500,
            10000,
            1500,
            9000,
            1000,
        ):
            raise RuntimeError("Unexpected historical split sizes")
        details = {
            "fold": fold,
            "seed": 42 + fold,
            "sampled": len(y),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "learner_fit": len(y_fit),
            "learner_val": len(y_val),
        }
        for label, xx, yy in (("val_learners", fit, y_fit), ("tabpfn", train, y_train)):
            if set(detect_text_features(xx)) != TEXT_COLUMNS:
                raise RuntimeError(
                    f"Native learner text detection changed: fold {fold}, {label}"
                )
            _, _, tar_fit, tar_val = split_to_val(
                xx, discretize_numerical(yy, n_bins=20), True
            )
            details[label + "_tar_fit"] = len(tar_fit)
            details[label + "_tar_val"] = len(tar_val)
        splits.append(details)
    report = {
        "versions": versions,
        "dataset_id": dataset_id.name,
        "target": full.y.name,
        "rows": len(full.y),
        "conditions": conditions,
        "splits": splits,
        "entrypoint_evaluation_tested": False,
    }
    atomic_json(args.runtime / "dataset-inspection.json", report)
    print(json.dumps(report, indent=2))


def native_arguments(
    model: str, condition: str, fold: int, project: str, smoke=False
) -> list[str]:
    state, tune = CONDITIONS[condition]
    args = [
        "--model",
        model,
        "--dataset_name",
        DATASET,
        "--fold",
        str(fold),
        "--multimodal_state",
        state,
        "--tune_e5",
        tune,
        "--project",
        project,
        "--verbose",
    ]
    # Deliberately omit production overrides: historical benchmark.py supplies all defaults.
    if smoke:
        args += ["--train_examples", "512", "--e5_epochs", "1"]
    return args


def grid(smoke=False) -> list[dict]:
    if smoke:
        return [
            {"model": "light", "condition": c, "fold": 0}
            for c in ("structured", "text_frozen", "joint_tar")
        ]
    return [
        {"model": m, "condition": c, "fold": f}
        for f in FOLDS
        for c in CONDITIONS
        for m in MODELS
    ]


def job_key(job: dict) -> str:
    return f"{job['model']}__{job['condition']}__{job['fold']}"


def gpu_inventory() -> list[dict]:
    fields = ("index", "uuid", "pci.bus_id", "name", "memory.total", "driver_version")
    output = command(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).stdout
    return [
        dict(zip(fields, [v.strip() for v in row]))
        for row in csv.reader(output.splitlines())
    ]


def selected_gpus(value: str) -> list[str]:
    inventory = gpu_inventory()
    choices = [v.strip() for v in value.split(",") if v.strip()]
    lookup = {row["index"]: row["uuid"] for row in inventory}
    known = {row["uuid"] for row in inventory}
    result = [lookup.get(v, v) for v in choices]
    if not result or len(set(result)) != len(result) or not set(result) <= known:
        raise RuntimeError(
            "GPU selection must name distinct allocated GPU indices or UUIDs"
        )
    return result


def cuda_mask_for_uuid(gpu_uuid: str, inventory: list[dict]) -> str:
    """Historical TabSTAR accepts numeric masks; derive CUDA PCI order explicitly."""
    ordered = sorted(
        inventory,
        key=lambda row: tuple(
            int(part, 16) for part in re.split(r"[:.]", row["pci.bus_id"])
        ),
    )
    matches = [i for i, row in enumerate(ordered) if row["uuid"] == gpu_uuid]
    if len(matches) != 1:
        raise RuntimeError(
            "Allocated GPU UUID is absent or duplicated in PCI inventory"
        )
    return str(matches[0])


def isolated_environment(gpu_uuid: str, source: Path, work: Path) -> dict:
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=cuda_mask_for_uuid(gpu_uuid, gpu_inventory()),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        GPU="0",
        PYTHONPATH=str(source),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        WANDB_DIR=str(work / "wandb"),
        TMPDIR=str(work / "tmp"),
    )
    # No precision, thread, batch, trainer, seed or learner-default overrides.
    for folder in (work / "wandb", work / "tmp"):
        folder.mkdir(parents=True, exist_ok=True)
    return env


def verify_one_gpu(expected: str) -> dict:
    if (
        not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit()
        or os.environ.get("GPU") != "0"
    ):
        raise RuntimeError("GPU mask was not set before worker imports")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker must see exactly one functioning CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(getattr(properties, "uuid", "")).removeprefix("GPU-").lower()
    if actual_uuid != expected.removeprefix("GPU-").lower():
        raise RuntimeError(
            "CUDA device UUID does not match this worker's allocated GPU"
        )
    a = torch.arange(1024, device="cuda:0", dtype=torch.float32).reshape(32, 32)
    if not torch.isclose((a @ a.T)[0, 0], torch.tensor(10416.0, device="cuda:0")):
        raise RuntimeError("Real CUDA matrix multiplication check failed")
    torch.cuda.synchronize()
    report = {
        "uuid": expected,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "visible_devices": 1,
        "device": "cuda:0",
        "name": properties.name,
        "vram_bytes": properties.total_memory,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_tensor_computation": True,
    }
    del a
    torch.cuda.empty_cache()
    return report


def require_mount(runtime: Path, mount: Path) -> dict:
    runtime, mount = runtime.resolve(), mount.resolve()
    mount_record = None
    if Path("/proc/self/mountinfo").exists():
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            target = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
            if Path(target) == mount:
                mount_record = line
                break
    if (
        mount == Path("/")
        or not (mount.is_mount() or mount_record)
        or not runtime.is_relative_to(mount)
    ):
        raise RuntimeError("Runtime must be under the verified persistent mount")
    probe = runtime / ".storage-probe"
    with probe.open("wb") as f:
        f.write(b"july-native-storage-check\n")
        f.flush()
        os.fsync(f.fileno())
    if probe.read_bytes() != b"july-native-storage-check\n":
        raise RuntimeError("Persistent storage write/read failed")
    probe.unlink()
    return {
        "mount": str(mount),
        "mountpoint_verified": True,
        "mount_record": mount_record,
        "free_bytes": shutil.disk_usage(runtime).free,
    }


def checkpoint_test() -> list[dict]:
    import numpy as np
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Checkpoint fitting/inference requires the isolated GPU; CPU fallback is forbidden"
        )

    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, 3))
    y = x[:, 0] - 0.5 * x[:, 1]
    records = []
    for version in (ModelVersion.V2, ModelVersion.V2_5):
        model = TabPFNRegressor.create_default_for_version(
            version=version,
            random_state=42,
            categorical_features_indices=[],
            ignore_pretraining_limits=True,
            device="cuda:0",  # Preflight only; production uses upstream wrapper defaults.
        )
        model.fit(x[:32], y[:32])
        predictions = model.predict(x[32:])
        if not np.isfinite(predictions).all() or len(predictions) != 8:
            raise RuntimeError(
                f"Invalid predictions from regression checkpoint {version}"
            )
        records.append({"version": str(version), "checkpoint_fit_predict_ok": True})
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    return records


def worker(args) -> None:
    """Invoked only by a process whose CUDA mask was already set by its parent."""
    verify_source(args.source)
    versions = verify_dependencies(args.source)
    identity = verify_one_gpu(args.gpu)
    native_imports(args.source)
    if args.kind == "preflight":
        if args.checkpoints:
            identity["regression_checkpoints"] = checkpoint_test()
        identity["versions"] = versions
        atomic_json(args.work / "preflight.json", identity)
        print(json.dumps(identity, indent=2), flush=True)
        return
    atomic_json(args.work / "gpu-identity.json", identity)
    print("Executing official benchmark.py in a fresh interpreter", flush=True)
    arguments = read_json(args.work / "job.json")["native_arguments"]
    # A separate native working directory avoids concurrent CatBoost/trainer file
    # collisions. It is inside the external Git checkout so upstream's git query
    # still resolves to the official detached commit.
    job = read_json(args.work / "job.json")
    native_cwd = args.source / ".tabstar_linkedin_runtime" / job["run_id"]
    native_cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(native_cwd)
    # exec starts the actual upstream CLI, with no patched logging/evaluation functions.
    os.execv(
        sys.executable, [sys.executable, str(args.source / "benchmark.py"), *arguments]
    )


def worker_command(args, kind: str, gpu: str, work: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--runtime",
        str(args.runtime),
        "--source",
        str(args.source),
        "_worker",
        "--kind",
        kind,
        "--gpu",
        gpu,
        "--work",
        str(work),
    ]


def preflight(args) -> None:
    verify_source(args.source)
    verify_dependencies(args.source)
    gpus = selected_gpus(args.gpus)
    storage = require_mount(args.runtime, args.persistent_mount)
    if not (args.runtime / "dataset-inspection.json").exists():
        raise RuntimeError("Run inspect in the locked environment first")
    for key in ("WANDB_API_KEY", "WANDB_ENTITY", "TABPFN_TOKEN"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing required credential: {key}")
    import wandb

    # Read-only auth check; native smoke jobs verify project write access.
    viewer = wandb.Api(timeout=60).viewer
    if not viewer:
        raise RuntimeError("W&B authentication failed")
    records = []
    for index, gpu in enumerate(gpus):
        work = args.runtime / "preflight" / gpu
        work.mkdir(parents=True, exist_ok=True)
        cmd = worker_command(args, "preflight", gpu, work)
        if index == 0:
            cmd.append("--checkpoints")
        with (work / "console.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                cmd,
                env=isolated_environment(gpu, args.source, work),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        records.append(read_json(work / "preflight.json"))
        print(f"Preflight passed: {gpu}", flush=True)
    atomic_json(
        args.runtime / "preflight.json",
        {
            "storage": storage,
            "gpus": records,
            "inventory": gpu_inventory(),
            "pin": PIN,
            "csv_sha256": CSV_SHA256,
            "evaluator_sha256": digest(Path(__file__)),
        },
    )


def check_credentials(args) -> None:
    """Read-only authentication/license checks; never loads or fits a model."""
    verify_dependencies(args.source)
    for key in ("WANDB_API_KEY", "WANDB_ENTITY", "TABPFN_TOKEN"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing {key}; fill the external credential file")
    import wandb
    from tabpfn.browser_auth import (
        verify_token,
        check_license_accepted,
        _get_license_name,
    )
    from tabpfn.settings import settings

    api = wandb.Api(timeout=30)
    if not api.viewer:
        raise RuntimeError("W&B credentials were not accepted")
    # Read an entity's project listing to detect a misspelled/wrong account name.
    next(iter(api.projects(entity=os.environ["WANDB_ENTITY"])), None)
    token = os.environ["TABPFN_TOKEN"]
    url = settings.tabpfn.auth_api_url
    if verify_token(token, url) is not True:
        raise RuntimeError(
            "PriorLabs key is invalid/expired or its server is unreachable"
        )
    license_name = _get_license_name("tabpfn_2_5")
    if not license_name or check_license_accepted(token, url, license_name) is not True:
        raise RuntimeError(
            "Accept the TabPFN 2.5 license in the existing PriorLabs account before renting a GPU"
        )
    report = {
        "wandb_authenticated": True,
        "wandb_entity": os.environ["WANDB_ENTITY"],
        "priorlabs_authenticated": True,
        "tabpfn_2_5_license": license_name,
        "license_accepted": True,
        "hf_token_required_for_current_public_files": False,
        "model_loaded_or_fitted": False,
    }
    atomic_json(args.runtime / "credential-check.json", report)
    print(json.dumps(report, indent=2))


def export_native(job: dict, destination: Path) -> dict:
    import wandb

    run = wandb.Api(timeout=60).run(job["wandb_path"])
    if run.state != "finished":
        raise RuntimeError(f"W&B run state is {run.state}, expected finished")
    summary = dict(run.summary)
    validate_native_summary(job, summary)
    atomic_json(destination, {"wandb_path": job["wandb_path"], "summary": summary})
    return summary


def native_metrics(summary: dict) -> dict:
    """Read official W&B exports without rounding; preserve their raw JSON separately."""
    if isinstance(summary.get("metrics_dict"), dict):
        return summary["metrics_dict"]
    prefix = "metrics_dict.metrics."
    return {
        "score": summary.get("metrics_dict.score"),
        "metrics": {
            k[len(prefix) :]: v for k, v in summary.items() if k.startswith(prefix)
        },
    }


def validate_native_summary(job: dict, summary: dict) -> None:
    metrics = native_metrics(summary)
    scores = metrics.get("metrics", {})
    if (
        summary.get("dataset") != DATASET
        or summary.get("fold") != job["fold"]
        or not str(summary.get("git", "")).startswith(PIN[:7])
    ):
        raise RuntimeError("Native W&B summary provenance does not match this job")
    names = {
        "tabm": "TabM",
        "cat": "CatBoost",
        "light": "LightGBM",
        "tabpfnv2": "TabPFN-v2 ",
        "tabpfnv2p5": "TabPFN-v2p5",
    }
    if not str(summary.get("model", "")).startswith(names[job["model"]]):
        raise RuntimeError("Native model name does not match the scheduled learner")
    smoke = job["smoke"]
    if summary.get("train_examples") != (512 if smoke else 10000):
        raise RuntimeError("Native train-example setting changed")
    if (
        summary.get("e5_model_name") != "intfloat/e5-small-v2"
        or summary.get("tune_dino") is not False
        or summary.get("request_memory") != "32G"
    ):
        raise RuntimeError(
            "Native encoder/image/memory settings differ from the historical defaults"
        )
    state, tune = CONDITIONS[job["condition"]]
    if (
        bool(summary.get("tune_e5")) != (tune == "yes")
        or summary.get("pca_components") != 30
        or summary.get("no_pca") is not False
    ):
        raise RuntimeError("Native fine-tuning/PCA settings differ from the protocol")
    expected_state_prefix = {
        "all": "all",
        "no_text": "no_text",
        "text_only": "text_only",
    }[state]
    encoded_state = str(summary.get("multimodal_state", "")).lower().split(" ", 1)[0]
    if encoded_state != expected_state_prefix:
        raise RuntimeError("Native feature ablation does not match the job")
    if not smoke and (summary.get("n_train"), summary.get("n_test")) != (10000, 1500):
        raise RuntimeError("Native result has unexpected split counts")
    if job["condition"] == "joint_tar":
        expected = {
            "batch_size": 256,
            "epochs": 1 if smoke else 50,
            "learning_rate": 0.0001,
            "weight_decay": 0.001,
            "lora_rank": 16,
            "text_layers": 3,
            "patience": 3,
        }
        if any(summary.get(k) != v for k, v in expected.items()):
            raise RuntimeError("Effective native TAR settings changed")
    for metric in ("r2", "mse", "rmse"):
        if metric not in scores or not math.isfinite(float(scores[metric])):
            raise RuntimeError(f"Missing/nonfinite native regression metric: {metric}")
    if float(summary["test_score"]) != float(scores["r2"]) or float(
        metrics["score"]
    ) != float(scores["r2"]):
        raise RuntimeError("Native regression score is not its recorded R-squared")


def new_manifest(args, smoke: bool) -> dict:
    for key in ("WANDB_ENTITY", "WANDB_API_KEY"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing {key}")
    identifier = ("smoke-" if smoke else "production-") + uuid.uuid4().hex[:12]
    jobs = []
    for job in grid(smoke):
        run_id = identifier + "-" + job_key(job)
        job = {
            **job,
            "id": job_key(job),
            "run_id": run_id,
            "smoke": smoke,
            "wandb_path": f"{os.environ['WANDB_ENTITY']}/{args.project}/{run_id}",
            "native_arguments": native_arguments(
                job["model"], job["condition"], job["fold"], args.project, smoke
            ),
        }
        jobs.append(job)
    return {
        "experiment": identifier,
        "pin": PIN,
        "csv_sha256": CSV_SHA256,
        "evaluator_sha256": digest(Path(__file__)),
        "folds": list(FOLDS),
        "smoke": smoke,
        "jobs": jobs,
    }


def validate_manifest(manifest: dict, smoke: bool) -> None:
    if (
        manifest["pin"] != PIN
        or manifest["csv_sha256"] != CSV_SHA256
        or manifest["smoke"] != smoke
    ):
        raise RuntimeError("Experiment provenance mismatch")
    expected = {job_key(j) for j in grid(smoke)}
    actual = [j["id"] for j in manifest["jobs"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError(
            "Experiment contains missing, duplicate or unexpected evaluations"
        )
    if len({j["run_id"] for j in manifest["jobs"]}) != len(actual):
        raise RuntimeError("Duplicate W&B run IDs")
    for job in manifest["jobs"]:
        project = job["wandb_path"].split("/")[1]
        if job["native_arguments"] != native_arguments(
            job["model"], job["condition"], job["fold"], project, smoke
        ):
            raise RuntimeError(
                "Recorded native command differs from the agreed protocol"
            )


def result_records(experiment: Path, manifest: dict) -> list[dict]:
    records = []
    for job in manifest["jobs"]:
        work = experiment / "jobs" / job["id"]
        status = (
            read_json(work / "status.json")
            if (work / "status.json").exists()
            else {"status": "pending"}
        )
        record = {**job, "status": status["status"]}
        if status["status"] == "completed":
            exported = read_json(work / "native-summary.json")
            if exported["wandb_path"] != job["wandb_path"]:
                raise RuntimeError("Exported run ID mismatch")
            validate_native_summary(job, exported["summary"])
            record["summary"] = exported["summary"]
        records.append(record)
    return records


def learner_table(records: list[dict]) -> list[dict]:
    table = []
    for model in MODELS:
        row = {"model": model}
        complete = True
        for condition in CONDITIONS:
            group = [
                r
                for r in records
                if r["model"] == model and r["condition"] == condition
            ]
            scores = [
                float(r["summary"]["test_score"])
                for r in group
                if r["status"] == "completed"
            ]
            ready = len(scores) == len(FOLDS) and {r["fold"] for r in group} == set(
                FOLDS
            )
            complete &= ready
            row[condition] = statistics.mean(scores) if ready else None
        row["complete"] = complete
        row["joint_delta"] = (
            row["joint_frozen"] - max(row["structured"], row["text_frozen"])
            if complete
            else None
        )
        row["tar_delta"] = row["joint_tar"] - row["joint_frozen"] if complete else None
        table.append(row)
    return table


def print_table(records: list[dict]) -> None:
    print(
        "model          structured text_frozen joint_frozen joint_tar   joint_delta tar_delta"
    )
    for row in learner_table(records):
        values = [
            "pending" if row[k] is None else f"{row[k]:.8f}"
            for k in (*CONDITIONS, "joint_delta", "tar_delta")
        ]
        print(f"{row['model']:<14} " + " ".join(f"{v:>11}" for v in values))
    table = learner_table(records)
    if all(row["complete"] for row in table):
        joint = sum(row["joint_delta"] > 0 for row in table)
        tar = sum(row["tar_delta"] > 0 for row in table)
        print(
            f"Staff criteria, raw scores: Joint Signal {joint}/5; TAR Gain {tar}/5; pass={joint >= 3 and tar >= 3}"
        )
    else:
        print("Incomplete experiment: no final curation verdict.")


def print_repetition(records: list[dict], fold: int) -> None:
    print(f"Repetition {fold}, native R-squared (rounded for display only):")
    print("model          structured text_frozen joint_frozen joint_tar")
    for model in MODELS:
        cells = []
        for condition in CONDITIONS:
            record = next(
                r
                for r in records
                if r["model"] == model
                and r["condition"] == condition
                and r["fold"] == fold
            )
            cells.append(
                f"{record['summary']['test_score']:.8f}"
                if record["status"] == "completed"
                else record["status"]
            )
        print(f"{model:<14} " + " ".join(f"{cell:>11}" for cell in cells))


def write_results(records: list[dict], output: Path) -> None:
    if any(r["smoke"] for r in records):
        raise RuntimeError("Smoke-test results cannot enter the submission results CSV")
    table = {r["model"]: r for r in learner_table(records)}
    columns = [
        "model",
        "native_model",
        "condition",
        "fold",
        "split_seed",
        "status",
        "r2",
        "mse",
        "rmse",
        "native_score",
        "n_train",
        "n_test",
        "runtime_seconds",
        "mean_condition_r2",
        "joint_delta",
        "tar_delta",
        "wandb_run",
        "official_commit",
        "dataset_sha256",
        "native_metrics_json",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for job in records:
            summary = job.get("summary", {}) if job["status"] == "completed" else {}
            metrics = native_metrics(summary).get("metrics", {})
            group = table[job["model"]]
            writer.writerow(
                {
                    "model": job["model"],
                    "native_model": summary.get("model", ""),
                    "condition": job["condition"],
                    "fold": job["fold"],
                    "split_seed": 42 + job["fold"],
                    "status": job["status"],
                    **{k: metrics.get(k, "") for k in ("r2", "mse", "rmse")},
                    "native_score": summary.get("test_score", ""),
                    "n_train": summary.get("n_train", ""),
                    "n_test": summary.get("n_test", ""),
                    "runtime_seconds": summary.get("runtime", ""),
                    "mean_condition_r2": group[job["condition"]],
                    "joint_delta": group["joint_delta"],
                    "tar_delta": group["tar_delta"],
                    "wandb_run": "https://wandb.ai/" + job["wandb_path"],
                    "official_commit": PIN,
                    "dataset_sha256": CSV_SHA256,
                    "native_metrics_json": json.dumps(
                        native_metrics(summary), ensure_ascii=False
                    ),
                }
            )
    os.replace(temp, output)


def run_jobs(args, smoke=False) -> None:
    verify_source(args.source)
    verify_dependencies(args.source)
    gpus = selected_gpus(args.gpus)
    require_mount(args.runtime, args.persistent_mount)
    pre = read_json(args.runtime / "preflight.json")
    if pre["evaluator_sha256"] != digest(Path(__file__)) or not set(gpus) <= {
        g["uuid"] for g in pre["gpus"]
    }:
        raise RuntimeError("Run preflight for this script and each selected GPU first")
    if not smoke:
        passed = read_json(args.runtime / "smoke-passed.json")
        if passed["evaluator_sha256"] != digest(Path(__file__)) or passed["pin"] != PIN:
            raise RuntimeError(
                "A current isolated smoke test is required before production"
            )
    experiment = args.runtime / ("smoke" if smoke else "production")
    experiment.mkdir(parents=True, exist_ok=True)
    with lock(args.runtime / "coordinator.lock"):
        path = experiment / "manifest.json"
        manifest = read_json(path) if path.exists() else new_manifest(args, smoke)
        validate_manifest(manifest, smoke)
        if manifest["evaluator_sha256"] != digest(Path(__file__)):
            raise RuntimeError("Evaluator changed after this experiment began")
        atomic_json(path, manifest)
        pending, active = [], {}
        for job in manifest["jobs"]:
            work = experiment / "jobs" / job["id"]
            work.mkdir(parents=True, exist_ok=True)
            state = (
                read_json(work / "status.json")
                if (work / "status.json").exists()
                else {}
            )
            if state.get("status") == "completed":
                continue
            if state.get("status") == "running":
                raise RuntimeError(
                    "Previous running job needs reconciliation; never launch a duplicate"
                )
            if state.get("status") == "failed" and not args.retry_failed:
                continue
            if state:
                raise RuntimeError(
                    "Use a fresh experiment after inspecting a failed job; automatic retries are disabled"
                )
            pending.append((job, work))
        reported = set()
        try:
            while pending or active:
                free = [gpu for gpu in gpus if gpu not in active]
                for gpu in free:
                    if not pending:
                        break
                    job, work = pending.pop(0)
                    atomic_json(work / "job.json", job)
                    log = (work / "console.log").open("w", encoding="utf-8")
                    env = isolated_environment(gpu, args.source, work)
                    env.update(
                        WANDB_RUN_ID=job["run_id"],
                        WANDB_RESUME="never",
                        WANDB_TAGS=",".join(
                            (
                                manifest["experiment"],
                                job["model"],
                                job["condition"],
                                str(job["fold"]),
                            )
                        ),
                    )
                    proc = subprocess.Popen(
                        worker_command(args, "native", gpu, work),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    atomic_json(
                        work / "status.json",
                        {
                            "status": "running",
                            "pid": proc.pid,
                            "gpu_uuid": gpu,
                            "started": time.time(),
                        },
                    )
                    active[gpu] = (proc, job, work, log, time.time())
                    print(f"Started {job['id']} on {gpu}", flush=True)
                for gpu, item in list(active.items()):
                    proc, job, work, log, started = item
                    if proc.poll() is None:
                        continue
                    log.close()
                    error = None
                    if proc.returncode == 0:
                        try:
                            export_native(job, work / "native-summary.json")
                        except Exception as exc:
                            error = safe_error(exc)
                    else:
                        error = f"Native CLI exited {proc.returncode}; inspect {work / 'console.log'}"
                    atomic_json(
                        work / "status.json",
                        {
                            "status": "failed" if error else "completed",
                            "exit_code": proc.returncode,
                            "error": error,
                            "gpu_uuid": gpu,
                            "elapsed_seconds": time.time() - started,
                        },
                    )
                    print(
                        f"{'FAILED' if error else 'Finished'} {job['id']}"
                        + (f": {error}" if error else ""),
                        flush=True,
                    )
                    del active[gpu]
                    records = result_records(experiment, manifest)
                    complete = sum(r["status"] == "completed" for r in records)
                    failed = sum(r["status"] == "failed" for r in records)
                    print(
                        f"Progress: {complete}/{len(records)} complete, {failed} failed",
                        flush=True,
                    )
                    if not smoke:
                        write_results(records, args.output)
                        for fold in FOLDS:
                            fold_rows = [r for r in records if r["fold"] == fold]
                            if fold not in reported and all(
                                r["status"] in {"completed", "failed"}
                                for r in fold_rows
                            ):
                                print(
                                    f"Repetition {fold} finished; {sum(r['status'] == 'completed' for r in fold_rows)}/20 successful"
                                )
                                print_repetition(records, fold)
                                reported.add(fold)
                    estimate_remaining(records, len(gpus))
                time.sleep(2)
        finally:
            # Do not terminate healthy child jobs or a shared Pod on coordinator interruption.
            if active:
                print(
                    "Coordinator interrupted; native child jobs are left running. Reconcile before restart.",
                    flush=True,
                )
        records = result_records(experiment, manifest)
        if smoke and all(r["status"] == "completed" for r in records):
            atomic_json(
                args.runtime / "smoke-passed.json",
                {
                    "pin": PIN,
                    "evaluator_sha256": digest(Path(__file__)),
                    "experiment": manifest["experiment"],
                    "production_results_included": False,
                    "overrides": {"train_examples": 512, "e5_epochs": 1},
                },
            )
        elif not smoke:
            print_table(records)
        if any(r["status"] != "completed" for r in records):
            raise RuntimeError(
                "Experiment is incomplete; healthy jobs finished and failures were retained"
            )


def estimate_remaining(records: list[dict], gpu_count: int) -> None:
    remaining = 0.0
    for record in records:
        if record["status"] not in {"pending", "running"}:
            continue
        samples = [
            r["summary"]["runtime"]
            for r in records
            if r["status"] == "completed"
            and r["model"] == record["model"]
            and r["condition"] == record["condition"]
        ]
        if not samples:
            print(
                "ETA pending: need production timing for each remaining learner/condition.",
                flush=True,
            )
            return
        remaining += statistics.mean(samples)
    print(
        f"Timing-based remaining work / {gpu_count} GPUs: approximately {remaining / gpu_count / 60:.1f} min; scheduling tails and active-job time remain uncertain.",
        flush=True,
    )


def collect(args) -> None:
    experiment = args.runtime / "production"
    with lock(args.runtime / "coordinator.lock"):
        manifest = read_json(experiment / "manifest.json")
        validate_manifest(manifest, False)
        records = result_records(experiment, manifest)
        write_results(records, args.output)
        print_table(records)


def reconcile(args) -> None:
    """Recover exported results after a coordinator interruption without rerunning models."""
    experiment = args.runtime / ("smoke" if args.smoke else "production")
    with lock(args.runtime / "coordinator.lock"):
        manifest = read_json(experiment / "manifest.json")
        validate_manifest(manifest, args.smoke)
        for job in manifest["jobs"]:
            work = experiment / "jobs" / job["id"]
            path = work / "status.json"
            if not path.exists():
                continue
            status = read_json(path)
            if status["status"] == "completed":
                continue
            try:
                export_native(job, work / "native-summary.json")
            except Exception as exc:
                print(f"Unresolved {job['id']}: {safe_error(exc)}", flush=True)
                continue
            atomic_json(
                path,
                {
                    **status,
                    "status": "completed",
                    "recovered_from_official_wandb": True,
                    "previous_collection_error": status.get("error"),
                    "error": None,
                },
            )
            print(f"Recovered {job['id']} from official W&B export", flush=True)
        records = result_records(experiment, manifest)
        if args.smoke and all(r["status"] == "completed" for r in records):
            atomic_json(
                args.runtime / "smoke-passed.json",
                {
                    "pin": PIN,
                    "evaluator_sha256": digest(Path(__file__)),
                    "execution_evaluator_sha256": manifest["evaluator_sha256"],
                    "experiment": manifest["experiment"],
                    "reconciled_from_official_wandb": True,
                    "production_results_included": False,
                    "overrides": {"train_examples": 512, "e5_epochs": 1},
                },
            )


def start_durable(args) -> None:
    """Launch the coordinator under tmux; all scheduling remains in this file."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.session):
        raise ValueError("Use an alphanumeric tmux session name")
    if not shutil.which("tmux"):
        raise RuntimeError("Install tmux before starting a durable coordinator")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--runtime",
        str(args.runtime),
        "--source",
        str(args.source),
        "--dataset",
        str(args.dataset),
    ]
    if args.secrets:
        cmd += ["--secrets", str(args.secrets.resolve())]
    cmd += [
        "run",
        "--gpus",
        args.gpus,
        "--persistent-mount",
        str(args.persistent_mount.resolve()),
        "--project",
        args.project,
        "--output",
        str(args.output.resolve()),
    ]
    log = args.runtime / "coordinator.log"
    shell_command = shlex.join(cmd) + " >> " + shlex.quote(str(log)) + " 2>&1"
    command(["tmux", "new-session", "-d", "-s", args.session, shell_command])
    print(f"Durable session: {args.session}; log: {log}")


def status(args) -> None:
    experiment = args.runtime / "production"
    manifest = read_json(experiment / "manifest.json")
    validate_manifest(manifest, False)
    records = result_records(experiment, manifest)
    counts = {
        state: sum(r["status"] == state for r in records)
        for state in ("completed", "failed", "running", "pending")
    }
    print(json.dumps(counts, indent=2))
    print_table(records)


def backup(args) -> None:
    """Run on the local computer to pull the remote runtime; never stops a Pod."""
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", args.ssh_host):
        raise ValueError("Unexpected SSH host format")
    if not args.remote_runtime.startswith("/") or any(
        c in args.remote_runtime for c in "\n\r"
    ):
        raise ValueError("Provide an absolute remote Linux runtime path")
    destination = args.runtime / "remote-backup"
    destination.mkdir(parents=True, exist_ok=True)
    base = ["scp", "-P", str(args.ssh_port)]
    if args.identity:
        base += ["-i", str(args.identity.resolve())]
    # Pull small authoritative results first, separately from potentially large weights.
    for name in (
        "production",
        "integration.json",
        "dataset-inspection.json",
        "preflight.json",
        "coordinator.log",
    ):
        remote = f"{args.ssh_host}:{args.remote_runtime.rstrip('/')}/{name}"
        command([*base, "-r", remote, str(destination)])
    manifest = read_json(destination / "production/manifest.json")
    validate_manifest(manifest, False)
    records = result_records(destination / "production", manifest)
    atomic_json(
        destination / "backup-verification.json",
        {
            "pin": PIN,
            "experiment": manifest["experiment"],
            "completed_results": sum(r["status"] == "completed" for r in records),
            "expected_results": 100,
            "pod_stopped": False,
        },
    )
    print(f"External backup verified at {destination}; Pod remains running.")


def offers(args) -> None:
    if not os.environ.get("RUNPOD_API_KEY"):
        raise RuntimeError("Fill RUNPOD_API_KEY in the external credential file first")
    query = """query($input: GpuLowestPriceInput, $filter: GpuTypeFilter) {
      gpuTypes(input: $filter) { id displayName memoryInGb securePrice communityPrice
        lowestPrice(input: $input) { uninterruptablePrice stockStatus minMemory minVcpu
          minDisk minDownload minUpload } }
    }"""
    payload = {
        "query": query,
        "variables": {
            "filter": {"id": args.gpu_type},
            "input": {
                "gpuCount": args.gpu_count,
                "minMemoryInGb": 64,
                "minVcpuCount": 8,
                "minDisk": 100,
                "totalDisk": args.total_disk,
                "minCudaVersion": args.min_cuda,
                "supportPublicIp": True,
                "secureCloud": args.cloud == "secure",
            },
        },
    }
    request = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(safe_error(result["errors"]))
    rows = [r for r in result["data"]["gpuTypes"] if r["memoryInGb"] >= 48]
    atomic_json(
        args.runtime / f"runpod-offers-{args.gpu_count}-gpu.json",
        {
            "observed_unix_time": time.time(),
            "request": payload["variables"],
            "offers": rows,
        },
    )
    print(json.dumps(rows, indent=2))
    print("Read-only availability query; no Pod has been created.")


def completion_safe_to_stop(records: list[dict], gpu_pids: list[str]) -> bool:
    expected = {job_key(j) for j in grid()}
    return (
        len(records) == 100
        and {job_key(r) for r in records} == expected
        and all(r["status"] == "completed" for r in records)
        and not gpu_pids
    )


def monitor(args) -> None:
    """Local, model-free supervisor: verified backups and optional completion-only stop.

    This command does not launch/restart evaluations or change the Pod's source.
    It can survive a chat interruption as a separate local process.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", args.ssh_host):
        raise ValueError("Unexpected SSH host format")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", args.remote_runtime):
        raise ValueError("Unexpected remote runtime path")
    options = [
        "-i",
        str(args.identity.resolve()),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UserKnownHostsFile=" + str(args.known_hosts.resolve()),
    ]
    ssh = ["ssh", "-p", str(args.ssh_port), *options, args.ssh_host]
    destination = args.runtime / "automatic-backup"
    destination.mkdir(parents=True, exist_ok=True)
    remote_code = """from pathlib import Path
import tarfile,subprocess,json
r=Path(REMOTE_ROOT)
with tarfile.open(r/"supervisor-backup.tar.gz","w:gz",compresslevel=1) as t:
 for name in ["production/manifest.json","integration.json","dataset-inspection.json","preflight.json","smoke-passed.json","coordinator.log"]:
  p=r/name
  if p.exists():t.add(p,arcname=name)
 for name in ["status.json","job.json","gpu-identity.json","native-summary.json","console.log"]:
  for p in (r/"production/jobs").glob("*/"+name):t.add(p,arcname=str(p.relative_to(r)))
gpu=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],capture_output=True,text=True,check=True)
print(json.dumps({"gpu_pids":[s.strip() for s in gpu.stdout.splitlines() if s.strip()]}))
""".replace("REMOTE_ROOT", repr(args.remote_runtime))

    def api(query: str, variables: dict) -> dict:
        if not os.environ.get("RUNPOD_API_KEY"):
            raise RuntimeError("Missing RUNPOD_API_KEY for completion-only Pod stop")
        request = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data=json.dumps({"query": query, "variables": variables}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if result.get("errors"):
            raise RuntimeError(safe_error(result["errors"]))
        return result["data"]

    with lock(args.runtime / "local-supervisor.lock"):
        while True:
            try:
                health = json.loads(
                    command(
                        ssh + [shlex.join(["python3", "-c", remote_code])], capture=True
                    ).stdout
                )
                archive = destination / "supervisor-backup.tar.gz"
                command(
                    [
                        "scp",
                        "-P",
                        str(args.ssh_port),
                        *options,
                        args.ssh_host
                        + ":"
                        + args.remote_runtime
                        + "/supervisor-backup.tar.gz",
                        archive,
                    ]
                )
                with tarfile.open(archive) as stream:
                    members = stream.getmembers()
                    for member in members:
                        target = (destination / member.name).resolve()
                        if not target.is_relative_to(destination.resolve()) or not (
                            member.isfile() or member.isdir()
                        ):
                            raise RuntimeError("Unsafe result-backup archive member")
                    stream.extractall(destination, members=members)
                manifest = read_json(destination / "production/manifest.json")
                validate_manifest(manifest, False)
                records = result_records(destination / "production", manifest)
                write_results(records, args.output)
                counts = {
                    s: sum(r["status"] == s for r in records)
                    for s in ("completed", "running", "failed", "pending")
                }
                receipt = {
                    "timestamp_unix": time.time(),
                    "pin": PIN,
                    "counts": counts,
                    "verified_native_summaries": counts["completed"],
                    "archive_sha256": digest(archive),
                    "pod_stopped": False,
                }
                atomic_json(destination / "verification.json", receipt)
                print(json.dumps(receipt), flush=True)
                if completion_safe_to_stop(records, health["gpu_pids"]):
                    if args.stop_when_complete:
                        # Verify the exact Pod is this task's six-GPU Pod, never another resource.
                        details = api(
                            "query($input:PodFilter!){pod(input:$input){id name gpuCount desiredStatus}}",
                            {"input": {"podId": args.pod_id}},
                        )["pod"]
                        if (
                            not details
                            or details["name"] != args.pod_name
                            or details["gpuCount"] != 6
                        ):
                            raise RuntimeError("Pod identity differs; refusing to stop")
                        stopped = api(
                            "mutation($input:PodStopInput!){podStop(input:$input){id desiredStatus}}",
                            {"input": {"podId": args.pod_id}},
                        )
                        atomic_json(
                            destination / "pod-stop-receipt.json",
                            {"pod": stopped, "verification": receipt},
                        )
                        print(
                            "All 100 native results backed up and verified; completion-only Pod stop requested.",
                            flush=True,
                        )
                    return
            except Exception as exc:
                atomic_json(
                    destination / "supervisor-error.json",
                    {"timestamp_unix": time.time(), "error": safe_error(exc)},
                )
                print("Supervisor retry: " + safe_error(exc), flush=True)
            time.sleep(args.interval)


def self_test() -> None:
    flat = {
        "metrics_dict.score": 0.12345678901234567,
        "metrics_dict.metrics.r2": 0.12345678901234567,
        "metrics_dict.metrics.mse": 0.321,
        "metrics_dict.metrics.rmse": 0.5665686189686118,
    }
    expected_metrics = {
        "score": 0.12345678901234567,
        "metrics": {
            "r2": 0.12345678901234567,
            "mse": 0.321,
            "rmse": 0.5665686189686118,
        },
    }
    assert native_metrics(flat) == expected_metrics
    assert native_metrics({"metrics_dict": expected_metrics}) == expected_metrics
    summary = {
        **flat,
        "dataset": DATASET,
        "fold": 0,
        "git": PIN[:7],
        "model": "LightGBM",
        "train_examples": 10000,
        "e5_model_name": "intfloat/e5-small-v2",
        "tune_dino": False,
        "request_memory": "32G",
        "tune_e5": False,
        "pca_components": 30,
        "no_pca": False,
        "multimodal_state": "NO_TEXT",
        "n_train": 10000,
        "n_test": 1500,
        "test_score": 0.12345678901234567,
    }
    validate_native_summary(
        {"fold": 0, "model": "light", "condition": "structured", "smoke": False},
        summary,
    )
    fixture = [
        {"uuid": "GPU-second", "pci.bus_id": "00000000:20:00.0"},
        {"uuid": "GPU-first", "pci.bus_id": "00000000:03:00.0"},
    ]
    assert cuda_mask_for_uuid("GPU-first", fixture) == "0"
    assert cuda_mask_for_uuid("GPU-second", fixture) == "1"
    jobs = grid()
    assert len(jobs) == len({job_key(j) for j in jobs}) == 100
    assert sum(j["condition"] == "joint_tar" for j in jobs) == 25
    assert {j["fold"] for j in jobs} == set(range(5))
    for job in jobs:
        argv = native_arguments(job["model"], job["condition"], job["fold"], "test")
        assert "--train_examples" not in argv and "--e5_epochs" not in argv
        assert ("--tune_e5" in argv) and "--text_encoder" not in argv
    records = []
    means = {
        "structured": 0.4,
        "text_frozen": 0.5,
        "joint_frozen": 0.50000001,
        "joint_tar": 0.50000002,
    }
    for job in jobs:
        records.append(
            {
                **job,
                "status": "completed",
                "summary": {"test_score": means[job["condition"]]},
            }
        )
    assert all(
        r["joint_delta"] > 0 and r["tar_delta"] > 0 for r in learner_table(records)
    )
    records[0]["status"] = "failed"
    assert learner_table(records)[0]["joint_delta"] is None
    assert not completion_safe_to_stop(records, [])
    records[0]["status"] = "completed"
    assert completion_safe_to_stop(records, [])
    assert not completion_safe_to_stop(records, ["1234"])
    assert not completion_safe_to_stop(records[:-1], [])
    records[0]["status"] = "failed"
    manifest = {
        "pin": PIN,
        "csv_sha256": CSV_SHA256,
        "smoke": False,
        "jobs": [
            {
                **j,
                "id": job_key(j),
                "run_id": job_key(j),
                "wandb_path": "test/project/" + job_key(j),
                "native_arguments": native_arguments(
                    j["model"], j["condition"], j["fold"], "project"
                ),
            }
            for j in jobs
        ],
    }
    validate_manifest(manifest, False)
    manifest["jobs"][0]["native_arguments"] += ["--e5_epochs", "1"]
    try:
        validate_manifest(manifest, False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("A shortened production command was accepted")
    with tempfile.TemporaryDirectory(prefix="july-runner-test-") as directory:
        path = Path(directory) / "coordinator.lock"
        with lock(path):
            try:
                with lock(path):
                    raise AssertionError("Two coordinators acquired the same lock")
            except RuntimeError:
                pass
        assert not path.exists()
        result_path = Path(directory) / "partial.csv"
        for row in records:
            row.update(smoke=False, wandb_path="test/project/" + job_key(row))
            row["summary"]["metrics_dict"] = {
                "score": row["summary"]["test_score"],
                "metrics": {
                    "r2": row["summary"]["test_score"],
                    "mse": 0.12345678901234567,
                    "rmse": 0.3513641828820144,
                },
            }
        write_results(records, result_path)
        with result_path.open(newline="", encoding="utf-8") as stream:
            exported = list(csv.DictReader(stream))
        assert len(exported) == 100 and exported[0]["r2"] == ""
        assert float(exported[1]["r2"]) == records[1]["summary"]["test_score"]
        assert float(exported[1]["mse"]) == 0.12345678901234567
    print(
        "Passed: unique 100-job grid, 25 TAR fits, unchanged production commands, raw-score deltas, partial-result blanks, full-precision CSV round trip and single-coordinator lock."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path.home() / ".local/share/linkedin-july-native",
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/dataset.csv",
    )
    parser.add_argument("--secrets", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in (
        "prepare",
        "inspect",
        "self-test",
        "verify-deps",
        "status",
        "auth-check",
    ):
        sub.add_parser(name)
    setup = sub.add_parser("install")
    setup.add_argument("--python", default="python3.11")
    setup.add_argument(
        "--venv-dir",
        type=Path,
        help="Optional local SSD environment; runtime/venv links to it",
    )
    setup.add_argument(
        "--system-tools", action="store_true", help="Install tmux/git using apt as root"
    )
    for name in ("preflight", "smoke", "run", "start"):
        action = sub.add_parser(name)
        action.add_argument(
            "--gpus",
            required=True,
            help="Comma-separated allocated indices or GPU UUIDs",
        )
        action.add_argument("--persistent-mount", type=Path, required=True)
        action.add_argument("--project")
        action.add_argument(
            "--output",
            type=Path,
            default=Path(__file__).resolve().parents[1] / "evaluation_results.csv",
        )
        action.add_argument(
            "--retry-failed", action="store_true", help=argparse.SUPPRESS
        )
        if name == "start":
            action.add_argument("--session", default="linkedin-july-native")
    action = sub.add_parser("collect")
    action.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation_results.csv",
    )
    action = sub.add_parser("reconcile")
    action.add_argument("--smoke", action="store_true")
    action = sub.add_parser("backup")
    action.add_argument("--ssh-host", required=True)
    action.add_argument("--ssh-port", type=int, default=22)
    action.add_argument("--identity", type=Path)
    action.add_argument("--remote-runtime", required=True)
    action = sub.add_parser("monitor")
    action.add_argument("--ssh-host", required=True)
    action.add_argument("--ssh-port", type=int, default=22)
    action.add_argument("--identity", type=Path, required=True)
    action.add_argument("--known-hosts", type=Path, required=True)
    action.add_argument("--remote-runtime", required=True)
    action.add_argument("--pod-id", required=True)
    action.add_argument("--pod-name", default="linkedin-july-native-20261026")
    action.add_argument("--stop-when-complete", action="store_true")
    action.add_argument("--interval", type=int, default=60)
    action.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation_results.csv",
    )
    action = sub.add_parser("offers")
    action.add_argument("--gpu-count", type=int, default=6)
    action.add_argument("--gpu-type", default="NVIDIA A40")
    action.add_argument("--cloud", choices=("secure", "community"), default="secure")
    action.add_argument(
        "--min-cuda",
        default="12.8",
        help="Host driver compatibility for the proposed CUDA 12.8 image",
    )
    action.add_argument("--total-disk", type=int, default=200)
    action = sub.add_parser("plan")
    action.add_argument("--project", default="linkedin-salary-july-native")
    action = sub.add_parser("_worker", help=argparse.SUPPRESS)
    action.add_argument("--kind", choices=("native", "preflight"), required=True)
    action.add_argument("--gpu", required=True)
    action.add_argument("--work", type=Path, required=True)
    action.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args()
    args.runtime = args.runtime.resolve()
    args.source = (args.source or args.runtime / "MulTaBench").resolve()
    args.dataset = args.dataset.resolve()
    args.runtime.mkdir(parents=True, exist_ok=True)
    if args.runtime.is_relative_to(Path(__file__).resolve().parents[1]):
        raise RuntimeError(
            "Runtime artifacts must stay outside the final submission directory"
        )
    load_secrets(args.secrets)
    if hasattr(args, "project") and args.project is None:
        args.project = os.environ.get("WANDB_PROJECT", "linkedin-salary-july-native")
    if args.action == "prepare":
        prepare(args)
    elif args.action == "install":
        install(args)
    elif args.action == "inspect":
        inspect_dataset(args)
    elif args.action == "verify-deps":
        print(json.dumps(verify_dependencies(args.source), indent=2))
    elif args.action == "auth-check":
        check_credentials(args)
    elif args.action == "preflight":
        preflight(args)
    elif args.action == "smoke":
        run_jobs(args, smoke=True)
    elif args.action == "run":
        run_jobs(args)
    elif args.action == "collect":
        collect(args)
    elif args.action == "reconcile":
        reconcile(args)
    elif args.action == "start":
        start_durable(args)
    elif args.action == "status":
        status(args)
    elif args.action == "backup":
        backup(args)
    elif args.action == "monitor":
        monitor(args)
    elif args.action == "offers":
        offers(args)
    elif args.action == "self-test":
        self_test()
    elif args.action == "_worker":
        worker(args)
    elif args.action == "plan":
        for job in grid():
            print(
                shlex.join(
                    [
                        "python",
                        "benchmark.py",
                        *native_arguments(
                            job["model"], job["condition"], job["fold"], args.project
                        ),
                    ]
                )
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"ERROR ({type(error).__name__}): {safe_error(error)}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
