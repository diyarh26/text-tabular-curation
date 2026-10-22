"""Official-budget LinkedIn fold worker; parallel workers own separate fold directories.

Run --prepare_only once before workers, then --folds 0,2,4 and --folds 1,3,5
with different CUDA_VISIBLE_DEVICES. Run --report_only after both workers stop.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

from colab_tar_compare import persistent_tar, write_json

DATA_SHA256_LF = "8880f84563535c60eda01be5b52010dea6f80e318f228ade00166d7cb66d57fa"
MODELS = ["tabm", "cat", "light", "tabpfnv2", "tabpfnv2p5"]
CONDITIONS = ["structured", "unstructured", "joint_frozen", "joint_tar"]


def text_digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fold_dir(output, fold):
    return Path(output) / "folds" / f"fold{fold}"


@contextlib.contextmanager
def full_tar(cache, micro_batch=256):
    """Keep the official budget and restore early-stopping patience across sessions too."""
    from multabench.finetune import lora
    with persistent_tar(cache, epochs=50, micro_batch=micro_batch):
        original_arguments = lora.TrainingArguments

        def arguments(*args, **kwargs):
            kwargs["restore_callback_states_from_checkpoint"] = True
            return original_arguments(*args, **kwargs)

        lora.TrainingArguments = arguments
        try:
            yield
        finally:
            lora.TrainingArguments = original_arguments


def audit_runs(output, folds=range(6)):
    """Require every expected identity and finite regression metrics; R2 can be negative."""
    problems = []
    for fold in folds:
        for model in MODELS:
            for condition in CONDITIONS:
                path = fold_dir(output, fold) / "runs" / f"{model}_{condition}_10k_fold{fold}.json"
                label = f"fold{fold}/{path.name}"
                if not path.exists():
                    problems.append(f"{label}: missing")
                    continue
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                    finite = all(isinstance(result.get(k), (float, int)) and math.isfinite(result[k])
                                 for k in ("test_score", "test_error", "r2"))
                    encoder = "e5-small-tar" if condition == "joint_tar" else (
                        None if condition == "structured" else "e5-small")
                    valid = (finite and result["test_error"] >= 0 and result["r2"] <= 1 + 1e-8 and
                             math.isclose(result["test_score"], -result["test_error"], abs_tol=1e-8) and
                             result.get("dataset") == "linkedin_salary" and result.get("short_name") == model and
                             result.get("condition") == condition and result.get("fold") == fold and
                             result.get("n_train") == 10_000 and result.get("n_test") == 5_000 and
                             result.get("text_encoder") == encoder)
                    if result.get("status") != "ok":
                        problems.append(f"{label}: {result.get('error', result.get('status'))}")
                    elif not valid:
                        problems.append(f"{label}: unexpected metadata or regression metrics")
                except (ValueError, OSError, TypeError) as exc:
                    problems.append(f"{label}: unreadable ({type(exc).__name__})")
    return problems


def collect_runs(output):
    import pandas as pd
    rows = []
    for fold in range(6):
        for model in MODELS:
            for condition in CONDITIONS:
                path = fold_dir(output, fold) / "runs" / f"{model}_{condition}_10k_fold{fold}.json"
                if path.exists():
                    try:
                        rows.append(json.loads(path.read_text(encoding="utf-8")))
                    except ValueError:
                        pass  # Audit lists interrupted JSON writes as unreadable.
    return pd.DataFrame(rows)


def make_report(output):
    from tt_curation.criteria import curation_verdict, evaluation_report, summary_lines
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    problems = audit_runs(output)
    runs = collect_runs(output)
    if not runs.empty and (runs.status == "ok").any():
        evaluation_report(runs).to_csv(output / "evaluation_report.csv", index=False)
    if problems:
        text = f"INCOMPLETE: {len(problems)} of 120 expected evaluations missing/failed/invalid.\n"
        text += "No final curation verdict. Resume the same experiment.\n" + "\n".join(problems)
        # A resumed partial report must never retain a stale final verdict.
        (output / "curation_verdict.csv").unlink(missing_ok=True)
    else:
        verdict = curation_verdict(runs)
        verdict.to_csv(output / "curation_verdict.csv", index=False)
        text = "COMPLETE: all 120 evaluations passed the audit.\n" + "\n".join(summary_lines(verdict, runs))
    (output / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return not problems


def main():
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--micro_batch", type=int, choices=[16, 32, 256], default=256,
                        help="Official default 256; smaller micro-batches require an approved memory adjustment.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare_only", action="store_true")
    mode.add_argument("--report_only", action="store_true")
    mode.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()
    folds = list(dict.fromkeys(int(f) for f in args.folds.split(",")))
    if not folds or set(folds) - set(range(6)):
        parser.error("folds must be selected from 0..5")
    output = Path(args.results_root).resolve() / "linkedin_salary"
    if args.report_only:
        make_report(output)
        return
    if args.audit_only:
        problems = audit_runs(output)
        if problems:
            raise SystemExit("\n".join(problems))
        print("PASS: all 120 LinkedIn evaluations audited.")
        return

    from dotenv import load_dotenv
    from tt_curation import MULTABENCH_COMMIT, REPO_ROOT
    load_dotenv(REPO_ROOT / ".env")
    from tt_curation.conditions import Condition, MODELS as upstream_models
    from tt_curation.data import load_dataset, describe
    from tt_curation.spec import load_spec

    assert set(MODELS) == set(upstream_models) and CONDITIONS == [str(c) for c in Condition]
    config = REPO_ROOT / "configs/linkedin_salary.yaml"
    spec = load_spec(config)
    if text_digest(spec.csv) != DATA_SHA256_LF:
        raise ValueError("LinkedIn CSV differs from the agreed 15K sample. Do not resample.")
    data = load_dataset(spec)
    if len(data.y) != 15_000 or spec.task != "regression":
        raise ValueError("Expected 15K LinkedIn regression rows")
    paths = [*sorted((REPO_ROOT / "src/tt_curation").glob("*.py")), Path(__file__).resolve(),
             Path(__file__).resolve().with_name("colab_tar_compare.py")]
    settings = dict(dataset=spec.name, csv_sha256_lf=DATA_SHA256_LF, config_sha256_lf=text_digest(config),
                    multabench_commit=MULTABENCH_COMMIT, models=MODELS, conditions=CONDITIONS,
                    folds=list(range(6)), expected_evaluations=120, size="10k", frozen_encoder="e5-small",
                    source_revision=(REPO_ROOT / "SOURCE_REVISION").read_text().strip()
                        if (REPO_ROOT / "SOURCE_REVISION").exists() else None,
                    tar_max_epochs=50, micro_batch=args.micro_batch, effective_batch=256, precision="float32",
                    restore_early_stopping_state=True,
                    code={p.relative_to(REPO_ROOT).as_posix(): text_digest(p) for p in paths},
                    packages={n: version(n) for n in ["multabench", "torch", "transformers", "peft", "catboost",
                                                    "lightgbm", "pytabkit", "tabpfn", "scikit-learn", "pandas",
                                                    "numpy", "skrub"]})
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != settings:
            raise ValueError("Saved experiment settings differ. Do not mix results or checkpoints.")
    elif args.prepare_only and (not output.exists() or not any(output.iterdir())):
        write_json(manifest, settings)
    else:
        raise ValueError("Initialize an empty output using --prepare_only before launching workers.")
    print("\n".join(describe(data)), flush=True)
    if args.prepare_only:
        print("Manifest verified. Workers can now start in separate fold directories.", flush=True)
        return

    import torch
    from tt_curation.runner import run_all
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see exactly one GPU. Set CUDA_VISIBLE_DEVICES before starting Python.")
    print(f"Assigned GPU: {torch.cuda.get_device_name(0)}; folds {folds}; micro-batch {args.micro_batch}, "
          "effective batch 256, cap 50.", flush=True)
    for fold in folds:
        destination = fold_dir(output, fold)
        with full_tar(destination / "checkpoints", micro_batch=args.micro_batch):
            run_all(data, destination, models=MODELS, conditions=list(Condition), folds=[fold],
                    size="10k", device="cuda", frozen_encoder="e5-small")
        problems = audit_runs(output, [fold])
        if problems:
            raise RuntimeError("Fold incomplete; saved work can be resumed.\n" + "\n".join(problems))
        print(f"Fold {fold}: all 20 evaluations audited successfully.", flush=True)


if __name__ == "__main__":
    main()
