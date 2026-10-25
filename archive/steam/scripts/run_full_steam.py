"""Run all 120 Steam evaluations with full-budget, resumable TAR on one GPU.

    python -u scripts/run_full_steam.py

Uses the existing official wrappers/splits and the tested Colab checkpoint helper.
The default uses the official batch of 256 without gradient accumulation.
Smaller execution micro-batches are an explicit opt-in, preserving effective batch 256.
No Colab pilot epoch cap or learner restriction is used.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

from colab_tar_compare import persistent_tar, write_json

EXPECTED_DATA_SHA256_LF = "c9a9594066403fda72ee765d17ebb9461ed7bdb461d82161aaa745e294e631f5"


def text_digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def audit_runs(runs_dir, folds, models, conditions):
    """Audit the expected Cartesian product, not just the CLI's exit status or file count."""
    failures = []
    for fold in folds:
        for model in models:
            for condition in conditions:
                path = Path(runs_dir) / f"{model}_{condition}_10k_fold{fold}.json"
                if not path.exists():
                    failures.append(f"{path.name}: missing")
                    continue
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    failures.append(f"{path.name}: unreadable ({type(exc).__name__})")
                    continue
                if result.get("status") != "ok":
                    failures.append(f"{path.name}: {result.get('error', result.get('status'))}")
                elif (not isinstance(result.get("test_score"), (int, float)) or
                      not math.isfinite(result["test_score"]) or not 0 <= result["test_score"] <= 1 or
                      result.get("dataset") != "steam_recommended" or result.get("short_name") != model or
                      result.get("condition") != str(condition) or result.get("fold") != fold or
                      result.get("n_train") != 10_000 or result.get("n_test") != 5_000 or
                      result.get("text_encoder") != ("e5-small-tar" if str(condition) == "joint_tar"
                                                     else None if str(condition) == "structured" else "e5-small")):
                    failures.append(f"{path.name}: unexpected experiment metadata")
    return failures


def main():
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results/steam_full_e5")
    parser.add_argument("--folds", default="0,1,2,3,4,5", help="Default: all six. A subset resumes selected folds.")
    parser.add_argument("--micro_batch", type=int, choices=[8, 16, 32, 256], default=256,
                        help="Default: official batch 256. Smaller micro-batches require an approved execution adjustment.")
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()
    folds = list(dict.fromkeys(int(f) for f in args.folds.split(",")))
    if not folds or set(folds) - set(range(6)):
        parser.error("folds must be selected from 0..5")

    from dotenv import load_dotenv
    from tt_curation import MULTABENCH_COMMIT, REPO_ROOT
    # Load before importing MulTaBench: it reads HF_TOKEN into its constants at import time.
    load_dotenv(REPO_ROOT / ".env")
    from tt_curation.conditions import Condition, MODELS
    from tt_curation.spec import load_spec

    conditions, models = list(Condition), list(MODELS)
    output = Path(args.results_root).resolve() / "steam_recommended"
    if args.audit_only:
        problems = audit_runs(output / "runs", range(6), models, conditions)
        if problems:
            print("\n".join(problems), flush=True)
            raise SystemExit(f"Incomplete full run: {len(problems)} of 120 expected evaluations failed/missing.")
        print("PASS: all 120 expected Steam evaluations completed successfully with matching settings.")
        return

    import torch
    from tt_curation.__main__ import report
    from tt_curation.criteria import evaluation_report, load_runs
    from tt_curation.data import describe, load_dataset
    from tt_curation.runner import run_all

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available; the full GPU run has not started.")
    config = REPO_ROOT / "configs/steam_recommended.yaml"
    spec = load_spec(config)
    if text_digest(spec.csv) != EXPECTED_DATA_SHA256_LF:
        raise ValueError("Steam CSV differs from the agreed 15K sample (line endings normalized). Do not resample.")
    data = load_dataset(spec)
    if len(data.y) != 15_000 or data.y.value_counts().to_dict() != {"yes": 13_405, "no": 1_595}:
        raise ValueError("Steam row count or target counts differ from the agreed sample")
    code_paths = [*sorted((REPO_ROOT / "src/tt_curation").glob("*.py")), Path(__file__),
                  Path(__file__).with_name("colab_tar_compare.py")]
    settings = {"dataset": spec.name, "csv_sha256_lf": text_digest(spec.csv), "config_sha256_lf": text_digest(config),
                "multabench_commit": MULTABENCH_COMMIT, "models": models,
                "conditions": [str(c) for c in conditions], "folds": list(range(6)), "expected_evaluations": 120,
                "size": "10k", "frozen_encoder": "e5-small", "tar_max_epochs": 50, "micro_batch": args.micro_batch,
                "effective_batch": 256, "precision": "float32",
                "code": {p.relative_to(REPO_ROOT).as_posix(): text_digest(p) for p in code_paths},
                "packages": {name: version(name) for name in ["multabench", "torch", "transformers", "peft",
                                                             "catboost", "lightgbm", "pytabkit", "tabpfn",
                                                             "scikit-learn", "pandas", "numpy", "skrub"]}}
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != settings:
            raise ValueError("Existing full-run settings differ. Use a separate results root; do not mix experiments.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Results exist without a matching manifest. Verify them or choose a fresh results root.")
    else:
        write_json(manifest, settings)
    print("\n".join(describe(data)), flush=True)
    print(f"Full run: 5 learners x 4 conditions x 6 folds = 120 evaluations; selected folds: {folds}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}; results: {output}", flush=True)
    print("TAR: max 50 epochs with early stopping; effective batch 256; independent adapter per fold.", flush=True)
    for fold in folds:
        with persistent_tar(output / "checkpoints" / f"fold{fold}", epochs=50, micro_batch=args.micro_batch):
            run_all(data, output, models=models, conditions=conditions, folds=[fold], size="10k", device="cuda",
                    frozen_encoder="e5-small")
        evaluation_report(load_runs(output)).to_csv(output / "evaluation_report.csv", index=False)
        problems = audit_runs(output / "runs", [fold], models, conditions)
        if problems:
            print("\n".join(problems), flush=True)
            raise RuntimeError("Fold incomplete. Completed work is saved; fix failures and rerun to resume.")
    problems = audit_runs(output / "runs", range(6), models, conditions)
    if problems:
        print(f"Selected folds completed. Full experiment still has {len(problems)} missing/failed evaluations.")
    else:
        report(output)
        print("COMPLETE: all 120 expected Steam evaluations succeeded. Final reports have been generated.")


if __name__ == "__main__":
    main()
