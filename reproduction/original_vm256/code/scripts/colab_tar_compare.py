"""Quota-conscious, resumable E5 TAR pilot using the project's existing runner.

Default: only joint_tar, one fold, CatBoost + LightGBM. The notebook supplies
the prepared 15K-row CSVs. TF-IDF baselines are context, not a frozen-E5 TAR control.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


@contextlib.contextmanager
def persistent_tar(cache: Path, *, epochs: int, micro_batch: int):
    """Persist epoch checkpoints and the completed adapter; restore upstream functions on exit."""
    from multabench.baselines.preprocessing import text_embeddings
    from multabench.e5.constants import E5_LORA
    from multabench.finetune import lora
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint

    if E5_LORA.batch_size % micro_batch:
        raise ValueError("micro_batch must divide the official effective batch size")
    original_finetune = text_embeddings.finetune_e5
    original_arguments, original_trainer = lora.TrainingArguments, lora.Trainer
    original_workers = lora.DATALOADER_WORKERS
    training_dir = cache / "trainer"
    ready = cache / "complete.json"

    def arguments(*args, **kwargs):
        kwargs.update(output_dir=str(training_dir), per_device_train_batch_size=micro_batch,
                      per_device_eval_batch_size=micro_batch,
                      gradient_accumulation_steps=E5_LORA.batch_size // micro_batch)
        # Keep full precision and the official LR/rank/layers/patience/effective batch.
        return original_arguments(*args, **kwargs)

    class ResumableTrainer(original_trainer):
        def train(self, *args, resume_from_checkpoint=None, **kwargs):
            if resume_from_checkpoint is None and training_dir.exists():
                resume_from_checkpoint = get_last_checkpoint(str(training_dir))
                if resume_from_checkpoint:
                    print(f"Resuming the saved training epoch: {resume_from_checkpoint}", flush=True)
            return super().train(*args, resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    def finetune(x, y, columns, *, is_cls, device, model_name, **kwargs):
        if ready.exists():
            saved = json.loads(ready.read_text(encoding="utf-8"))
            if saved["columns"] != sorted(columns) or saved["model_name"] != model_name:
                raise ValueError("Saved adapter metadata does not match this run")
            print(f"Reusing completed TAR adapter: {cache}", flush=True)
            backbone = PeftModel.from_pretrained(AutoModel.from_pretrained(model_name), str(cache / "adapter"))
            backbone.to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(str(cache / "tokenizer"))
            return backbone, tokenizer, {**saved["stats"], "adapter_reused": True}
        cache.mkdir(parents=True, exist_ok=True)
        training_args = replace(E5_LORA, epochs=epochs)
        backbone, tokenizer, stats = original_finetune(x, y, columns, is_cls=is_cls, device=device,
                                                      model_name=model_name, args=training_args)
        backbone.save_pretrained(str(cache / "adapter"), safe_serialization=True)
        tokenizer.save_pretrained(str(cache / "tokenizer"))
        write_json(ready, {"columns": sorted(columns), "model_name": model_name, "stats": stats})
        return backbone, tokenizer, stats

    text_embeddings.finetune_e5 = finetune
    lora.TrainingArguments, lora.Trainer, lora.DATALOADER_WORKERS = arguments, ResumableTrainer, 0
    try:
        yield
    finally:
        text_embeddings.finetune_e5 = original_finetune
        lora.TrainingArguments, lora.Trainer = original_arguments, original_trainer
        lora.DATALOADER_WORKERS = original_workers


def build_summary(results_root: Path):
    """Use the same completed folds for every comparison and keep encoder changes explicit."""
    import pandas as pd
    from tt_curation import REPO_ROOT

    rows = []
    for manifest_path in sorted(results_root.glob("*/manifest.json")):
        dataset_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_paths = sorted((dataset_dir / "runs").glob("*.json"))
        runs = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in run_paths])
        if runs.empty:
            continue
        baseline_paths = sorted((REPO_ROOT / "results" / f"{manifest['dataset']}_screen_tfidf" / "runs").glob("*.json"))
        baseline = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in baseline_paths])
        metric = "auroc" if manifest["task"] == "classification" else "r2"
        ok = runs[runs.status == "ok"]
        for model, group in ok[ok.condition == "joint_tar"].groupby("model"):
            folds = sorted(group.fold.unique().tolist())
            row = {"dataset": manifest["dataset"], "experiment": dataset_dir.name, "learner": model, "metric": metric,
                   "folds": ",".join(map(str, folds)), "n_folds": len(folds), "max_epochs": manifest["max_epochs"],
                   "E5_TAR_joint": group[metric].mean()}
            if not baseline.empty:
                matched = baseline[(baseline.status == "ok") & (baseline.model == model) &
                                   (baseline.condition == "joint_frozen") & baseline.fold.isin(folds)]
                if set(matched.fold) == set(folds):
                    row["TFIDF_joint_same_folds"] = matched[metric].mean()
                    row["TAR_minus_TFIDF_not_isolated_TAR_gain"] = row["E5_TAR_joint"] - matched[metric].mean()
            frozen = ok[(ok.model == model) & (ok.condition == "joint_frozen") & ok.fold.isin(folds)]
            if set(frozen.fold) == set(folds):
                row["E5_frozen_joint_same_folds"] = frozen[metric].mean()
                row["isolated_TAR_gain"] = row["E5_TAR_joint"] - frozen[metric].mean()
            rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        summary = pd.DataFrame(columns=["dataset", "experiment", "learner", "metric", "folds", "E5_TAR_joint"])
    results_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_root / "tar_comparison.csv", index=False)
    if not summary.empty:
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print("TF-IDF versus E5-TAR changes both the encoder and tuning; only E5-frozen versus E5-TAR isolates TAR gain.")
    return summary


def main():
    # Colab exports its inline backend into child environments where that module is unavailable.
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--models", default="light,cat")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--micro_batch", type=int, choices=[8, 16, 32], default=16)
    parser.add_argument("--include_frozen", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        build_summary(Path(args.results_root))
        return
    if not args.config:
        parser.error("--config is required unless --summary_only is used")
    if args.max_epochs < 1:
        parser.error("--max_epochs must be positive")
    models = args.models.split(",")
    folds = list(dict.fromkeys(int(f) for f in args.folds.split(",")))
    if not models or set(models) - {"light", "cat"} or not folds or set(folds) - set(range(6)):
        parser.error("models must be light/cat; folds must be in 0..5")

    import torch
    from tt_curation import MULTABENCH_COMMIT
    from tt_curation.conditions import Condition
    from tt_curation.criteria import evaluation_report, load_runs
    from tt_curation.data import describe, load_dataset
    from tt_curation.runner import run_all
    from tt_curation.spec import load_spec

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU: choose a GPU runtime before starting TAR. No CPU fallback.")
    spec = load_spec(args.config)
    data = load_dataset(spec)
    if len(data.y) != 15_000:
        raise ValueError(f"Expected the prepared 15K sample; got {len(data.y)} rows")
    config_path = Path(args.config)
    settings = {"dataset": spec.name, "task": spec.task, "csv_sha256": digest(spec.csv),
                "config_sha256": digest(config_path), "helper_sha256": digest(Path(__file__)),
                "runner_sha256": digest(Path(sys.modules[run_all.__module__].__file__)),
                "multabench_commit": MULTABENCH_COMMIT, "max_epochs": args.max_epochs,
                "micro_batch": args.micro_batch, "effective_batch": 256, "precision": "float32",
                "size": "10k", "packages": {name: version(name) for name in
                  ["torch", "transformers", "peft", "multabench", "catboost", "lightgbm", "scikit-learn"]}}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    # Different datasets/settings never overwrite or silently reuse each other's results.
    dataset_dir = Path(args.results_root) / f"{spec.name}_{signature}"
    write_json(dataset_dir / "manifest.json", settings)
    print("\n".join(describe(data)), flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}; result directory: {dataset_dir}", flush=True)
    print(f"Pilot: {args.max_epochs} epoch cap; official cap is 50. Early stopping remains enabled.", flush=True)
    for fold in folds:
        cache = dataset_dir / "checkpoints" / f"fold{fold}"
        with persistent_tar(cache, epochs=args.max_epochs, micro_batch=args.micro_batch):
            run_all(data, dataset_dir, models=models, conditions=[Condition.JOINT_TAR], folds=[fold], device="cuda")
        if args.include_frozen:
            run_all(data, dataset_dir, models=models, conditions=[Condition.JOINT_FROZEN], folds=[fold], device="cuda")
        runs = load_runs(dataset_dir)
        evaluation_report(runs).to_csv(dataset_dir / "evaluation_report.csv", index=False)
        build_summary(Path(args.results_root))
        expected_conditions = ["joint_tar", "joint_frozen"] if args.include_frozen else ["joint_tar"]
        selected = runs[runs.fold.eq(fold) & runs.short_name.isin(models) & runs.condition.isin(expected_conditions)]
        if len(selected) != len(models) * len(expected_conditions) or not selected.status.eq("ok").all():
            raise RuntimeError(f"A learner failed; inspect JSONs under {dataset_dir / 'runs'}, then rerun to retry.")


if __name__ == "__main__":
    main()
