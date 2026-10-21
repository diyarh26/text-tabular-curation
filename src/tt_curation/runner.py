"""Run MulTaBench's learners on our own dataset, in every condition and fold, one JSON per run.

MulTaBench's benchmark.py only runs the datasets it hosts, so we call one level below it: the same splits
(multabench.benchmark.load.split_dataset), the same model wrappers and preprocessing, and the same metric. A wrapper
given dataset=None reads no cache and embeds the text itself, frozen or LoRA-tuned on the fold's training rows.

As in MulTaBench's TAR cache, E5 is tuned once per fold, on the fold's training rows, and every learner reads that
encoder; `shared_tar_encoder` makes the wrappers reuse it instead of each tuning its own.
"""
import contextlib
import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from tt_curation import MULTABENCH_COMMIT, REPO_ROOT
from tt_curation.conditions import CONDITION_ENCODER, Condition, MODELS, condition_features
from tt_curation.data import LoadedDataset


def run_name(model: str, condition: Condition, size: str, fold: int) -> str:
    return f"{model}_{condition}_{size}_fold{fold}"


def run_all(data: LoadedDataset, results_dir: Path, models: Iterable[str], conditions: Iterable[Condition],
            folds: Iterable[int], size: str = "10k", device: Optional[str] = None, overwrite: bool = False,
            frozen_encoder: str = "e5-small"):
    import torch
    from multabench.benchmark.load import split_dataset
    from multabench.datasets.objects import MultimodalDataset, SupervisedTask
    from multabench.utils.devices import get_device

    device = get_device(device)
    models, conditions = list(models), list(conditions)
    if Condition.JOINT_TAR in conditions and device.type == "cpu":
        raise RuntimeError("joint_tar LoRA-tunes E5, which needs a GPU; run it on the GPU machine, or leave it out "
                           "with --conditions structured,unstructured,joint_frozen (or --preset screen)")
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if data.spec.is_cls:
        task = SupervisedTask.BINARY if data.y.nunique() == 2 else SupervisedTask.MULTICLASS
    else:
        task = SupervisedTask.REGRESSION
    dataset = MultimodalDataset(x=data.x, y=data.y, task_type=task, dataset_id=None)
    print(f"🚀 {data.spec.name}: {len(models)} models x {len(conditions)} conditions x folds {list(folds)} on {device}")

    for fold in folds:
        split = split_dataset(dataset, fold=fold, size=size)
        for condition in conditions:
            pending = [m for m in models
                       if overwrite or not (runs_dir / f"{run_name(m, condition, size, fold)}.json").exists()]
            if not pending:
                continue
            with shared_tar_encoder(split, data.text_columns, data.spec.is_cls, device,
                                    enabled=condition == Condition.JOINT_TAR) as tar_stats:
                for model in pending:
                    result = run_one(model, split, task, data, condition, fold, size, device, frozen_encoder)
                    result["tar_finetune"] = tar_stats or None
                    path = runs_dir / f"{run_name(model, condition, size, fold)}.json"
                    path.write_text(json.dumps(result, indent=2, default=str))
                    status = (f"score {result['test_score']:.4f}" if result["status"] == "ok"
                              else f"ERROR {result['error']}")
                    print(f"  fold {fold} {condition:<13} {model:<11} {status} ({result['runtime_s']:.0f}s)")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_one(model: str, split, task, data: LoadedDataset, condition: Condition, fold: int, size: str,
            device, frozen_encoder: str = "e5-small") -> Dict:
    """frozen_encoder replaces e5-small in the frozen conditions; "tfidf" makes a cheap CPU screen of a candidate."""
    from multabench.baselines.preprocessing.target import transform_preprocess_y
    from multabench.baselines.training.metrics import calculate_metric

    start = time.time()
    encoder = CONDITION_ENCODER[condition]
    if encoder == "e5-small":
        encoder = frozen_encoder
    x_train = condition_features(split.x_train, data.text_columns, condition)
    x_test = condition_features(split.x_test, data.text_columns, condition)
    result = {
        "dataset": data.spec.name, "model": MODELS[model], "short_name": model, "condition": str(condition),
        "text_encoder": encoder, "fold": fold, "size": size, "task": str(task.name).lower(),
        "n_train": len(x_train), "n_test": len(x_test), "n_features": x_train.shape[1],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": str(device),
        "multabench_commit": MULTABENCH_COMMIT, "git": _git_commit(),
    }
    try:
        model_cls = _model_class(model)
        wrapper = model_cls(problem_type=task, device=device, dataset=None, text_encoder=encoder, fold=fold,
                            size=size)
        wrapper.fit(x_train, split.y_train)
        y_true = transform_preprocess_y(y=split.y_test.copy(), scaler=wrapper.target_transformer)
        y_pred = wrapper.predict(x_test.copy())
        metrics = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=wrapper.d_output)
        result.update({k.value if hasattr(k, "value") else k: v for k, v in metrics.items()})
        result.update(_course_metric(np.asarray(y_true), np.asarray(y_pred), wrapper.d_output))
        result["status"] = "ok"
    except Exception as e:
        traceback.print_exc()
        result.update({"status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
    result["runtime_s"] = time.time() - start
    return result


def _course_metric(y_true: np.ndarray, y_pred: np.ndarray, d_output: int) -> Dict:
    """AUROC for classification, R^2 for regression, as the course reports them; MulTaBench's own metric is the
    error 1-AUROC (binary), log loss (multiclass) or RMSE (regression)."""
    from sklearn.metrics import r2_score, roc_auc_score
    if d_output == 1:
        return {"r2": float(r2_score(y_true, y_pred))}
    if d_output == 2:
        return {"auroc": float(roc_auc_score(y_true, y_pred))}
    return {"auroc": float(roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro",
                                         labels=list(range(y_pred.shape[1]))))}


@contextlib.contextmanager
def shared_tar_encoder(split, text_columns: List[str], is_cls: bool, device, enabled: bool):
    """Tune E5 once on the fold's training rows and make every wrapper's finetune_e5 return it."""
    if not enabled:
        yield {}
        return
    from multabench.baselines.preprocessing import text_embeddings
    from multabench.baselines.preprocessing.feature_types import transform_feature_types
    from multabench.e5.constants import E5_SMALL_V2
    from multabench.finetune import lora

    if sys.platform == "win32":
        # Windows spawns DataLoader workers, which can't pickle MulTaBench's local featurize function.
        lora.DATALOADER_WORKERS = 0
    columns = sorted(text_columns)
    # The texts exactly as the wrappers embed them, after their missing-text fill.
    texts = transform_feature_types(split.x_train[columns], numerical_features=set(), image_features=set())
    start = time.time()
    backbone, tokenizer, stats = text_embeddings.finetune_e5(texts, split.y_train, columns, is_cls=is_cls,
                                                             device=device, model_name=E5_SMALL_V2)
    stats = {**(stats or {}), "finetune_seconds": time.time() - start, "rows": len(texts)}
    print(f"  🎯 E5 LoRA-tuned on {len(texts)} rows in {stats['finetune_seconds']:.0f}s")

    original = text_embeddings.finetune_e5

    def reuse(x, y, cols, **kwargs):
        if sorted(cols) != columns:
            raise ValueError(f"The shared TAR encoder was tuned on {columns}, not {sorted(cols)}")
        return backbone, tokenizer, stats

    text_embeddings.finetune_e5 = reuse
    try:
        yield stats
    finally:
        text_embeddings.finetune_e5 = original


def _model_class(short_name: str):
    from multabench.baselines.catboost import CatBoost
    from multabench.baselines.lgbm import LightGBM
    from multabench.baselines.tabm import TabM
    from multabench.baselines.tabpfn import TabPFNv2, TabPFNv2p5
    classes = {c.SHORT_NAME: c for c in (TabM, CatBoost, LightGBM, TabPFNv2, TabPFNv2p5)}
    return classes[short_name]


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
