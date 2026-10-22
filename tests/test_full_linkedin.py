"""Protect expensive parallel experiments against mixed results and incomplete acceptance."""
import json
from pathlib import Path
import sys
import zipfile
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_full_linkedin import audit_runs, collect_runs, fold_dir, make_report, MODELS, CONDITIONS
from kaggle_linkedin import export_archives, worker_environment
from tt_curation.conditions import MODELS as DISPLAY_NAMES


def make_runs(output):
    for fold in range(6):
        runs = fold_dir(output, fold) / "runs"
        runs.mkdir(parents=True)
        for model in MODELS:
            for condition in CONDITIONS:
                row = dict(dataset="linkedin_salary", model=DISPLAY_NAMES[model], short_name=model,
                           condition=condition, fold=fold, status="ok", test_score=-.3, test_error=.3,
                           r2=-.1, metric="root_mean_squared_error", n_train=10_000, n_test=5_000,
                           text_encoder="e5-small-tar" if condition == "joint_tar" else
                               None if condition == "structured" else "e5-small")
                (runs / f"{model}_{condition}_10k_fold{fold}.json").write_text(json.dumps(row))


def test_audit_all_regression_runs_including_negative_r2_and_real_pilot(tmp_path):
    make_runs(tmp_path)
    assert audit_runs(tmp_path) == []
    # A real successful pilot run must satisfy metric validation, after encoder/settings isolation.
    pilot = Path(__file__).resolve().parents[1] / "results/colab_tar_pilot/linkedin_salary_81312a3d1c69/runs"
    if pilot.exists():
        path = next(pilot.glob("cat*.json"))
        result = json.loads(path.read_text())
        destination = fold_dir(tmp_path, 0) / "runs/cat_joint_tar_10k_fold0.json"
        destination.write_text(json.dumps(result))
        assert audit_runs(tmp_path) == []
    missing = fold_dir(tmp_path, 3) / "runs/cat_structured_10k_fold3.json"
    missing.unlink()
    (missing.parent / "unrelated.json").write_text("{}")
    wrong = fold_dir(tmp_path, 4) / "runs/light_joint_frozen_10k_fold4.json"
    row = json.loads(wrong.read_text())
    row["text_encoder"] = "tfidf"
    wrong.write_text(json.dumps(row))
    failed = fold_dir(tmp_path, 2) / "runs/tabm_joint_tar_10k_fold2.json"
    row = json.loads(failed.read_text())
    row.update(status="error", error="CUDA out of memory")
    failed.write_text(json.dumps(row))
    issues = audit_runs(tmp_path)
    assert len(issues) == 3
    assert any("CUDA out of memory" in i for i in issues)
    assert any("missing" in i for i in issues)


def test_partial_report_never_claims_final_verdict_and_complete_report_has_six_folds(tmp_path):
    make_runs(tmp_path)
    assert make_report(tmp_path)
    assert (tmp_path / "curation_verdict.csv").exists()
    assert "COMPLETE:" in (tmp_path / "summary.txt").read_text(encoding="utf-8")
    import pandas as pd
    assert set(pd.read_csv(tmp_path / "evaluation_report.csv").n_folds) == {6}
    # Ignore unrelated JSONs rather than contaminating the six-fold averages.
    (fold_dir(tmp_path, 0) / "runs/extra.json").write_text("{}")
    assert len(collect_runs(tmp_path)) == 120
    (fold_dir(tmp_path, 5) / "runs/cat_joint_tar_10k_fold5.json").write_text('{"status":')
    assert not make_report(tmp_path)
    assert not (tmp_path / "curation_verdict.csv").exists()
    assert (tmp_path / "summary.txt").read_text(encoding="utf-8").startswith("INCOMPLETE:")


def test_mask_each_worker_before_imports_and_export_resumable_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert worker_environment(0)["CUDA_VISIBLE_DEVICES"] == "0"
    assert worker_environment(1)["CUDA_VISIBLE_DEVICES"] == "1"
    root = tmp_path / "linkedin_full"
    checkpoint = root / "linkedin_salary/folds/fold0/checkpoints/trainer/checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "optimizer.pt").write_bytes(b"saved optimizer")
    (root / "source_revision.txt").write_text("a" * 40)
    (tmp_path / "secret.txt").write_text("must never export")
    export_archives(root)
    with zipfile.ZipFile(tmp_path / "linkedin_full_checkpoint.zip") as archive:
        assert any(n.endswith("optimizer.pt") for n in archive.namelist())
        assert "linkedin_full/source_revision.txt" in archive.namelist()
        assert all("secret" not in n for n in archive.namelist())
    with zipfile.ZipFile(tmp_path / "linkedin_full_results.zip") as archive:
        assert not any("checkpoints" in n for n in archive.namelist())


@pytest.mark.bench
@pytest.mark.parametrize("micro_batch,accumulation", [(256, 1), (16, 16)])
def test_full_tar_restores_early_stopping_patience_without_changing_effective_batch(tmp_path, micro_batch, accumulation):
    import torch
    from transformers import EarlyStoppingCallback
    from multabench.finetune import lora
    from run_full_linkedin import full_tar
    original_arguments = lora.TrainingArguments
    with full_tar(tmp_path / "checkpoint", micro_batch=micro_batch):
        args = lora.TrainingArguments(output_dir=str(tmp_path / "unused"), report_to="none")
        assert args.per_device_train_batch_size == args.per_device_eval_batch_size == micro_batch
        assert args.gradient_accumulation_steps == accumulation
        assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 256
        assert args.restore_callback_states_from_checkpoint
        trainer = lora.Trainer(model=torch.nn.Linear(2, 2), args=args,
                               callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
        saved = EarlyStoppingCallback(early_stopping_patience=3)
        saved.early_stopping_patience_counter = 2
        trainer.state.stateful_callbacks = {"EarlyStoppingCallback": saved.state()}
        trainer._load_callback_state()
        restored = next(c for c in trainer.callback_handler.callbacks if isinstance(c, EarlyStoppingCallback))
        assert restored.early_stopping_patience_counter == 2
        assert restored.early_stopping_patience == 3
    assert lora.TrainingArguments is original_arguments


@pytest.mark.bench
def test_parking_frozen_encoder_keeps_cached_weights_and_embeddings(tmp_path, monkeypatch):
    import torch
    from multabench.e5 import e5_finetune
    from tt_curation import runner
    from run_full_linkedin import park_frozen_e5
    model = torch.nn.Linear(4, 3).eval()
    inputs = torch.ones(2, 4)
    expected = model(inputs).detach().clone()
    tokenizer = object()
    cache = {("cuda", "test-encoder"): (model, tokenizer)}
    monkeypatch.setattr(e5_finetune, "_VANILLA_E5_CACHE", cache)
    vectors = {"passage: review: good": [1.0, 2.0]}
    monkeypatch.setattr(runner, "_EMBEDDINGS", {id(model): (model, vectors)})
    moves = []
    original_to = model.to

    def move(device):
        moves.append(device)
        return original_to(device)

    monkeypatch.setattr(model, "to", move)
    park_frozen_e5()
    assert moves == ["cpu"]
    assert cache[("cuda", "test-encoder")] == (model, tokenizer)
    assert runner._EMBEDDINGS[id(model)][1] is vectors
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
