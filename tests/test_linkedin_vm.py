"""Guard the fresh VM run against mixed experiments, misleading reports and silent batch changes."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_full_linkedin_vm as vm
from tt_curation import MULTABENCH_COMMIT
from tt_curation.conditions import MODELS as NAMES


def saved_runs(output, folds=range(6)):
    vm.write_json(output / "manifest.json", {"source_revision": "a" * 40})
    for fold, model, condition, path in vm.expected_paths(output, folds):
        # Deliberately make native negative RMSE disagree with R2's criterion signs.
        r2 = {"structured": .4, "unstructured": .6, "joint_frozen": .65, "joint_tar": .7}[condition]
        error = {"structured": .2, "unstructured": .3, "joint_frozen": .4, "joint_tar": .5}[condition]
        row = dict(protocol=vm.PROTOCOL, dataset="linkedin_salary", model=NAMES[model], short_name=model,
            condition=condition, fold=fold, size="10k", task="regression", n_train=10_000, n_test=5_000,
            n_features=12 if condition == "structured" else 2 if condition == "unstructured" else 14,
            text_encoder="e5-small-tar" if condition == "joint_tar" else None if condition == "structured" else "e5-small",
            multabench_commit=MULTABENCH_COMMIT, git="a" * 40, metric="root_mean_squared_error",
            r2=r2, test_error=error, test_score=-error, runtime_s=1.2, device="cuda", status="ok")
        if condition == "joint_tar":
            row.update(tar_parameters=vm.TAR, training_arguments=vm.TRAINING,
                tar_finetune=dict(rows=10_000, epochs=8., best_val_loss=2., finetune_seconds=100.))
        vm.write_json(path, row)


def test_manifest_requires_empty_output_and_exact_settings(tmp_path):
    settings = {"batch": 256, "source_revision": "a" * 40}
    with pytest.raises(ValueError, match="EMPTY"):
        vm.initialize_manifest(tmp_path, settings)
    (tmp_path / "old_adapter").mkdir()
    with pytest.raises(ValueError, match="EMPTY"):
        vm.initialize_manifest(tmp_path, settings, prepare=True)
    (tmp_path / "old_adapter").rmdir()
    vm.initialize_manifest(tmp_path, settings, prepare=True)
    vm.initialize_manifest(tmp_path, settings)
    with pytest.raises(ValueError, match="Manifest differs"):
        vm.initialize_manifest(tmp_path, dict(settings, batch=16), prepare=True)


@pytest.mark.parametrize("change", [
    {"r2": float("nan")}, {"r2": 1.1}, {"test_score": .3}, {"protocol": "old_kaggle"},
    {"n_features": 13}, {"text_encoder": "tfidf"}, {"multabench_commit": "wrong"},
    {"git": "b" * 40}, {"training_arguments": dict(vm.TRAINING, gradient_accumulation_steps=16)},
    {"tar_parameters": dict(vm.TAR, weight_decay=.01)}, {"tar_finetune": {"rows": 10000}},
])
def test_audit_rejects_wrong_metrics_features_provenance_or_tar(tmp_path, change):
    saved_runs(tmp_path)
    assert not vm.audit_runs(tmp_path)
    path = tmp_path / "folds/fold0/runs/cat_joint_tar_10k_fold0.json"
    row = json.loads(path.read_text())
    row.update(change)
    vm.write_json(path, row)
    assert len(vm.audit_runs(tmp_path)) == 1


def test_audit_accepts_negative_r2_but_rejects_missing_and_extra(tmp_path):
    saved_runs(tmp_path)
    path = tmp_path / "folds/fold0/runs/cat_structured_10k_fold0.json"
    row = json.loads(path.read_text())
    vm.write_json(path, dict(row, r2=-.5))
    assert not vm.audit_runs(tmp_path)
    path.unlink()
    assert len(vm.audit_runs(tmp_path)) == 1
    assert not vm.audit_runs(tmp_path, allow_missing=True)
    vm.write_json(path.parent / "extra.json", {})
    assert len(vm.audit_runs(tmp_path, allow_missing=True)) == 1


def test_live_fold_table_uses_r2_and_excludes_partial_folds(tmp_path, capsys):
    saved_runs(tmp_path, folds=[0])
    partial = tmp_path / "folds/fold1/runs/cat_structured_10k_fold1.json"
    vm.write_json(partial, {"status": "ok", "r2": 1.})
    assert not vm.write_reports(tmp_path, just_finished=0)
    text = capsys.readouterr().out
    assert "FOLD 0 COMPLETE" in text and "folds [0]" in text
    assert "Joint Signal: 5/5 | TAR Gain: 5/5" in text
    assert "20/120" in text and "PRELIMINARY" in text
    verdict = pd.read_csv(tmp_path / "r2_cumulative_preliminary.csv")
    assert verdict.score_joint_tar.tolist() == [.7] * 5
    assert set(verdict.folds) == {1}
    assert not (tmp_path / "curation_verdict.csv").exists()
    # Raw native diagnostics have not been changed to R2.
    row = json.loads((tmp_path / "folds/fold0/runs/cat_joint_tar_10k_fold0.json").read_text())
    assert row["test_score"] == -.5


def test_final_report_requires_all120_and_removes_stale_verdict(tmp_path):
    saved_runs(tmp_path)
    assert vm.write_reports(tmp_path)
    assert "Final six-fold course verdict: PASS" in (tmp_path / "summary.txt").read_text()
    assert set(pd.read_csv(tmp_path / "evaluation_report.csv").n_folds) == {6}
    (tmp_path / "folds/fold5/runs/tabm_joint_tar_10k_fold5.json").unlink()
    assert not vm.write_reports(tmp_path)
    assert not (tmp_path / "curation_verdict.csv").exists()
    assert "100/120" in (tmp_path / "summary.txt").read_text()


def test_fail_fast_saves_traceback_stops_and_restores_runner(tmp_path, monkeypatch):
    from tt_curation import runner
    row = dict(status="error", error="CUDA out of memory", traceback="Traceback: CUDA out of memory",
               short_name="cat", condition="joint_tar", fold=0)
    original = lambda *args, **kwargs: row.copy()
    monkeypatch.setattr(runner, "run_one", original)
    with pytest.raises(RuntimeError, match="No memory fallback"):
        with vm.fail_fast(tmp_path):
            runner.run_one()
            pytest.fail("Must not reach another evaluation")
    assert runner.run_one is original
    saved = json.loads((tmp_path / "runs/cat_joint_tar_10k_fold0.json").read_text())
    assert "out of memory" in saved["traceback"] and saved["tar_parameters"] == vm.TAR


@pytest.mark.bench
def test_real_prepared_data_and_all_six_official_regression_splits():
    data, config = vm.verified_data()
    settings = vm.experiment_settings(data, config)
    assert settings["tar_parameters"]["batch_size"] == 256
    assert settings["training_arguments"]["gradient_accumulation_steps"] == 1
    assert settings["curation_metric"] == "r2"
    assert len(settings["splits"]) == 6
    assert all(s["n_train"] == 10000 and s["n_test"] == 5000 for s in settings["splits"].values())
    assert len({s["train_indices_sha256"] for s in settings["splits"].values()}) == 6


@pytest.mark.bench
@pytest.mark.parametrize("smoke", [False, True])
def test_guard_native_training_arguments_loss_and_smoke_isolation(tmp_path, smoke):
    import torch
    from multabench.finetune import lora
    from transformers import EarlyStoppingCallback
    original = lora.Trainer
    with vm.guarded_tar(tmp_path, smoke_steps=2 if smoke else None):
        args = lora.TrainingArguments(output_dir=str(tmp_path / "ignored"), report_to="none",
            num_train_epochs=50, per_device_train_batch_size=256, per_device_eval_batch_size=256,
            learning_rate=.0001, weight_decay=.001, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, dataloader_num_workers=4)
        dataset = [{"input_ids": torch.zeros(512, dtype=torch.long), "labels": torch.tensor(0)}]
        trainer = lora.Trainer(model=torch.nn.Linear(2, 2), args=args, train_dataset=dataset,
            eval_dataset=dataset, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
        assert trainer.compute_loss_func is None
        assert args.max_steps == -1 and args.num_train_epochs == 50
        smoke_callbacks = [c for c in trainer.callback_handler.callbacks if type(c).__name__ == "SmokeStop"]
        assert len(smoke_callbacks) == int(smoke)
        if smoke:
            control = SimpleNamespace(should_training_stop=False)
            smoke_callbacks[0].on_step_end(args, SimpleNamespace(global_step=2), control)
            assert control.should_training_stop
        else:
            assert not smoke_callbacks
        assert json.loads((tmp_path / "actual_training_arguments.json").read_text()) == vm.TRAINING
        args.gradient_accumulation_steps = 4
        with pytest.raises(ValueError, match="Refusing changed"):
            lora.Trainer(model=torch.nn.Linear(2, 2), args=args, train_dataset=dataset,
                eval_dataset=dataset, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    assert lora.Trainer is original


@pytest.mark.bench
def test_real_trainer_smoke_stops_after_two_updates_but_validates_and_keeps_50_epoch_schedule(tmp_path, monkeypatch):
    """Exercise Trainer's callback/epoch/save flow, not just callback methods in isolation."""
    import torch
    from multabench.finetune import lora
    from transformers import EarlyStoppingCallback
    torch.set_num_threads(2)
    # Only this tiny CPU callback test uses zero workers, allowing its local model/dataset
    # classes on Windows. Production experiment_settings rejects non-official worker counts.
    monkeypatch.setitem(vm.TRAINING, "dataloader_num_workers", 0)

    class Examples(torch.utils.data.Dataset):
        def __len__(self):
            return 1024

        def __getitem__(self, index):
            return {"input_ids": torch.arange(512) % 32, "labels": torch.tensor(index % 2)}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(32, 4)
            self.head = torch.nn.Linear(4, 2)

        def forward(self, input_ids, labels, **kwargs):
            logits = self.head(self.emb(input_ids).mean(1))
            return {"logits": logits, "loss": torch.nn.functional.cross_entropy(logits, labels)}

    with vm.guarded_tar(tmp_path, smoke_steps=2):
        args = lora.TrainingArguments(output_dir=str(tmp_path), report_to="none",
            num_train_epochs=50, learning_rate=.0001, weight_decay=.001,
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, remove_unused_columns=False,
            disable_tqdm=True)
        trainer = lora.Trainer(model=Model(), args=args, train_dataset=Examples(), eval_dataset=Examples(),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
        trainer.train()
        assert trainer.state.global_step == 2
        assert trainer.state.max_steps == 200  # Four updates per epoch * the unchanged official cap 50.
        assert trainer.state.epoch == .5
        assert trainer.state.best_metric is not None
        checkpoint = tmp_path / "trainer/checkpoint-2"
        assert (checkpoint / "optimizer.pt").exists() and (checkpoint / "rng_state.pth").exists()
