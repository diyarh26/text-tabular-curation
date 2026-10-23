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


def test_live_autosave_excludes_partial_checkpoints_and_rejects_changing_files(tmp_path, monkeypatch):
    root = tmp_path / "linkedin_full"
    trainer = root / "linkedin_salary/folds/fold0/checkpoints/trainer"
    complete, partial = trainer / "checkpoint-1", trainer / "checkpoint-2"
    complete.mkdir(parents=True)
    partial.mkdir()
    (complete / "optimizer.pt").write_bytes(b"complete optimizer")
    (complete / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
    (partial / "optimizer.pt").write_bytes(b"partially written optimizer")
    (root / "source_revision.txt").write_text("a" * 40)
    assert export_archives(root, live=True)
    destination = tmp_path / "linkedin_full_checkpoint.zip"
    previous = destination.read_bytes()
    with zipfile.ZipFile(destination) as archive:
        assert any("checkpoint-1/optimizer.pt" in n for n in archive.namelist())
        assert not any("checkpoint-2/" in n for n in archive.namelist())
    original_write = zipfile.ZipFile.write

    def mutate_after_copy(self, filename, *args, **kwargs):
        result = original_write(self, filename, *args, **kwargs)
        if Path(filename).name == "optimizer.pt":
            Path(filename).write_bytes(b"changed during copy")
        return result

    monkeypatch.setattr(zipfile.ZipFile, "write", mutate_after_copy)
    assert export_archives(root, live=True) is False
    assert destination.read_bytes() == previous


def test_supervisor_continues_past_ten_hours_without_time_cutoff(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    import kaggle_linkedin as supervisor
    clock = [0]

    class Process:
        def poll(self):
            return None if clock[0] < 22 * 3600 else 0

        @property
        def returncode(self):
            return self.poll()

    audits = iter([["missing TAR"], []])
    monkeypatch.setattr(sys, "argv", ["kaggle_linkedin.py", "--results_root", str(tmp_path / "run")])
    monkeypatch.setattr(supervisor, "audit_runs", lambda output: next(audits))
    monkeypatch.setattr(supervisor, "progress", lambda root: (90, []))
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(supervisor, "stop_workers", lambda processes: None)
    monkeypatch.setattr(supervisor, "export_archives", lambda *a, **kw: True)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + 11 * 3600))
    supervisor.main()
    output = capsys.readouterr().out
    assert "elapsed 11.00 h" in output
    assert "COMPLETE: all 120 evaluations audited." in output
    assert "PAUSED" not in output


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
@pytest.mark.parametrize("n_examples,micro_batch", [(256, 16), (35, 16), (256, 256)])
def test_actual_trainer_accumulated_gradients_match_reference_full_batch(tmp_path, n_examples, micro_batch):
    """Use the upstream head and actual Trainer backward; detect scaling bugs, including uneven tails."""
    import copy
    from types import SimpleNamespace
    import torch
    from multabench.finetune import lora
    from run_full_linkedin import accumulated_cross_entropy, full_tar

    torch.set_num_threads(2)
    torch.manual_seed(42)

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=self.linear(input_ids).unsqueeze(1))

    model = lora.EncoderWithHead(Backbone(), pool=lambda h, inputs: h[:, 0], d_output=20)
    reference = copy.deepcopy(model)
    x = torch.randn(n_examples, 4)
    y = torch.arange(n_examples) % 20
    expected_loss = reference(input_ids=x, labels=y)["loss"]
    expected_loss.backward()
    original_trainer = lora.Trainer
    with full_tar(tmp_path / "checkpoint", micro_batch=micro_batch):
        args = lora.TrainingArguments(output_dir=str(tmp_path / "unused"), use_cpu=True, report_to="none")
        trainer = lora.Trainer(model=model, args=args)
        if micro_batch < 256:
            assert trainer.compute_loss_func is accumulated_cross_entropy
        else:
            assert trainer.compute_loss_func is None
        batches = [{"input_ids": x[i:i + micro_batch], "labels": y[i:i + micro_batch]}
                   for i in range(0, n_examples, micro_batch)]
        trainer.current_gradient_accumulation_steps = len(batches)
        count = trainer._get_num_items_in_batch(batches, torch.device("cpu"))
        assert int(count) == n_examples
        actual_loss = sum(trainer.training_step(model, batch, num_items_in_batch=count).item() for batch in batches)
        assert actual_loss == pytest.approx(expected_loss.item(), rel=1e-6)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-7)
    assert lora.Trainer is original_trainer


@pytest.mark.bench
def test_actual_training_updates_evaluation_and_checkpoint_resume(tmp_path, monkeypatch):
    """Compare real optimizer/scheduler updates, evaluation, and a saved-epoch restart on CPU."""
    import copy
    from types import SimpleNamespace
    import torch
    from transformers import TrainerCallback
    from multabench.finetune import lora
    from run_full_linkedin import full_tar

    torch.set_num_threads(2)
    torch.manual_seed(42)

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=self.linear(input_ids).unsqueeze(1))

    initial = lora.EncoderWithHead(Backbone(), pool=lambda h, inputs: h[:, 0], d_output=20)
    x = torch.randn(515, 4)  # Includes an optimizer update with only three examples.
    dataset = [{"input_ids": row, "labels": torch.tensor(i % 20)} for i, row in enumerate(x)]
    original_trainer = lora.Trainer

    def train(cache, micro_batch):
        with full_tar(cache, micro_batch=micro_batch):
            args = lora.TrainingArguments(output_dir=str(tmp_path / "unused"), use_cpu=True,
                report_to="none", num_train_epochs=2, learning_rate=1e-4, weight_decay=.01,
                eval_strategy="epoch", save_strategy="epoch", seed=42,
                remove_unused_columns=False, dataloader_pin_memory=False, disable_tqdm=True)
            model = copy.deepcopy(initial)
            trainer = lora.Trainer(model=model, args=args, train_dataset=dataset, eval_dataset=dataset)
            trainer.train()
            return model, trainer.state.global_step, trainer.evaluate()["eval_loss"]

    reference, full_steps, full_loss = train(tmp_path / "reference", 256)
    accumulated, micro_steps, micro_loss = train(tmp_path / "accumulated", 16)
    assert full_steps == micro_steps == 6
    assert micro_loss == pytest.approx(full_loss, rel=1e-6)
    for a, b in zip(accumulated.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-7)

    class InterruptAfterSave(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            raise RuntimeError("simulated interruption after first corrected epoch")

    class InterruptingTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(InterruptAfterSave())

    monkeypatch.setattr(lora, "Trainer", InterruptingTrainer)
    cache = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(cache, 16)
    assert list(cache.glob("trainer/checkpoint-*/trainer_state.json"))
    monkeypatch.setattr(lora, "Trainer", original_trainer)
    resumed, resumed_steps, resumed_loss = train(cache, 16)
    assert resumed_steps == micro_steps
    assert resumed_loss == pytest.approx(micro_loss, rel=1e-7)
    for a, b in zip(resumed.parameters(), accumulated.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.bench
def test_reject_legacy_tar_checkpoint_before_loading(tmp_path):
    from run_full_linkedin import full_tar
    trainer_dir = tmp_path / "trainer/checkpoint-1"
    trainer_dir.mkdir(parents=True)
    (trainer_dir / "optimizer.pt").write_bytes(b"legacy")
    with pytest.raises(ValueError, match="Legacy TAR checkpoint"):
        with full_tar(tmp_path, micro_batch=16):
            pytest.fail("Legacy checkpoint must be rejected before execution")


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
