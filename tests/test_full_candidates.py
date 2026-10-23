"""Protect the two expensive candidate runs, especially live reporting and checkpoint isolation."""
from __future__ import annotations

import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_kaggle_candidate_notebooks import build_notebook
from kaggle_candidates import emit_fold_reports, export_archives, restore_archive
from run_full_candidate import (CANDIDATES, CONDITIONS, MODELS, audit_runs, collect_runs,
                                course_verdict_auroc, experiment_run_name, experiment_settings, fold_dir,
                                initialize_manifest, make_report, official_splits, paper_score_table,
                                save_fold_report, verified_data)
from tt_curation import MULTABENCH_COMMIT
from tt_curation.conditions import MODELS as NAMES


def make_runs(root, candidate, folds=range(6)):
    output = root / candidate.name
    splits = {}
    for fold in range(6):
        test = candidate.rows//3 + int(fold % 3 < candidate.rows % 3)
        splits[str(fold)] = dict(n_train=candidate.rows-test, n_test=test)
    manifest = dict(protocol_version="candidate_full_v1", dataset=candidate.name,
        source_revision="a"*40, micro_batch=64, gradient_accumulation_steps=4,
        rows=candidate.rows, csv_sha256_lf=candidate.sha256,
        metric=candidate.metric, splits=splits)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest))
    (root / "source_revision.txt").write_text("a"*40 + "\n")
    for fold in folds:
        runs = fold_dir(output, fold) / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        for model in MODELS:
            for condition in CONDITIONS:
                index = CONDITIONS.index(condition)
                score = [.70, .72, .75, .77][index] if candidate.metric == "roc_auc" else [-.5, -.4, -.35, -.3][index]
                row = dict(dataset=candidate.name, model=NAMES[model], short_name=model, condition=condition,
                    fold=fold, status="ok", task=candidate.task, size="10k", test_score=score,
                    test_error=1-score if candidate.metric == "roc_auc" else -score,
                    auroc=score if candidate.metric == "roc_auc" else [.70, .72, .75, .77][index],
                    metric=candidate.metric, n_train=splits[str(fold)]["n_train"],
                    n_test=splits[str(fold)]["n_test"], runtime_s=1.0, multabench_commit=MULTABENCH_COMMIT,
                    n_features=candidate.structured if condition == "structured" else
                        3 if condition == "unstructured" else candidate.structured+3,
                    text_encoder="e5-small-tar" if condition == "joint_tar" else
                        None if condition == "structured" else "e5-small")
                (runs / f"{model}_{condition}_10k_fold{fold}.json").write_text(json.dumps(row))
    return output


@pytest.mark.parametrize("key", CANDIDATES)
def test_exact_grid_accepts_binary_auroc_and_negative_multiclass_scores(tmp_path, key):
    candidate = CANDIDATES[key]
    output = make_runs(tmp_path, candidate)
    assert audit_runs(output, candidate) == []
    assert len(collect_runs(output)) == 120
    path = fold_dir(output, 3) / "runs/cat_joint_frozen_10k_fold3.json"
    row = json.loads(path.read_text())
    for field, value in [("text_encoder", "tfidf"), ("dataset", "steam_recommended"),
                         ("n_train", 10000 if key == "food" else 5278), ("test_score", float("nan")),
                         ("test_error", -1), ("auroc", 1.2), ("task", "regression"),
                         ("multabench_commit", "wrong"), ("status", "error")]:
        path.write_text(json.dumps({**row, field: value}))
        assert len(audit_runs(output, candidate)) == 1, field
    path.unlink()
    (path.parent / "unrelated.json").write_text(json.dumps(row))
    assert len(audit_runs(output, candidate)) == 1


@pytest.mark.parametrize("key", CANDIDATES)
def test_final_verdict_requires_all_six_folds_and_partial_clears_stale_verdict(tmp_path, key):
    candidate = CANDIDATES[key]
    output = make_runs(tmp_path, candidate)
    assert make_report(output, candidate)
    import pandas as pd
    report = pd.read_csv(output / "evaluation_report.csv")
    assert len(report) == 20 and set(report.n_folds) == {6}
    verdict = pd.read_csv(output / "curation_verdict.csv")
    assert verdict.joint_signal.all() and verdict.tar_gain.all()
    assert verdict.delta_frozen_vs_best_unimodal.iloc[0] == pytest.approx(.03)
    assert set(verdict.curation_score_column) == {"auroc"}
    assert (output / "paper_verdict.csv").exists()
    assert json.loads((output / "curation_summary.json").read_text())["course"]["accepted"]
    (fold_dir(output, 5) / "runs/light_joint_tar_10k_fold5.json").write_text('{"status":')
    assert not make_report(output, candidate)
    assert not (output / "curation_verdict.csv").exists()
    assert not (output / "paper_verdict.csv").exists()
    assert not (output / "curation_summary.json").exists()
    assert "INCOMPLETE" in (output / "summary.txt").read_text()


@pytest.mark.parametrize("key", CANDIDATES)
def test_live_fold_table_prints_before_workers_finish_and_only_once(tmp_path, capsys, key):
    candidate = CANDIDATES[key]
    output = make_runs(tmp_path, candidate, folds=[0])
    save_fold_report(output, candidate, 0, elapsed=60)
    assert (fold_dir(output, 0) / "scores.csv").exists()
    capsys.readouterr()  # The worker's own output goes to its log; test supervisor forwarding separately.
    # Some later baselines may exist, but must not enter cumulative comparisons.
    extra = fold_dir(output, 2) / "runs"
    extra.mkdir(parents=True)
    sample = json.loads((fold_dir(output, 0) / "runs/cat_structured_10k_fold0.json").read_text())
    (extra / "cat_structured_10k_fold2.json").write_text(json.dumps({**sample, "fold": 2, "test_score": .99}))
    announced = set()
    emit_fold_reports(tmp_path, candidate, announced)
    printed = capsys.readouterr().out
    assert "FOLD 0 COMPLETE" in printed and "20/120" in printed
    assert "Joint delta" in printed and "TAR delta" in printed
    assert all(name in printed for name in NAMES.values())
    assert "PRELIMINARY" in printed and "completed folds [0]" in printed
    assert announced == {0}
    if key in ("food", "device"):
        assert "Primary curation metric: MACRO OvR AUROC" in printed
        assert "Diagnostic repository score: NEGATIVE LOG LOSS" in printed
    assert "Paper rule on these course folds:" in printed
    assert (output / "progress_report.txt").exists()
    emit_fold_reports(tmp_path, candidate, announced)
    assert capsys.readouterr().out == ""
    make_runs(tmp_path, candidate, folds=[1])
    save_fold_report(output, candidate, 1, elapsed=90)
    capsys.readouterr()
    emit_fold_reports(tmp_path, candidate, announced)
    printed = capsys.readouterr().out
    assert "FOLD 1 COMPLETE" in printed and "completed folds [0, 1]" in printed and "40/120" in printed
    assert announced == {0, 1}


def test_resume_manifest_rejects_changes_to_batch_data_code_and_source(tmp_path):
    settings = dict(micro_batch=64, gradient_accumulation_steps=4, csv="fixed", code="fixed", source_revision="a"*40)
    output = tmp_path / "experiment"
    initialize_manifest(output, settings, prepare=True)
    initialize_manifest(output, settings)
    for field, value in [("micro_batch", 16), ("gradient_accumulation_steps", 16),
                         ("csv", "different"), ("code", "different"),
                         ("source_revision", "b"*40)]:
        with pytest.raises(ValueError, match="settings differ"):
            initialize_manifest(output, {**settings, field: value}, prepare=True)


@pytest.mark.parametrize("key", CANDIDATES)
def test_candidate_archives_restore_exact_dataset_and_exclude_partial_checkpoints(tmp_path, key):
    candidate = CANDIDATES[key]
    root = tmp_path / candidate.run_name
    output = make_runs(root, candidate, folds=[0])
    trainer = fold_dir(output, 0) / "checkpoints/trainer"
    complete, partial = trainer / "checkpoint-1", trainer / "checkpoint-2"
    complete.mkdir(parents=True)
    partial.mkdir()
    (complete / "optimizer.pt").write_bytes(b"saved optimizer")
    (complete / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
    (partial / "optimizer.pt").write_bytes(b"incomplete")
    assert export_archives(root, live=True)
    checkpoint = tmp_path / f"{candidate.run_name}_checkpoint.zip"
    with zipfile.ZipFile(checkpoint) as archive:
        assert any("checkpoint-1/optimizer.pt" in name for name in archive.namelist())
        assert not any("checkpoint-2/" in name for name in archive.namelist())
    with zipfile.ZipFile(tmp_path / f"{candidate.run_name}_results.zip") as archive:
        assert not any("checkpoints/" in name for name in archive.namelist())
    other = tmp_path / "restored" / candidate.run_name
    other.parent.mkdir()
    assert restore_archive(checkpoint, other, candidate.name) == "a"*40
    assert not audit_runs(other / candidate.name, candidate, [0])
    with pytest.raises(ValueError, match="existing output"):
        restore_archive(checkpoint, other, candidate.name)


def test_live_copy_keeps_previous_checkpoint_zip_when_files_change(tmp_path, monkeypatch):
    candidate = CANDIDATES["food"]
    root = tmp_path / candidate.run_name
    output = make_runs(root, candidate, folds=[0])
    trainer = fold_dir(output, 0) / "checkpoints/trainer/checkpoint-1"
    trainer.mkdir(parents=True)
    (trainer / "trainer_state.json").write_text('{"global_step":1}')
    (trainer / "optimizer.pt").write_bytes(b"original")
    assert export_archives(root, live=True)
    checkpoint = tmp_path / f"{candidate.run_name}_checkpoint.zip"
    previous = checkpoint.read_bytes()
    original = zipfile.ZipFile.write

    def mutate(self, filename, *args, **kwargs):
        result = original(self, filename, *args, **kwargs)
        if Path(filename).name == "optimizer.pt":
            Path(filename).write_bytes(b"changed while snapshot was running")
        return result

    monkeypatch.setattr(zipfile.ZipFile, "write", mutate)
    assert not export_archives(root, live=True)
    assert checkpoint.read_bytes() == previous


@pytest.mark.parametrize("member", ["../outside.txt", "fda_food_full/../../outside.txt",
                                    "/absolute.txt", "lendingclub_full/source_revision.txt", "fda_food_full\\bad"])
def test_restore_rejects_other_roots_and_unsafe_paths_before_extracting(tmp_path, member):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(member, "bad")
    root = tmp_path / "fda_food_full"
    with pytest.raises(ValueError, match="Unsafe|metadata"):
        restore_archive(archive, root, "fda_food_severity")
    assert not root.exists()


def test_restore_rejects_symlinks(tmp_path):
    archive = tmp_path / "bad.zip"
    info = zipfile.ZipInfo("fda_food_full/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(info, "../../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        restore_archive(archive, tmp_path / "fda_food_full", "fda_food_severity")


@pytest.mark.parametrize("key", CANDIDATES)
def test_notebook_cells_and_embedded_model_check_compile_with_live_unbuffered_output(key):
    notebook = build_notebook(CANDIDATES[key], "a"*40)
    code = []
    for index, entry in enumerate(notebook["cells"]):
        if entry["cell_type"] == "code":
            compile(entry["source"], f"cell{index}", "exec")
            assert entry["outputs"] == [] and entry["execution_count"] is None
            code.append(entry["source"])
    full = "\n".join(code)
    assert f"MICRO_BATCH = {256 if key == 'device' else 64}" in full and "'PYTHONUNBUFFERED': '1'" in full
    expected = "physical batch 256, accumulation 1" if key == "device" else "micro-batch 64, accumulation 4"
    assert expected in notebook["cells"][0]["source"]
    assert "scripts/kaggle_candidates.py" in full and "print(line, end='', flush=True)" in full
    assert CANDIDATES[key].run_name in code[0]
    captured = []
    check = next(source for source in code if "model_check =" in source)
    namespace = dict(PYTHON="python", ENV={}, run_checked=lambda args, **kw: captured.append(args))
    exec(check, namespace)
    compile(captured[0][-1], "embedded_model_check", "exec")
    assert "20/20 smoke" not in full and "--max_hours" not in full


@pytest.mark.bench
@pytest.mark.parametrize("key", CANDIDATES)
def test_real_prepared_csv_and_all_official_split_sizes(key):
    candidate = CANDIDATES[key]
    data, config = verified_data(candidate)
    splits = official_splits(data)
    batch = 256 if key == "device" else 64
    settings = experiment_settings(candidate, data, config, micro_batch=batch)
    assert settings["splits"] == splits and settings["tar_max_epochs"] == 50
    assert settings["effective_batch"] == 256 and settings["precision"] == "float32"
    assert settings["micro_batch"] == batch and settings["gradient_accumulation_steps"] == 256//batch
    assert settings["accumulation_loss"] == ("reference_single_batch_mean" if key == "device" else "ce_sum_over_effective_batch_items_v1")
    assert settings["tar_parameters"] == dict(learning_rate=1e-4, epochs=50, lora_rank=16,
        layers=3, patience=3, weight_decay=.001, batch_size=256)
    assert settings["curation_metric"] == ("auroc" if key == "lendingclub" else "macro_ovr_auroc")
    assert len(splits) == 6
    for fold, split in splits.items():
        assert split["n_train"] + split["n_test"] == candidate.rows
        if key in ("lendingclub", "device"):
            assert (split["n_train"], split["n_test"]) == (10000, 5000)
        else:
            assert split["n_train"] in (5278, 5279) and split["n_test"] in (2639, 2640)


@pytest.mark.parametrize("saved_batch", [16, 64])
def test_notebook_resume_checks_batch_before_downloading_source(tmp_path, saved_batch):
    import re
    import tempfile
    candidate = CANDIDATES["food"]
    root = tmp_path / candidate.run_name
    output = make_runs(root, candidate, folds=[])
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(micro_batch=saved_batch, gradient_accumulation_steps=256//saved_batch)
    manifest_path.write_text(json.dumps(manifest))
    export_archives(root)
    target = tmp_path / "restore" / candidate.run_name
    target.parent.mkdir()
    target.mkdir()
    namespace = dict(Path=Path, zipfile=zipfile, json=json, stat=stat, re=re, tempfile=tempfile,
        RUN_ROOT=target, RUN_NAME=candidate.run_name, DATASET_NAME=candidate.name,
        RESUME_ZIP=str(tmp_path / f"{candidate.run_name}_checkpoint.zip"), REPO_REF="b"*40, MICRO_BATCH=64)
    source = build_notebook(candidate, "b"*40)["cells"][2]["source"]
    if saved_batch == 16:
        with pytest.raises(RuntimeError, match="Checkpoint batch differs"):
            exec(source, namespace)
    else:
        exec(source, namespace)
        assert namespace["REPO_REF"] == "a"*40
    assert (target / candidate.name / "manifest.json").exists()  # Saved work is preserved in both cases.


def test_supervisor_has_no_ten_hour_cutoff(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    import kaggle_candidates as supervisor
    candidate = CANDIDATES["food"]
    root = tmp_path / candidate.run_name
    make_runs(root, candidate, folds=[])
    clock = [0]

    class Process:
        def poll(self):
            return None if clock[0] < 22*3600 else 0

    monkeypatch.setattr(sys, "argv", ["kaggle_candidates.py", "--dataset", "food", "--results_root", str(root),
                                    "--micro_batch", "64"])
    monkeypatch.setattr(supervisor, "audit_runs", lambda *a, **kw: ["missing"])
    monkeypatch.setattr(supervisor, "emit_fold_reports", lambda *a, **kw: None)
    monkeypatch.setattr(supervisor, "progress", lambda *a: (90, []))
    monkeypatch.setattr(supervisor, "make_report", lambda *a: True)
    monkeypatch.setattr(supervisor, "export_archives", lambda *a, **kw: True)
    monkeypatch.setattr(supervisor, "stop_workers", lambda *a: None)
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0]+11*3600))
    supervisor.main()
    printed = capsys.readouterr().out
    assert "elapsed 11.00 h" in printed and "COMPLETE: all 120" in printed


def test_device_never_silently_changes_official_batch():
    with pytest.raises(ValueError, match="physical batch 256"):
        experiment_settings(CANDIDATES["device"], None, None, micro_batch=64)


def test_device_notebook_verifies_classification_only_and_isolates_output():
    notebook = build_notebook(CANDIDATES["device"], "a"*40)
    full = "\n".join(c["source"] for c in notebook["cells"])
    assert "Class I=988, Class II=13,426, Class III=586" in full
    assert "MICRO_BATCH = 256" in full and "fda_device_full_kaggle" in full
    assert "TabPFNRegressor" not in full and "--max_epochs" not in full
    assert "No micro-batch fallback is authorized" in full


def test_bootstrap_source_revision_uses_real_newline():
    import ast
    notebook = build_notebook(CANDIDATES["device"], "a"*40)
    source = next(c["source"] for c in notebook["cells"] if "revision_file.write_text" in c["source"])
    writes = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "write_text"]
    assert len(writes) == 2
    for call in writes:
        assert isinstance(call.args[0], ast.BinOp)
        assert call.args[0].right.value == "\n"


def test_food_physical_batch256_notebook_preserves_original_batch64():
    import ast
    batch256 = build_notebook(CANDIDATES['food'], 'a'*40, micro_batch=256)
    original = build_notebook(CANDIDATES['food'], 'a'*40)
    assert 'MICRO_BATCH = 64' in original['cells'][1]['source']
    assert 'physical batch 256, accumulation 1' in batch256['cells'][0]['source']
    assert 'micro-batch 64' not in batch256['cells'][0]['source']
    settings = ast.parse(batch256['cells'][1]['source'])
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in settings.body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in ('DATASET', 'DATASET_NAME', 'MICRO_BATCH')}
    assert values == dict(DATASET='food', DATASET_NAME='fda_food_severity', MICRO_BATCH=256)
    for entry in batch256['cells']:
        if entry['cell_type'] == 'code':
            compile(entry['source'], 'food256_cell', 'exec')
            assert entry['outputs'] == []
    full = '\n'.join(entry['source'] for entry in batch256['cells'])
    assert 'Checkpoint batch differs' in full
    assert 'No micro-batch fallback is authorized' in full
    assert 'fda_food_full_batch256' in batch256['cells'][1]['source']
    assert 'fda_food_full_batch256' not in original['cells'][1]['source']
    assert experiment_run_name(CANDIDATES['food'], 256) == 'fda_food_full_batch256'


@pytest.mark.parametrize("key", ["food", "device"])
def test_multiclass_verdict_uses_auroc_even_when_log_loss_worsens(tmp_path, key):
    candidate = CANDIDATES[key]
    output = make_runs(tmp_path, candidate)
    originals = {}
    for path in output.glob("folds/fold*/runs/*.json"):
        row = json.loads(path.read_text())
        row['test_score'] = [-.5, -.4, -.45, -.50][CONDITIONS.index(row['condition'])]
        row['test_error'] = -row['test_score']
        path.write_text(json.dumps(row))
        originals[path] = path.read_bytes()
    assert not audit_runs(output, candidate)
    assert make_report(output, candidate)
    import pandas as pd
    verdict = pd.read_csv(output / 'curation_verdict.csv')
    assert verdict.joint_signal.all() and verdict.tar_gain.all()
    assert verdict.delta_frozen_vs_best_unimodal.iloc[0] == pytest.approx(.03)
    assert verdict.delta_tar_vs_frozen.iloc[0] == pytest.approx(.02)
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_course_and_paper_count_different_learners_separately(tmp_path):
    output = make_runs(tmp_path, CANDIDATES['food'])
    # Joint passes A/B/C; TAR passes B/C/D. The course accepts; the paper's same-three rule does not.
    for path in output.glob('folds/fold*/runs/*.json'):
        row = json.loads(path.read_text())
        index = MODELS.index(row['short_name'])
        joint = .61 if index < 3 else .59
        tar = joint + (.01 if index in (1, 2, 3) else -.01)
        row['auroc'] = [.5, .6, joint, tar][CONDITIONS.index(row['condition'])]
        path.write_text(json.dumps(row))
    assert make_report(output, CANDIDATES['food'])
    summary = json.loads((output / 'curation_summary.json').read_text())
    assert summary['course']['joint_learners'] == summary['course']['tar_learners'] == 3
    assert summary['course']['accepted'] is True
    assert summary['paper']['both_learners'] == 2 and summary['paper']['accepted'] is False


def test_paper_rounds_condition_means_before_deltas_and_handles_exact_margin(tmp_path):
    output = make_runs(tmp_path, CANDIDATES['food'], folds=[0])
    for path in output.glob('folds/fold*/runs/*.json'):
        row = json.loads(path.read_text())
        values = [.4, .60049, .60051, .60149] if row['short_name'] == 'cat' else [.4, .600, .601, .602]
        row['auroc'] = values[CONDITIONS.index(row['condition'])]
        path.write_text(json.dumps(row))
    runs = collect_runs(output, [0])
    paper = paper_score_table(runs)
    assert paper.loc['CatBoost', 'Joint delta'] == .001
    assert paper.loc['CatBoost', 'TAR delta'] == 0
    assert paper.loc['CatBoost', 'joint_signal'] and not paper.loc['CatBoost', 'tar_gain']
    assert paper.loc['LightGBM', 'passes_both']
    course = course_verdict_auroc(runs).set_index('model')
    assert course.loc['CatBoost', 'joint_signal'] and course.loc['CatBoost', 'tar_gain']
