"""Only unchanged baseline evaluations may survive the approved TAR execution adjustment."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import re

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from migrate_linkedin_microbatch import LEGACY_WORKER_SHA, WORKER_KEY, migrate
from run_full_linkedin import ACCUMULATION_LOSS, text_digest
from test_full_linkedin import make_runs


def fixture_experiment(tmp_path):
    root = tmp_path / "linkedin_full"
    output = root / "linkedin_salary"
    make_runs(output)
    # Match the actual failed Kaggle attempt: first two folds have 15 successful baselines each.
    # Also leave a tuned result/checkpoint to prove they cannot be imported under new TAR settings.
    for path in output.glob("folds/fold*/runs/*.json"):
        row = json.loads(path.read_text())
        if row["fold"] >= 2:
            path.unlink()
    checkpoint = output / "folds/fold0/checkpoints/trainer/checkpoint-1/optimizer.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"old batch-256 state")
    old = dict(micro_batch=256, effective_batch=256, source_revision="a" * 40,
               csv_sha256_lf="sample", config_sha256_lf="config", precision="float32",
               tar_max_epochs=50, code={"src/tt_curation/runner.py": "unchanged"}, packages={"torch": "2.7.1"})
    (output / "manifest.json").write_text(json.dumps(old))
    (root / "source_revision.txt").write_text(old["source_revision"])
    expected = {**deepcopy(old), "micro_batch": 16, "source_revision": "b" * 40}
    return root, old, expected


def test_preserve_old_experiment_import_30_baselines_and_never_tar(tmp_path):
    root, old, expected = fixture_experiment(tmp_path)
    assert migrate(root, expected, expected["source_revision"]) == 30
    rows = [json.loads(p.read_text()) for p in root.glob("linkedin_salary/folds/fold*/runs/*.json")]
    assert len(rows) == 30
    assert all(row["condition"] != "joint_tar" for row in rows)
    assert not list(root.glob("**/optimizer.pt"))
    assert json.loads((root / "linkedin_salary/manifest.json").read_text()) == expected
    backup, = tmp_path.glob("linkedin_full_batch256_backup_*")
    assert json.loads((backup / "linkedin_salary/manifest.json").read_text()) == old
    assert (backup / "source_revision.txt").read_text().strip() == old["source_revision"]
    assert list(backup.glob("**/optimizer.pt"))
    assert len(list(backup.glob("**/*joint_tar*.json"))) == 10
    provenance = json.loads((root / "baseline_reuse.json").read_text())
    assert len(provenance["imported_baselines"]) == 30
    assert provenance["tar_results_imported"] == 0
    assert migrate(root, expected, expected["source_revision"]) == 0  # Safe notebook rerun.


@pytest.mark.parametrize("changed", ["csv_sha256_lf", "code", "packages", "tar_max_epochs"])
def test_reject_scientific_changes_without_touching_old_results(tmp_path, changed):
    root, old, expected = fixture_experiment(tmp_path)
    expected[changed] = "changed"
    with pytest.raises(ValueError, match="Only a TAR micro-batch"):
        migrate(root, expected, expected["source_revision"])
    assert json.loads((root / "linkedin_salary/manifest.json").read_text()) == old
    assert not list(tmp_path.glob("*_backup_*"))


def test_reject_corrupt_successful_baseline_before_mutation(tmp_path):
    root, old, expected = fixture_experiment(tmp_path)
    path = root / "linkedin_salary/folds/fold0/runs/cat_joint_frozen_10k_fold0.json"
    row = json.loads(path.read_text())
    row["text_encoder"] = "tfidf"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="invalid baseline"):
        migrate(root, expected, expected["source_revision"])
    assert json.loads((root / "linkedin_salary/manifest.json").read_text()) == old


def legacy_accumulation_experiment(tmp_path):
    root, old, expected = fixture_experiment(tmp_path)
    # The overnight run saved all 90 baselines and TAR results in folds 0..4.
    for directory in root.glob("linkedin_salary/folds/fold*"):
        # Fixture files only, enumerated within this test's isolated temporary directory.
        for path in directory.glob("runs/*.json"):
            path.unlink()
    # make_runs requires new fold directories, so populate a separate temporary tree then copy JSONs.
    import shutil
    complete = tmp_path / "complete"
    make_runs(complete)
    for path in complete.glob("folds/fold*/runs/*.json"):
        row = json.loads(path.read_text())
        if row["condition"] == "joint_tar" and row["fold"] == 5:
            continue
        target = root / "linkedin_salary" / path.relative_to(complete)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    old.update(micro_batch=16, code={"src/tt_curation/runner.py": "unchanged", WORKER_KEY: LEGACY_WORKER_SHA})
    expected = {**deepcopy(old), "source_revision": "b" * 40, "accumulation_loss": ACCUMULATION_LOSS}
    expected["code"][WORKER_KEY] = text_digest(Path(__file__).resolve().parents[1] / WORKER_KEY)
    (root / "linkedin_salary/manifest.json").write_text(json.dumps(old))
    return root, old, expected


def test_loss_correction_preserves_115_old_runs_and_reuses_only_90_baselines(tmp_path):
    root, old, expected = legacy_accumulation_experiment(tmp_path)
    assert migrate(root, expected, expected["source_revision"]) == 90
    assert len(list(root.glob("**/runs/*.json"))) == 90
    assert not list(root.glob("**/*joint_tar*.json"))
    assert not list(root.glob("**/optimizer.pt"))
    backup, = tmp_path.glob("linkedin_full_legacy_tar_backup_*")
    assert len(list(backup.glob("**/runs/*.json"))) == 115
    assert len(list(backup.glob("**/*joint_tar*.json"))) == 25
    assert list(backup.glob("**/optimizer.pt"))
    assert json.loads((backup / "linkedin_salary/manifest.json").read_text()) == old
    assert json.loads((root / "linkedin_salary/manifest.json").read_text()) == expected
    assert migrate(root, expected, expected["source_revision"]) == 0


@pytest.mark.parametrize("change", ["unknown_worker", "other_code", "packages", "csv_sha256_lf"])
def test_loss_correction_rejects_unrelated_changes(tmp_path, change):
    root, old, expected = legacy_accumulation_experiment(tmp_path)
    if change == "unknown_worker":
        old["code"][WORKER_KEY] = "unknown"
        (root / "linkedin_salary/manifest.json").write_text(json.dumps(old))
    elif change == "other_code":
        expected["code"]["src/tt_curation/runner.py"] = "changed"
    else:
        expected[change] = "changed"
    with pytest.raises(ValueError, match="Only a TAR micro-batch"):
        migrate(root, expected, expected["source_revision"])
    assert len(list(root.glob("**/runs/*.json"))) == 115
    assert not list(tmp_path.glob("*_backup_*"))


@pytest.mark.parametrize("corrected", [False, True])
def test_generated_notebook_fetches_fix_for_legacy_but_pins_corrected_resume(tmp_path, monkeypatch, corrected):
    import build_kaggle_linkedin_notebook as builder
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    (tmp_path / "notebooks").mkdir()
    builder.main()  # Compiles every generated code cell.
    notebook = json.loads((tmp_path / "notebooks/run_linkedin_full_kaggle.ipynb").read_text())
    root = tmp_path / "linkedin_full"
    output = root / "linkedin_salary"
    output.mkdir(parents=True)
    saved = {"micro_batch": 16}
    if corrected:
        saved["accumulation_loss"] = ACCUMULATION_LOSS
    (output / "manifest.json").write_text(json.dumps(saved))
    (root / "source_revision.txt").write_text("a" * 40)
    inputs = tmp_path / "input"
    inputs.mkdir()

    def path(value):
        return inputs if value == "/kaggle/input" else Path(value)

    variables = dict(Path=path, RUN_ROOT=root, WORK=tmp_path, RESUME_ZIP="", REPO_REF="main",
                     MICRO_BATCH=16, json=json, re=re)
    exec(notebook["cells"][2]["source"], variables)
    assert variables["REPO_REF"] == ("a" * 40 if corrected else "main")
