"""Resume-only safeguards: preserve completed results and reject accidental fresh/mixed runs."""
import ast
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_linkedin_resume_notebook import (EXPECTED_SOURCE, LOSS_VERSION, REPORT_PROGRAM,
                                           build_notebook, restore_input, validate_resume)


def checkpoint(root, complete=False):
    output = root / "linkedin_salary"
    output.mkdir(parents=True)
    manifest = dict(source_revision=EXPECTED_SOURCE, dataset="linkedin_salary", micro_batch=16,
                    effective_batch=256, accumulation_loss=LOSS_VERSION, tar_max_epochs=50,
                    precision="float32", csv_sha256_lf="8880f84563535c60eda01be5b52010dea6f80e318f228ade00166d7cb66d57fa")
    (output / "manifest.json").write_text(json.dumps(manifest))
    (root / "source_revision.txt").write_text(EXPECTED_SOURCE + "\n")
    names = {"tabm": "TabM", "cat": "CatBoost", "light": "LightGBM",
             "tabpfnv2": "TabPFN-v2", "tabpfnv2p5": "TabPFN-v2p5"}
    conditions = ["structured", "unstructured", "joint_frozen", "joint_tar"]
    for fold in range(6):
        for model, display in names.items():
            for condition in conditions:
                if not complete and fold == 5 and condition == "joint_tar":
                    continue
                folder = output / "folds" / f"fold{fold}" / "runs"
                folder.mkdir(parents=True, exist_ok=True)
                row = dict(dataset="linkedin_salary", short_name=model, model=display, condition=condition,
                           fold=fold, task="regression", size="10k", status="ok", n_train=10000,
                           n_test=5000, multabench_commit="ff0c9c8b40a2085a06a7771190e53a261341802f",
                           metric="root_mean_squared_error", test_score=-.3, test_error=.3,
                           r2=[.1, .2, .3, .4][conditions.index(condition)],
                           text_encoder=None if condition == "structured" else
                           "e5-small-tar" if condition == "joint_tar" else "e5-small")
                (folder / f"{model}_{condition}_10k_fold{fold}.json").write_text(json.dumps(row))
    cache = output / "folds/fold5/checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "accumulation_loss.json").write_text(json.dumps(dict(version=LOSS_VERSION,
        micro_batch=16, effective_batch=256)))
    trainer = cache / "trainer/checkpoint-2130"
    trainer.mkdir(parents=True)
    (trainer / "trainer_state.json").write_text(json.dumps(dict(epoch=30., global_step=2130)))
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"):
        (trainer / name).write_bytes(b"fixture checkpoint file; not executable")
    return output


def test_resume_retains_115_results_and_epoch30_without_migration(tmp_path):
    root = tmp_path / "linkedin_full"
    output = checkpoint(root)
    originals = {p: p.read_bytes() for p in output.glob("folds/fold*/runs/*.json")}
    state = validate_resume(root)
    assert state["n_results"] == 115 and state["missing_evaluations"] == 5
    assert state["checkpoint_epoch"] == 30
    assert state["micro_batch"] == state["accumulation"] == 16
    assert all(path.read_bytes() == data for path, data in originals.items())


@pytest.mark.parametrize("unpacked", [False, True])
def test_resume_restores_zip_or_unpacked_kaggle_input(tmp_path, unpacked):
    source = tmp_path / "input/linkedin_full"
    checkpoint(source)
    if unpacked:
        incoming = source
    else:
        incoming = tmp_path / "linkedin_full_checkpoint.zip"
        with zipfile.ZipFile(incoming, "w") as archive:
            for path in source.rglob("*"):
                if path.is_file():
                    archive.write(path, Path("linkedin_full") / path.relative_to(source))
    target = tmp_path / "working/linkedin_full"
    target.parent.mkdir()
    restore_input(incoming, target)
    assert validate_resume(target)["checkpoint_epoch"] == 30
    with pytest.raises(ValueError, match="overwrite"):
        restore_input(incoming, target)


@pytest.mark.parametrize("change", [dict(micro_batch=64), dict(accumulation_loss="old_loss"),
                                   dict(source_revision="a"*40), dict(tar_max_epochs=10)])
def test_resume_rejects_setting_changes(tmp_path, change):
    root = tmp_path / "linkedin_full"
    output = checkpoint(root)
    path = output / "manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(AssertionError):
        validate_resume(root)


def test_resume_refuses_restart_when_optimizer_state_or_finished_result_is_missing(tmp_path):
    root = tmp_path / "linkedin_full"
    output = checkpoint(root)
    path = output / "folds/fold5/checkpoints/trainer/checkpoint-2130/optimizer.pt"
    path.unlink()
    with pytest.raises(AssertionError, match="optimizer.pt"):
        validate_resume(root)
    path.write_bytes(b"optimizer state")
    (output / "folds/fold0/runs/cat_structured_10k_fold0.json").unlink()
    with pytest.raises(AssertionError, match="previously completed"):
        validate_resume(root)


def test_finished_snapshot_does_not_require_a_training_checkpoint(tmp_path):
    root = tmp_path / "linkedin_full"
    output = checkpoint(root, complete=True)
    shutil.rmtree(output / "folds/fold5/checkpoints")
    assert validate_resume(root)["missing_evaluations"] == 0


@pytest.mark.parametrize("name", ["../escape.txt", "linkedin_full/../../escape.txt", "another_root/file.txt"])
def test_zip_restore_rejects_unsafe_members_before_extraction(tmp_path, name):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(name, "bad")
    with pytest.raises(ValueError, match="Unexpected checkpoint"):
        restore_input(archive_path, tmp_path / "linkedin_full")
    assert not (tmp_path / "linkedin_full").exists()


def test_notebook_cells_and_embedded_programs_compile_without_time_cutoff_or_toy_run():
    notebook = build_notebook()
    sources = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    for index, source in enumerate(sources):
        compile(source, f"resume_cell{index}", "exec")
    full = "\n".join(sources)
    assert EXPECTED_SOURCE in full and "MICRO_BATCH = 16" in full
    assert "--max_seconds" not in full and "make_toy_dataset" not in full and "migrate_linkedin" not in full
    assert "restore_input(selected,RUN_ROOT)" in full and "resume_state=validate_resume(RUN_ROOT)" in full
    source = next(s for s in sources if "(PROJECT/'SOURCE_REVISION').write_text" in s)
    writes = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text"]
    assert len(writes) == 1 and writes[0].args[0].right.value == "\n"
    captured = []
    check = next(s for s in sources if "model_check=" in s)
    exec(check, dict(resume_state=dict(missing_evaluations=5), ENV={}, PYTHON="python",
                     run_checked=lambda args, **kwargs: captured.append(args)))
    compile(captured[0][-1], "regression_checkpoint_check", "exec")


@pytest.mark.parametrize("complete", [False, True])
def test_final_report_uses_raw_r2_and_same_completed_folds(tmp_path, complete):
    root = tmp_path / "linkedin_full"
    output = checkpoint(root, complete=complete)
    if not complete:
        for path in (output / "folds/fold5/runs").glob("*.json"):
            row = json.loads(path.read_text())
            row["r2"] = .99  # Partial baselines must not enter the five-fold table.
            path.write_text(json.dumps(row))
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", REPORT_PROGRAM, str(root)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    text = (output / "progress_r2.txt").read_text()
    assert "0.300000" in text and "0.100000" in text and "0.990000" not in text
    assert "Joint Signal 5/5; TAR Gain 5/5" in text
    if complete:
        assert "FINAL: 120/120" in text
        summary = json.loads((output / "curation_summary_r2.json").read_text())
        assert summary["accepted"] and summary["metric"] == "r2"
        scores = pd.read_csv(output / "curation_verdict_r2.csv")
        assert scores["Joint delta"].between(.099999, .100001).all()
    else:
        assert "matching completed folds [0, 1, 2, 3, 4]" in text
        assert not (output / "curation_verdict_r2.csv").exists()
