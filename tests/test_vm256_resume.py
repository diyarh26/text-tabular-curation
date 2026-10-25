"""VM-to-Kaggle transfer must preserve all results, full state, native batch and source."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import resume_linkedin_vm256 as resume
from build_vm256_resume_notebook import build_notebook, build_bundle

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    """Use real manifest provenance with synthetic scores and inert checkpoint files."""
    manifest_path = ROOT / "reproduction/original_vm256/experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output = root / "linkedin_salary"
    output.mkdir(parents=True)
    (output / "manifest.json").write_bytes(manifest_path.read_bytes())
    original = {}
    for model, condition, fold in sorted(resume.MINIMUM):
        target = resume.run_path(output, (model, condition, fold))
        target.parent.mkdir(parents=True, exist_ok=True)
        row = dict(short_name=model, condition=condition, fold=fold, status="ok", dataset="linkedin_salary",
            model=resume.MODELS[model], task="regression", size="10k", protocol=resume.PROTOCOL,
            git=resume.SOURCE, multabench_commit=manifest["multabench_commit"], n_train=10000, n_test=5000,
            n_features=12 if condition=="structured" else 2 if condition=="unstructured" else 14,
            text_encoder=None if condition=="structured" else "e5-small-tar" if condition=="joint_tar" else "e5-small",
            test_score=-.3, test_error=.3, r2=[.1,.2,.3,.4][resume.CONDITIONS.index(condition)])
        if condition=="joint_tar": row.update(tar_parameters=manifest["tar_parameters"],training_arguments=manifest["training_arguments"])
        resume.write_json(target,row)
        relative = target.relative_to(root)
        original[relative.as_posix()] = resume.sha(target)
    checkpoint = output / "folds/fold4/checkpoints/trainer/checkpoint-71"
    checkpoint.mkdir(parents=True)
    state = dict(epoch=1., global_step=71, max_steps=3550,
        best_model_checkpoint="/home/student/original/fold4/checkpoints/trainer/checkpoint-71",
        stateful_callbacks={"EarlyStoppingCallback":{"attributes":{"early_stopping_patience_counter":2}}})
    for name in resume.CHECKPOINT_FILES:
        (checkpoint / name).write_bytes(b"inert test checkpoint")
    resume.write_json(checkpoint / "trainer_state.json", state)
    (root / "source_revision.txt").write_text(resume.SOURCE + "\n")
    index = dict(source_revision=resume.SOURCE, manifest_sha256=resume.sha(output / "manifest.json"),
        original_run_sha256=original, initial_checkpoint_sha256={
            p.relative_to(root).as_posix():resume.sha(p) for p in checkpoint.iterdir()})
    resume.write_json(root / "transfer_index.json", index)
    return output


def archive(source, destination):
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
        for p in source.rglob("*"):
            if p.is_file(): z.write(p, (Path(resume.RUN_NAME) / p.relative_to(source)).as_posix())


def test_95_originals_preserved_and_only_25_evaluations_missing(tmp_path):
    fixture(tmp_path)
    result = resume.validate_resume(tmp_path)
    assert result["saved_evaluations"] == 95 and result["remaining"] == 25
    assert result["physical_batch"] == 256 and result["accumulation"] == 1


@pytest.mark.parametrize("bad", ["changed_result", "missing_result", "changed_manifest", "missing_optimizer", "missing_checkpoint"])
def test_resume_rejects_missing_or_changed_original_state(tmp_path, bad):
    output = fixture(tmp_path)
    result = next(output.glob("folds/fold0/runs/*.json"))
    if bad == "changed_result": result.write_text(result.read_text() + " ")
    elif bad == "missing_result": result.unlink()
    elif bad == "changed_manifest":
        p = output / "manifest.json"; m = json.loads(p.read_text()); m["training_arguments"]["gradient_accumulation_steps"] = 16
        resume.write_json(p, m)
    elif bad == "missing_optimizer": (output / "folds/fold4/checkpoints/trainer/checkpoint-71/optimizer.pt").unlink()
    else: (output / "folds/fold4/checkpoints/trainer/checkpoint-71/trainer_state.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)): resume.validate_resume(tmp_path)


def test_relocation_changes_only_path_preserves_original_and_callback_state(tmp_path):
    fixture(tmp_path)
    p = tmp_path / "linkedin_salary/folds/fold4/checkpoints/trainer/checkpoint-71/trainer_state.json"
    original = p.read_bytes()
    before = json.loads(original)
    changed = resume.relocate_checkpoint_paths(tmp_path)
    after = json.loads(p.read_text())
    assert len(changed) == 1
    assert after.pop("best_model_checkpoint") == str(p.parent.resolve())
    before.pop("best_model_checkpoint")
    assert after == before
    backup = tmp_path / "transfer_metadata/original_trainer_states" / p.relative_to(tmp_path)
    assert backup.read_bytes() == original
    assert resume.relocate_checkpoint_paths(tmp_path) == []
    assert resume.validate_resume(tmp_path)["saved_evaluations"] == 95


@pytest.mark.parametrize("directory", [False, True])
def test_restore_zip_and_unpacked_then_restore_export_after_checkpoint_evolves(tmp_path, directory):
    initial = tmp_path / "input" / resume.RUN_NAME
    fixture(initial)
    incoming = tmp_path / "input.zip"
    archive(initial, incoming)
    root = tmp_path / "working" / resume.RUN_NAME
    resume.restore_input(initial if directory else incoming, root)
    resume.relocate_checkpoint_paths(root)
    assert resume.validate_resume(root)["remaining"] == 25
    with pytest.raises(ValueError, match="overwrite"): resume.restore_input(incoming, root)
    # A legitimate future export has a newer checkpoint and no longer contains input checkpoint-71.
    checkpoint = root / "linkedin_salary/folds/fold4/checkpoints/trainer/checkpoint-71"
    newer = checkpoint.with_name("checkpoint-142")
    assert newer.resolve().is_relative_to(root.resolve())
    checkpoint.rename(newer)
    p = newer / "trainer_state.json"
    state = json.loads(p.read_text()); state.update(epoch=2., global_step=142, best_model_checkpoint=str(newer))
    resume.write_json(p, state)
    exported = tmp_path / "newer.zip"
    archive(root, exported)
    destination = tmp_path / "next_session" / resume.RUN_NAME
    resume.restore_input(exported, destination)
    assert resume.validate_resume(destination)["saved_evaluations"] == 95


def test_partial_newer_checkpoint_is_not_silently_used_or_discarded(tmp_path):
    output = fixture(tmp_path)
    partial = output / "folds/fold4/checkpoints/trainer/checkpoint-142"
    partial.mkdir()
    (partial / "model.safetensors").write_bytes(b"incomplete write")
    with pytest.raises(ValueError, match="Incomplete checkpoint directory"):
        resume.validate_resume(tmp_path)
    assert partial.exists()


def test_input_rejects_corrupted_checkpoint_and_unsafe_zip(tmp_path):
    source = tmp_path / "source" / resume.RUN_NAME
    fixture(source)
    next(source.rglob("optimizer.pt")).write_bytes(b"damaged")
    zpath = tmp_path / "bad.zip"
    archive(source, zpath)
    with pytest.raises(ValueError, match="checkpoint file differs"):
        resume.restore_input(zpath, tmp_path / "restore" / resume.RUN_NAME)
    with zipfile.ZipFile(zpath, "a") as z: z.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        resume.restore_input(zpath, tmp_path / "another" / resume.RUN_NAME)


def test_notebook_keeps_original_git_pin_batch_no_short_smoke_no_secret_in_url():
    notebook = build_notebook()
    code = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "code")
    for i, c in enumerate(notebook["cells"]):
        if c["cell_type"] == "code": compile(c["source"], str(i), "exec")
    assert resume.SOURCE in code
    assert "GIT_CONFIG_VALUE_0" in code and "del git_env,auth" in code
    assert "x-access-token:" in code and "https://github.com/diyarh26/text-tabular-curation.git" in code
    assert "--gpu_smoke" not in code and "MICRO_BATCH = 16" not in code
    assert "autogluon-features==1.5.0" in code
    assert "--mode','prepare'" in code
    assert "--mode','supervise'" in code and "--mode','export'" in code


def test_actual_notebook_bootstrap_restores_once_handles_zip_plus_folder_and_records_runtime(tmp_path, monkeypatch):
    import subprocess
    inputs = tmp_path / "input"
    inputs.mkdir()
    source = inputs / resume.RUN_NAME
    fixture(source)
    archive(source, inputs / (resume.RUN_NAME + "_checkpoint.zip"))
    code = build_notebook()["cells"][1]["source"]
    for literal, replacement in [("/kaggle/input",inputs), ("/kaggle/working",tmp_path/"working"),
                                 ("/kaggle/temp/linkedin_vm256_resume_runtime",tmp_path/"runtime")]:
        code = code.replace(repr(literal), repr(str(replacement)))
    def hardware(command, **kwargs):
        return "0\n1\n" if "--query-gpu=index" in command else "index,name,memory.total\n0,T4,15360 MiB\n1,T4,15360 MiB\n"
    monkeypatch.setattr(subprocess,"check_output",hardware)
    namespace = {}
    exec(compile(code,"notebook_bootstrap","exec"),namespace)
    restored = tmp_path / "working" / resume.RUN_NAME
    assert resume.validate_resume(restored)["saved_evaluations"] == 95
    provenance = restored / "transfer_metadata/continuation_provenance.json"
    assert json.loads(provenance.read_text())["runtime_sha256"] == hashlib.sha256(namespace["RUNTIME_CODE"].encode()).hexdigest()
    # Running the setup cell again must preserve existing working output and the same provenance.
    original = {p:p.read_bytes() for p in restored.rglob("runs/*.json")}
    exec(compile(code,"notebook_bootstrap_again","exec"),namespace)
    assert all(p.read_bytes()==data for p,data in original.items())


def test_supervisor_separates_baseline_and_tar_processes_and_resumes_only_missing_folds(tmp_path, monkeypatch, capsys):
    output = fixture(tmp_path)
    completed = set()
    jobs = []
    clock = [0]
    class VM:
        def audit_runs(self, output, folds=range(6)):
            return [] if all(f < 4 or (f,"tar") in completed for f in folds) else ["missing"]
        def write_reports(self, output, **kwargs): return not self.audit_runs(output)
    class Process:
        def __init__(self, fold, phase): self.fold,self.phase,self.ends = fold,phase,clock[0]+11*3600
        def poll(self):
            if clock[0] < self.ends: return None
            completed.add((self.fold,self.phase))
            return 0
    def start(command, **kwargs):
        fold = int(command[command.index("--fold")+1]); phase = command[command.index("--phase")+1]
        jobs.append((fold,phase,kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        return Process(fold,phase)
    monkeypatch.setattr(resume,"verify_environment",lambda *a:(VM(),None))
    monkeypatch.setattr(resume.subprocess,"Popen",start)
    monkeypatch.setattr(resume.subprocess,"run",lambda *a,**k:SimpleNamespace(stdout="GPU"))
    monkeypatch.setattr(resume,"export_archives",lambda *a,**k:True)
    monkeypatch.setattr(resume,"stop_workers",lambda *a:None)
    monkeypatch.setattr(resume.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(resume.time,"sleep",lambda seconds:clock.__setitem__(0,clock[0]+11*3600))
    resume.supervise(tmp_path,tmp_path)
    assert [(f,p) for f,p,gpu in jobs] == [(4,"tar"),(5,"baselines"),(5,"tar")]
    assert jobs[0][2] != jobs[1][2]
    text = capsys.readouterr().out
    assert "elapsed 11.00 h" in text and "COMPLETE: all 120" in text


def test_real_bundle_build_preserves_inputs_and_restores_step71(tmp_path):
    backup = ROOT / "docs/linkedin_vm256_backup/linkedin_salary"
    supplement = ROOT / "docs/linkedin_vm256_resume_supplement.zip"
    if not backup.exists() or not supplement.exists():
        pytest.skip("Integration test requires the user's private VM checkpoint input files")
    before = resume.sha(supplement)
    bundle = build_bundle(tmp_path / "bundle.zip")
    root = tmp_path / resume.RUN_NAME
    resume.restore_input(bundle, root)
    assert resume.validate_resume(root)["saved_evaluations"] == 95
    assert resume.sha(supplement) == before
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        assert not any(f"fold{f}/checkpoints/" in name for name in z.namelist() for f in range(4))
    assert (root / "linkedin_salary/manifest.json").read_bytes() == (backup / "manifest.json").read_bytes()
