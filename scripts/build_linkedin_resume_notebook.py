"""Resume the validated 115-result LinkedIn checkpoint without changing its experiment."""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import shutil
import stat
import textwrap
import zipfile

EXPECTED_SOURCE = "b75bdbb835d2ec0da13e076fce5249933879127b"
LOSS_VERSION = "ce_sum_over_effective_batch_items_v1"


def restore_input(source, root):
    """Restore either a ZIP or Kaggle's unpacked dataset, preserving existing working output."""
    source, root = Path(source), Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Existing working output must be validated and reused; do not overwrite it")
    if source.is_dir():
        assert (source / "linkedin_salary/manifest.json").is_file(), "Select the checkpoint root folder"
        assert not any(path.is_symlink() for path in source.rglob("*")), "Unexpected checkpoint symlink"
        shutil.copytree(source, root, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(source) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist())), "Duplicate archive members"
            for info in archive.infolist():
                destination = (root.parent / info.filename).resolve()
                if ("\\" in info.filename or not destination.is_relative_to(root.resolve())
                        or stat.S_ISLNK(info.external_attr >> 16)):
                    raise ValueError("Unexpected checkpoint member: " + info.filename)
            assert root.name + "/linkedin_salary/manifest.json" in archive.namelist(), "Wrong checkpoint ZIP"
            archive.extractall(root.parent)


def validate_resume(root):
    """Require the original corrected experiment and 115 completed evaluations before spending GPU time."""
    root = Path(root)
    output = root / "linkedin_salary"
    manifest = json.loads((output / "manifest.json").read_text())
    assert (root / "source_revision.txt").read_text().strip() == EXPECTED_SOURCE
    assert manifest["source_revision"] == EXPECTED_SOURCE, "Use this experiment's original source"
    assert manifest["dataset"] == "linkedin_salary"
    assert manifest["micro_batch"] == 16 and manifest["effective_batch"] == 256
    assert manifest["accumulation_loss"] == LOSS_VERSION
    assert manifest["tar_max_epochs"] == 50 and manifest["precision"] == "float32"
    assert manifest["csv_sha256_lf"] == "8880f84563535c60eda01be5b52010dea6f80e318f228ade00166d7cb66d57fa"
    names = {"tabm": "TabM", "cat": "CatBoost", "light": "LightGBM",
             "tabpfnv2": "TabPFN-v2", "tabpfnv2p5": "TabPFN-v2p5"}
    conditions = ["structured", "unstructured", "joint_frozen", "joint_tar"]
    expected = {(m, c, f) for m in names for c in conditions for f in range(6)}
    rows = []
    seen = set()
    for path in output.glob("folds/fold*/runs/*.json"):
        row = json.loads(path.read_text())
        identity = (row["short_name"], row["condition"], row["fold"])
        assert identity in expected and identity not in seen, "Unexpected or duplicate result"
        seen.add(identity)
        assert row["model"] == names[row["short_name"]] and row["dataset"] == "linkedin_salary"
        assert row["status"] == "ok" and row["task"] == "regression"
        assert row["size"] == "10k" and (row["n_train"], row["n_test"]) == (10000, 5000)
        assert row["multabench_commit"] == "ff0c9c8b40a2085a06a7771190e53a261341802f"
        assert all(math.isfinite(row[name]) for name in ("test_score", "test_error", "r2"))
        encoder = None if row["condition"] == "structured" else (
            "e5-small-tar" if row["condition"] == "joint_tar" else "e5-small")
        assert row["text_encoder"] == encoder
        assert path.name == f"{row['short_name']}_{row['condition']}_10k_fold{row['fold']}.json"
        assert path.parent.parent.name == f"fold{row['fold']}"
        rows.append(row)
    missing = expected - seen
    assert missing <= {(m, "joint_tar", 5) for m in names}, "A previously completed evaluation is missing"
    assert 115 <= len(rows) <= 120
    result = dict(n_results=len(rows), missing_evaluations=len(missing), source=EXPECTED_SOURCE,
                  micro_batch=16, accumulation=16, checkpoint_epoch=None)
    if missing:
        cache = output / "folds/fold5/checkpoints"
        marker = json.loads((cache / "accumulation_loss.json").read_text())
        assert marker == dict(version=LOSS_VERSION, micro_batch=16, effective_batch=256)
        checkpoints = list((cache / "trainer").glob("checkpoint-*/trainer_state.json"))
        assert checkpoints, "Saved fold-5 Trainer state is required; refusing to restart TAR from scratch"
        state_path = max(checkpoints, key=lambda path: int(path.parent.name.split("-")[-1]))
        state = json.loads(state_path.read_text())
        assert state_path.parent.name == f"checkpoint-{state['global_step']}"
        assert 0 < state["epoch"] <= 50
        for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"):
            path = state_path.parent / name
            assert path.is_file() and path.stat().st_size > 0, "Missing checkpoint state: " + name
        result.update(checkpoint_epoch=state["epoch"], checkpoint=str(state_path.parent))
    return result


REPORT_PROGRAM = r'''
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path('scripts').resolve()))
from run_full_linkedin import audit_runs, collect_runs
from kaggle_linkedin import export_archives
root=Path(sys.argv[1]); output=root/'linkedin_salary'
runs=collect_runs(output)
names=['CatBoost','LightGBM','TabM','TabPFN-v2','TabPFN-v2p5']
conditions=['structured','unstructured','joint_frozen','joint_tar']
grid={(m,c) for m in names for c in conditions}
folds=[int(f) for f,g in runs[runs.status.eq('ok')].groupby('fold')
       if len(g)==20 and set(zip(g.model,g.condition))==grid]
selected=runs[runs.fold.isin(folds)]
table=selected.pivot_table(index='model',columns='condition',values='r2').reindex(names)[conditions]
table['Joint delta']=table.joint_frozen-table[['structured','unstructured']].max(axis=1)
table['TAR delta']=table.joint_tar-table.joint_frozen
joint=int(table['Joint delta'].gt(0).sum()); tar=int(table['TAR delta'].gt(0).sum())
complete=len(runs)==120 and not audit_runs(output)
text=('FINAL: 120/120 evaluations audited.' if complete else f'PRELIMINARY: {len(runs)}/120 saved evaluations.')+'\n'
text+=f'R2 curation metric; matching completed folds {folds}; raw means/deltas.\n'
text+=table.to_string(float_format=lambda x:f'{x:.6f}')+'\n'
text+=f'Joint Signal {joint}/5; TAR Gain {tar}/5; need >=3/5 independently.\n'
if complete:
    text+=('PASSES' if joint>=3 and tar>=3 else 'DOES NOT PASS')+' the course numerical criteria.\n'
    table.to_csv(output/'curation_verdict_r2.csv')
    (output/'summary_r2.txt').write_text(text,encoding='utf-8')
    (output/'curation_summary_r2.json').write_text(json.dumps(dict(metric='r2',folds=folds,joint_learners=joint,
         tar_learners=tar,accepted=joint>=3 and tar>=3),indent=2)+'\n')
else:
    text+='No final verdict; preserve the checkpoint and resume the same experiment.\n'
    for name in ['curation_verdict_r2.csv','summary_r2.txt','curation_summary_r2.json']:
        (output/name).unlink(missing_ok=True)
(output/'progress_r2.txt').write_text(text,encoding='utf-8')
print(text,flush=True)
if len(sys.argv)>2 and sys.argv[2]=='export':
    export_archives(root)
'''


def cell(kind, source):
    value = dict(cell_type=kind, metadata={}, source=textwrap.dedent(source).strip() + "\n")
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build_notebook():
    cells = [cell("markdown", """
        # Finish the existing LinkedIn run: resume only

        This notebook requires your `linkedin_full_checkpoint.zip` (or its unpacked contents).
        It preserves the **115 completed evaluations**, resumes **fold 5 TAR at saved epoch 30**,
        and finishes the five remaining downstream evaluations. If an attached newer checkpoint
        already contains more results, those are retained. There is **no fresh-experiment fallback**.

        Keep **micro-batch 16, accumulation 16, effective batch 256**, float32, the original 50-epoch
        cap and early stopping. Original runtime source: `b75bdbb835d2ec0da13e076fce5249933879127b`.
        Model, optimizer, scheduler, random state and early-stopping callback state are restored by
        the original tested Trainer helper. No checkpoint migration or hyperparameter change occurs.

        1. If the original Kaggle session is still running, let it finish instead of starting a duplicate.
        2. Create a **private Kaggle Dataset** from `linkedin_full_checkpoint.zip`; upload the full ZIP.
           Attach that dataset to this notebook using **Add Input**. ZIP and unpacked inputs are supported.
        3. Select **GPU T4 x2**, enable **Internet**, and enable the **GH_TOKEN** secret for this notebook.
           Enable optional **TABPFN_TOKEN** / **HF_TOKEN** if needed for model access.
        4. Use **Save Version > Save & Run All**. Initial validation must show 115 results retained and
           fold-5 checkpoint epoch 30 for the currently downloaded archive. A different compatible snapshot
           may show a later epoch/result count. Missing or incompatible input stops execution.
        5. At completion, download `linkedin_full_results.zip` and retain `linkedin_full_checkpoint.zip`.
           The final correct R² table is printed and stored in `summary_r2.txt` / `curation_verdict_r2.csv`.

        Finished folds are skipped. One worker exits after finding its folds complete; the other resumes
        fold 5. It is normal for only one GPU to remain busy. There is no notebook-imposed time cutoff;
        Kaggle session/quota limits still apply. Rolling checkpoint ZIPs refresh every 15 minutes.
        Setup performs small environment/model-access checks; it does not rerun the 20-case toy experiment.
    """), cell("code", f"""
        import os, sys, time, json, math, stat, subprocess, shutil, zipfile
        from pathlib import Path
        EXPECTED_SOURCE = {EXPECTED_SOURCE!r}
        LOSS_VERSION = {LOSS_VERSION!r}
        MICRO_BATCH = 16
        RESUME_INPUT = ''  # Optional explicit ZIP path OR unpacked checkpoint-root folder under /kaggle/input.
        WORK = Path('/kaggle/working')
        RUN_ROOT = WORK / 'linkedin_full'
        SCRATCH = Path('/kaggle/temp/linkedin_runtime')  # Preserve original checkpoint path conventions.
        SCRATCH.mkdir(parents=True, exist_ok=True)
        print(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'],text=True))
        gpus=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip().splitlines()
        assert len(gpus)==2 and all('T4' in name for name in gpus), 'Select GPU T4 x2.'
    """)]
    cells.append(cell("code", inspect.getsource(restore_input) + "\n" + inspect.getsource(validate_resume) + "\n" + textwrap.dedent("""
        if (RUN_ROOT / 'linkedin_salary/manifest.json').exists():
            print('Validating existing working output; attached inputs will not overwrite it.')
        else:
            inputs=Path('/kaggle/input')
            if RESUME_INPUT:
                selected=Path(RESUME_INPUT)
            else:
                candidates=sorted(inputs.rglob('linkedin_full_checkpoint.zip'))
                if not candidates:
                    candidates=sorted({path.parent.parent for path in inputs.rglob('manifest.json')
                        if path.parent.name=='linkedin_salary' and (path.parent.parent/'source_revision.txt').exists()})
                assert len(candidates)==1, 'Attach exactly one LinkedIn checkpoint input, or set RESUME_INPUT. Found: '+str(candidates)
                selected=candidates[0]
            restore_input(selected,RUN_ROOT)
        resume_state=validate_resume(RUN_ROOT)
        print('RESUME VERIFIED:',json.dumps(resume_state,indent=2),flush=True)
    """)))
    cells.append(cell("code", """
        import urllib.request, urllib.error
        from kaggle_secrets import UserSecretsClient
        secrets=UserSecretsClient()
        try:
            gh_token=secrets.get_secret('GH_TOKEN')
        except Exception:
            raise RuntimeError('Enable GH_TOKEN for THIS notebook in Secrets.') from None
        def github_request(url):
            return urllib.request.urlopen(urllib.request.Request(url,headers={
                'Authorization':'Bearer '+gh_token,'Accept':'application/vnd.github+json',
                'User-Agent':'linkedin-checkpoint-resume'}),timeout=120)
        try:
            with github_request('https://api.github.com/repos/diyarh26/text-tabular-curation/commits/'+EXPECTED_SOURCE) as response:
                source_sha=json.load(response)['sha']
            assert source_sha==EXPECTED_SOURCE
            source_zip=SCRATCH/'source.zip'
            with github_request('https://api.github.com/repos/diyarh26/text-tabular-curation/zipball/'+source_sha) as response:
                with source_zip.open('wb') as output:
                    shutil.copyfileobj(response,output)
        finally:
            del gh_token
        project_parent=SCRATCH/'source'
        project_parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(source_zip) as archive:
            for info in archive.infolist():
                if ('\\\\' in info.filename or stat.S_ISLNK(info.external_attr>>16) or
                        not (project_parent/info.filename).resolve().is_relative_to(project_parent.resolve())):
                    raise ValueError('Unsafe repository archive member')
            prefix=archive.namelist()[0].split('/')[0]
            archive.extractall(project_parent)
        PROJECT=project_parent/prefix
        (PROJECT/'SOURCE_REVISION').write_text(source_sha+'\\n')
        ENV={**os.environ,'MPLBACKEND':'Agg','HF_HOME':str(SCRATCH/'huggingface'),
             'TORCH_HOME':str(SCRATCH/'torch'),'XDG_CACHE_HOME':str(SCRATCH/'cache'),
             'UV_CACHE_DIR':str(SCRATCH/'uv_cache'),'TOKENIZERS_PARALLELISM':'false',
             'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','PYTHONUNBUFFERED':'1',
             'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
        for name in ('HF_TOKEN','TABPFN_TOKEN'):
            try:
                ENV[name]=secrets.get_secret(name)
            except Exception:
                print('No optional',name,'; model-access check will verify availability.')
        def run_checked(command,*,env=None,cwd=PROJECT):
            process=subprocess.Popen([str(item) for item in command],cwd=cwd,env=env or ENV,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True)
                code=process.wait()
            except KeyboardInterrupt:
                import signal
                process.send_signal(signal.SIGINT)
                process.wait()
                raise
            if code:
                raise RuntimeError('Command failed; inspect preceding output. Exit code '+str(code))
        print('Restored exact original source:',source_sha)
    """))
    cells.append(cell("code", """
        uv=shutil.which('uv')
        if not uv:
            run_checked([sys.executable,'-m','pip','install','--quiet','uv'])
            uv=shutil.which('uv')
        assert uv
        VENV=SCRATCH/'venv'; PYTHON=VENV/'bin/python'
        if not PYTHON.exists():
            run_checked([uv,'venv','--python','3.11',VENV])
        run_checked([uv,'pip','install','--python',PYTHON,'-c',PROJECT/'requirements/constraints.txt',
                     '-e',str(PROJECT)+'[bench,dev]','autogluon-features==1.5.0'])
        for gpu in range(2):
            run_checked([PYTHON,'-c',
                "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; "
                "print('CUDA verified:',torch.__version__,torch.cuda.get_device_name(0)); "
                "from tt_curation.runner import _model_class; "
                "[_model_class(m) for m in ['tabm','cat','light','tabpfnv2','tabpfnv2p5']]"],
                env={**ENV,'CUDA_VISIBLE_DEVICES':str(gpu)})
        # Actual pinned Trainer checks on tiny CPU models; no toy GPU training grid.
        run_checked([PYTHON,'-m','pytest','-q','tests/test_full_linkedin.py','-m','bench'],
                    env={**ENV,'CUDA_VISIBLE_DEVICES':''})
        # The original worker checks all source/data/package settings against the saved manifest.
        run_checked([PYTHON,'-u',PROJECT/'scripts/run_full_linkedin.py','--results_root',RUN_ROOT,
                     '--micro_batch','16','--prepare_only'],env={**ENV,'CUDA_VISIBLE_DEVICES':'0'})
    """))
    cells.append(cell("code", """
        if resume_state['missing_evaluations']:
            model_check='\\n'.join([
                'import gc, numpy as np, torch',
                'from tabpfn import TabPFNRegressor',
                'from tabpfn.constants import ModelVersion',
                'rng=np.random.default_rng(42); X=rng.normal(size=(64,4)); y=X[:,0]',
                "for version in ['v2','v2.5']:",
                "    model=TabPFNRegressor.create_default_for_version(version=ModelVersion(version),device='cuda',random_state=42,ignore_pretraining_limits=True)",
                '    model.fit(X,y); prediction=model.predict(X[:2]); assert np.isfinite(prediction).all()',
                "    print('Regression checkpoint access verified:',version,flush=True)",
                '    del model; gc.collect(); torch.cuda.empty_cache()',
            ])
            run_checked([PYTHON,'-u','-c',model_check],env={**ENV,'CUDA_VISIBLE_DEVICES':'0'})
        print('Checkpoint/environment preflight complete. Original batch and source retained.',flush=True)
    """))
    cells.append(cell("code", f"REPORT_PROGRAM = {REPORT_PROGRAM!r}\n" + textwrap.dedent("""
        from IPython.display import display, FileLink
        run_checked([PYTHON,'-u','-c',REPORT_PROGRAM,RUN_ROOT])
        print('Resuming original run: completed evaluations will be skipped; fold 5 is the remaining work.',flush=True)
        print('Batch 16 x 16. No additional wall-time cutoff. One GPU may be idle.',flush=True)
        try:
            run_checked([PYTHON,'-u',PROJECT/'scripts/kaggle_linkedin.py','--results_root',RUN_ROOT,
                         '--micro_batch','16'])
        finally:
            try:
                run_checked([PYTHON,'-u','-c',REPORT_PROGRAM,RUN_ROOT,'export'])
            finally:
                for filename in ('linkedin_full_results.zip','linkedin_full_checkpoint.zip'):
                    path=WORK/filename
                    if path.exists():
                        display(FileLink(str(path)))
                print('Download results ZIP; retain checkpoint ZIP for resumption.',flush=True)
    """)))
    for index, entry in enumerate(cells):
        entry["id"] = f"linkedin-resume-{index}"
        if entry["cell_type"] == "code":
            compile(entry["source"], f"linkedin_resume_cell_{index}", "exec")
    return dict(nbformat=4, nbformat_minor=5, metadata={
        "kernelspec": dict(display_name="Python 3", language="python", name="python3"),
        "language_info": dict(name="python")}, cells=cells)


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "notebooks/resume_linkedin_remaining_kaggle.ipynb"
    destination.write_text(json.dumps(build_notebook(), indent=1) + "\n", encoding="utf-8")
    print(destination)
