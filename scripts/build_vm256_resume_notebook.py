"""Build a single audited VM checkpoint bundle and a resume-only Kaggle notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

from resume_linkedin_vm256 import CHECKPOINT_FILES, RUN_NAME, SOURCE, sha, validate_resume

ROOT = Path(__file__).resolve().parents[1]


def build_bundle(destination, backup=None, supplement=None):
    backup = Path(backup or ROOT / "docs/linkedin_vm256_backup/linkedin_salary")
    supplement = Path(supplement or ROOT / "docs/linkedin_vm256_resume_supplement.zip")
    files = {}
    for p in backup.rglob("*"):
        if p.is_file() and "checkpoints" not in p.relative_to(backup).parts:
            files["linkedin_salary/" + p.relative_to(backup).as_posix()] = p.read_bytes()
    with zipfile.ZipFile(supplement) as archive:
        if archive.testzip() is not None:
            raise ValueError("Supplement ZIP integrity failed")
        for name in archive.namelist():
            if name.startswith("folds/fold4/") or name.startswith("folds/fold5/") or name == "manifest.json":
                target = "linkedin_salary/" + name
            elif name.startswith("transfer_metadata/"):
                target = name
            else:
                raise ValueError("Unexpected supplement member: " + name)
            content = archive.read(name)
            if target in files and files[target] != content:
                raise ValueError("Original backup and supplement differ: " + target)
            files[target] = content
    manifest = json.loads(files["linkedin_salary/manifest.json"])
    if manifest["source_revision"] != SOURCE:
        raise ValueError("Unexpected VM source revision")
    files["source_revision.txt"] = (SOURCE + "\n").encode()
    original = {name: hashlib.sha256(content).hexdigest() for name, content in files.items() if "/runs/" in name}
    checkpoint = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()
                  if name.startswith("linkedin_salary/folds/fold4/checkpoints/")}
    if len(original) != 95 or not all(any(p.endswith('/'+n) for p in checkpoint) for n in CHECKPOINT_FILES):
        raise ValueError("Expected the audited 95 results plus full fold-4 checkpoint")
    index = dict(format="vm256_transfer_v1", source_revision=SOURCE,
        manifest_sha256=hashlib.sha256(files["linkedin_salary/manifest.json"]).hexdigest(),
        original_run_sha256=original, initial_checkpoint_sha256=checkpoint,
        supplement_sha256=sha(supplement), skipped_completed_fold_weights=[0, 1, 2, 3])
    files["transfer_index.json"] = (json.dumps(index, indent=2) + "\n").encode()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(RUN_NAME + "/" + name, content)
    return destination


def cell(kind, source):
    item = dict(cell_type=kind, metadata={}, source=textwrap.dedent(source).strip()+"\n")
    if kind == "code": item.update(execution_count=None, outputs=[])
    return item


def build_notebook():
    runtime = (ROOT / "scripts/resume_linkedin_vm256.py").read_text(encoding="utf-8")
    cells = [cell("markdown", """
        # Resume the VM LinkedIn run — physical batch 256

        **Preserve 95 completed evaluations. Resume fold 4 after epoch 1 / update 71, then finish fold 5.**
        Original VM source: `bfe0ba3b846cb8bd719d43a586dc4d25b36088cd`.
        Physical train/eval batch 256, accumulation 1, float32, official 50-epoch cap and patience 3.
        All original data, splits, model/LoRA settings, package versions and VM loader behavior remain unchanged.
        This continues the VM v1 experiment; it does not migrate it to the newer Kaggle v2 protocol.

        1. Create a **private Kaggle Dataset** from `linkedin_vm256_resume_checkpoint.zip` provided with this
           notebook, and attach it using **Add Input**. ZIP and unpacked inputs are supported.
        2. Keep this notebook private. Enable **Internet**, select **GPU T4 ×2** (one GPU also works),
           and enable your existing **GH_TOKEN** secret for this notebook.
           Optional existing **HF_TOKEN** / **TABPFN_TOKEN** credentials are used if present.
        3. Use **Save Version → Save & Run All** once. Initial validation must retain all 95 original results
           and find the fold-4 checkpoint. There is **no fresh-run fallback**.

        TAR runs in a fresh process, with no earlier baseline model fitted in that process.
        On two GPUs, fold 4 TAR and fold 5 baselines start concurrently; fold 5 TAR gets another fresh process.
        A single fold is never split across devices. Batch/precision/sequence length never change automatically.
        The previous Kaggle attempt ran out of memory: **this notebook does not promise batch 256 will fit**.
        There is no short training smoke test; the actual resumed TAR is monitored. An OOM stops and exports saved work.

        Live R² tables print after completed folds. Results/checkpoint ZIPs refresh every 15 minutes.
        There is no notebook-imposed time cutoff; Kaggle platform/session/quota limits still apply.
        Only the best-checkpoint filesystem path is relocated in the working copy; original JSON is preserved.
        Completed run JSONs and the experiment manifest must remain byte-identical to the VM originals.
    """)]
    cells.append(cell("code", f"""
        import os, sys, json, shutil, subprocess, zipfile, runpy, hashlib
        from pathlib import Path
        SOURCE = {SOURCE!r}
        RUN_NAME = {RUN_NAME!r}
        RESUME_INPUT = ''  # Optional explicit ZIP or unpacked prepared bundle root under /kaggle/input.
        WORK = Path('/kaggle/working')
        RUN_ROOT = WORK/RUN_NAME
        SCRATCH = Path('/kaggle/temp/linkedin_vm256_resume_runtime')
        SCRATCH.mkdir(parents=True,exist_ok=True)
        print(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'],text=True))
        GPUS = [int(s) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split()]
        assert 1<=len(GPUS)<=2,'Select one or two NVIDIA GPUs.'
        RUNTIME = SCRATCH/'resume_linkedin_vm256.py'
        RUNTIME_CODE = {runtime!r}
        RUNTIME.write_text(RUNTIME_CODE,encoding='utf-8')
        api = runpy.run_path(str(RUNTIME),run_name='resume_bootstrap')
        if (RUN_ROOT/'transfer_index.json').is_file():
            print('Validating existing working output; attached input will not overwrite it.')
        else:
            choices=list(Path('/kaggle/input').rglob(RUN_NAME+'_checkpoint.zip'))
            unpacked=[p.parent for p in Path('/kaggle/input').rglob('transfer_index.json')
                      if (p.parent/'linkedin_salary/manifest.json').is_file()]
            # Some uploads expose both the ZIP and its adjacent extracted folder.
            choices += [p for p in unpacked if not any(p.resolve()==(z.parent/RUN_NAME).resolve() for z in choices)]
            choices=sorted(set(choices))
            selected=Path(RESUME_INPUT) if RESUME_INPUT else choices[0] if len(choices)==1 else None
            assert selected is not None,'Attach exactly one prepared VM resume input, or set RESUME_INPUT: '+str(choices)
            api['restore_input'](selected,RUN_ROOT)
        print('VM RESUME VERIFIED:',json.dumps(api['validate_resume'](RUN_ROOT),indent=2),flush=True)
        provenance=RUN_ROOT/'transfer_metadata/continuation_provenance.json'
        record=dict(original_source=SOURCE,runtime_sha256=hashlib.sha256(RUNTIME_CODE.encode()).hexdigest(),
                    physical_batch=256,accumulation=1,tar_uses_fresh_process=True)
        if provenance.exists():
            assert json.loads(provenance.read_text())==record,'Continuation harness differs; preserve output and investigate.'
        else: api['write_json'](provenance,record)
    """))
    cells.append(cell("code", """
        import base64
        from kaggle_secrets import UserSecretsClient
        secrets=UserSecretsClient()
        try: gh_token=secrets.get_secret('GH_TOKEN')
        except Exception: raise RuntimeError('Enable GH_TOKEN for THIS notebook in Add-ons > Secrets.') from None
        PROJECT=SCRATCH/'original_source'
        ENV={**os.environ,'MPLBACKEND':'Agg','HF_HOME':str(SCRATCH/'huggingface'),
             'TORCH_HOME':str(SCRATCH/'torch'),'UV_CACHE_DIR':str(SCRATCH/'uv_cache'),
             'XDG_CACHE_HOME':str(SCRATCH/'cache'),'TOKENIZERS_PARALLELISM':'false',
             'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','PYTHONUNBUFFERED':'1',
             'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
        for label in ('HF_TOKEN','TABPFN_TOKEN'):
            try: ENV[label]=secrets.get_secret(label)
            except Exception: print('No optional',label,'; access check will verify availability.')
        def run_checked(command,*,env=None,cwd=None):
            p=subprocess.Popen([str(x) for x in command],cwd=cwd or PROJECT,env=env or ENV,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in p.stdout: print(line,end='',flush=True)
                code=p.wait()
            except KeyboardInterrupt:
                import signal
                p.send_signal(signal.SIGINT); p.wait(); raise
            if code: raise RuntimeError('Command failed; inspect output above. Exit code '+str(code))
        try:
            head=subprocess.run(['git','rev-parse','--verify','HEAD'],cwd=PROJECT,capture_output=True,text=True) if (PROJECT/'.git').exists() else None
            if head is not None and head.returncode==0:
                assert head.stdout.strip()==SOURCE,'Existing source checkout differs; preserve it and investigate.'
            else:
                PROJECT.mkdir(parents=True,exist_ok=True)
                if not (PROJECT/'.git').exists(): run_checked(['git','init',PROJECT],cwd=SCRATCH)
                # Credentials are passed only in the fetch process environment, never URLs, arguments or Git config.
                auth=base64.b64encode(('x-access-token:'+gh_token).encode()).decode()
                git_env={**ENV,'GIT_TERMINAL_PROMPT':'0','GIT_TRACE':'0','GIT_CURL_VERBOSE':'0','GIT_TRACE_CURL':'0',
                    'GIT_CONFIG_COUNT':'2','GIT_CONFIG_KEY_0':'http.https://github.com/.extraheader',
                    'GIT_CONFIG_VALUE_0':'AUTHORIZATION: basic '+auth,
                    'GIT_CONFIG_KEY_1':'credential.helper','GIT_CONFIG_VALUE_1':''}
                try: run_checked(['git','fetch','--depth','1','--no-tags','https://github.com/diyarh26/text-tabular-curation.git',SOURCE],env=git_env)
                finally: del git_env,auth
                run_checked(['git','checkout','--detach','FETCH_HEAD'])
        finally: del gh_token
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()==SOURCE
        print('Original VM Git source verified:',SOURCE)
    """))
    cells.append(cell("code", """
        uv=shutil.which('uv')
        if not uv:
            run_checked([sys.executable,'-m','pip','install','--quiet','uv'],cwd=SCRATCH)
            uv=shutil.which('uv')
        assert uv
        PYTHON=SCRATCH/'venv/bin/python'
        if not PYTHON.exists(): run_checked([uv,'venv','--python','3.11',PYTHON.parent.parent])
        run_checked([uv,'pip','install','--python',PYTHON,'-c',PROJECT/'requirements/constraints.txt',
                     '-e',str(PROJECT)+'[bench,dev]','autogluon-features==1.5.0'])
        run_checked([uv,'pip','check','--python',PYTHON])
        run_checked([PYTHON,'-u',RUNTIME,'--mode','prepare','--root',RUN_ROOT,'--project',PROJECT],
                     env={**ENV,'CUDA_VISIBLE_DEVICES':str(GPUS[0])})
        for gpu in GPUS:
            run_checked([PYTHON,'-c',
                "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; "
                "assert torch.ones(4,device='cuda').sum().item()==4; print(torch.__version__,torch.cuda.get_device_name(0)); "
                "from tt_curation.runner import _model_class; [_model_class(m) for m in ['tabm','cat','light','tabpfnv2','tabpfnv2p5']]; print('Five learner imports verified')"],
                env={**ENV,'CUDA_VISIBLE_DEVICES':str(gpu)})
        # Model-access probe runs in a separate process and exits before any TAR worker starts.
        probe='\\n'.join([
            'import gc,numpy as np,torch',
            'from tabpfn import TabPFNRegressor',
            'from tabpfn.constants import ModelVersion',
            'X=np.random.default_rng(42).normal(size=(32,4)); y=X[:,0]',
            "for v in ['v2','v2.5']:",
            "    model=TabPFNRegressor.create_default_for_version(version=ModelVersion(v),device='cuda',random_state=42,ignore_pretraining_limits=True)",
            '    model.fit(X,y); assert np.isfinite(model.predict(X[:2])).all()',
            "    print('Regression checkpoint access:',v,flush=True)",
            '    del model; gc.collect(); torch.cuda.empty_cache()',
        ])
        run_checked([PYTHON,'-u','-c',probe],env={**ENV,'CUDA_VISIBLE_DEVICES':str(GPUS[0])})
        print('Original manifest matched. Actual production TAR will test memory; no short smoke-test claim.',flush=True)
    """))
    cells.append(cell("code", """
        from IPython.display import display,FileLink
        print('Preserving all 95 originals. Resume fold 4 from checkpoint; run only missing fold-5 work.',flush=True)
        print('Physical batch 256; accumulation 1; fresh process for TAR. No automatic fallback or time cutoff.',flush=True)
        try:
            run_checked([PYTHON,'-u',RUNTIME,'--mode','supervise','--root',RUN_ROOT,'--project',PROJECT,
                         '--gpus',','.join(map(str,GPUS))],env={**ENV,'CUDA_VISIBLE_DEVICES':str(GPUS[0])})
        finally:
            try: run_checked([PYTHON,'-u',RUNTIME,'--mode','export','--root',RUN_ROOT])
            finally:
                for suffix in ('results','checkpoint'):
                    p=WORK/(RUN_NAME+'_'+suffix+'.zip')
                    if p.exists(): display(FileLink(str(p)))
        print((RUN_ROOT/'linkedin_salary/summary.txt').read_text())
    """))
    for i, c in enumerate(cells):
        c["id"] = f"vm256-resume-{i}"
        if c["cell_type"] == "code": compile(c["source"], f"cell{i}", "exec")
    return dict(nbformat=4, nbformat_minor=5, metadata={
        "kernelspec":dict(display_name="Python 3",language="python",name="python3"),
        "language_info":dict(name="python")},cells=cells)


if __name__ == "__main__":
    bundle = build_bundle(ROOT / "artifacts/linkedin_vm256_resume" / (RUN_NAME + "_checkpoint.zip"))
    notebook = ROOT / "notebooks/resume_linkedin_vm256_kaggle.ipynb"
    notebook.write_text(json.dumps(build_notebook(), indent=1)+"\n", encoding="utf-8")
    print(bundle)
    print(notebook)
