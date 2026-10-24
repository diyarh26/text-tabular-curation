"""Build the fresh LinkedIn notebook with official-source physical batch 256."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import re
import textwrap

from kaggle_linkedin_batch256 import restore_input


def cell(kind, source):
    row = dict(cell_type=kind, metadata={}, source=textwrap.dedent(source).strip()+"\n")
    if kind == "code":
        row.update(execution_count=None, outputs=[])
    return row


def build_notebook(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin the exact published project source commit")
    cells = [cell("markdown", """
        # LinkedIn — official-source training settings, physical batch 256

        **15,000 unchanged rows · 5 learners · 4 conditions · 6 official splits = 120 evaluations.**
        This is a NEW experiment; previous 16×16 results/adapters are not imported.
        Both GPUs take folds from one queue; a free GPU takes the next unstarted fold.
        Each process sees exactly one GPU. A running fold is never split across GPUs.

        Native TAR settings: physical train/eval batch **256**, accumulation **1**, float32,
        E5-small-v2, 512 tokens, LR 0.0001, weight decay 0.001, LoRA rank 16 / alpha 32 /
        dropout 0.1 / last 3 layers, maximum 50 epochs, early-stopping patience 3,
        and the official four DataLoader workers. Native loss, optimizer, scheduler and seed are retained.
        Frozen embedding inference keeps its own upstream default batch (32).

        Before running:
        1. Keep the notebook private. Select **GPU T4 ×2** and turn **Internet on**.
        2. Enable your **GH_TOKEN** secret for this notebook (private repository read access).
           Optional existing **HF_TOKEN** / **TABPFN_TOKEN** secrets are used if present;
           model-access checks determine whether any additional credential is actually needed.
        3. Use **Save Version → Save & Run All** once. Do not start an interactive duplicate.

        A real batch-256 memory/access smoke test runs on EACH GPU before the full evaluation.
        Its two optimizer updates and validation use an isolated temporary adapter that is discarded.
        Production retains the full 50-epoch schedule. An OOM stops execution with the traceback;
        there is no automatic batch, precision or sequence-length fallback.

        Live **R²** tables print in the running cell whenever a fold completes, followed by cumulative
        completed-fold results. Checkpoints include optimizer/scheduler/RNG and early-stopping state.
        Rolling results/checkpoint ZIPs refresh every 15 minutes. **No notebook-imposed runtime cutoff.**
        Kaggle's platform/session/quota limits still apply. Two T4s have separate 16 GB memories.

        To resume THIS experiment, attach its saved output as Input (ZIP or unpacked folder).
        Other LinkedIn experiments are rejected. Only 120 audited records produce a final verdict.

        The existing CSV adapter/runner remains around the official model, preprocessing and split code;
        this is not the repository's hosted-dataset `benchmark.py` entry point.

        Source checked on 2026-10-24: [native LoRA defaults](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/multabench/finetune/train_args.py),
        [Trainer](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/multabench/finetune/lora.py),
        [six splits](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/multabench/benchmark/splits.py).
        Upstream master was fe9adafbe0cd415ae5804de2afc4277eeaea8212; these training defaults and splits
        match our pinned official source ff0c9c8. Keep that pin for comparison with previous evaluations.
    """)]
    cells.append(cell("code", f"""
        import os, sys, json, shutil, stat, subprocess, zipfile
        from pathlib import Path
        EXPECTED_SOURCE = {revision!r}
        RUN_NAME = 'linkedin_full_batch256'
        RESUME_INPUT = ''  # Optional explicit ZIP or unpacked run-root folder under /kaggle/input.
        WORK = Path('/kaggle/working')
        RUN_ROOT = WORK / RUN_NAME
        SCRATCH = Path('/kaggle/temp/linkedin_batch256_runtime')
        SCRATCH.mkdir(parents=True, exist_ok=True)
        print(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'],text=True))
        gpus=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip().splitlines()
        assert len(gpus)==2 and all('T4' in name for name in gpus), 'Select GPU T4 x2.'
    """))
    bootstrap = inspect.getsource(restore_input)
    bootstrap += textwrap.dedent("""

        if RUN_ROOT.exists() and any(RUN_ROOT.iterdir()):
            assert (RUN_ROOT/'source_revision.txt').read_text().strip()==EXPECTED_SOURCE, 'Existing run source differs.'
            print('Preserving existing working output; manifest validation follows.')
        else:
            choices = list(Path('/kaggle/input').rglob(RUN_NAME+'_checkpoint.zip'))
            choices += [p.parents[1] for p in Path('/kaggle/input').rglob('manifest.json')
                        if p.parent.name=='linkedin_salary' and p.parents[1].name==RUN_NAME]
            choices = sorted(set(choices))
            if RESUME_INPUT:
                source = Path(RESUME_INPUT)
            elif len(choices)==1:
                source = choices[0]
            elif choices:
                raise RuntimeError('Several checkpoints found; set RESUME_INPUT explicitly: '+str(choices))
            else:
                source = None
            if source is not None:
                restore_input(source, RUN_ROOT, EXPECTED_SOURCE)
                print('Restored:',source)
            else:
                RUN_ROOT.mkdir(parents=True,exist_ok=True)
                print('New physical-256 experiment. No previous results imported.')
    """)
    cells.append(cell("code", bootstrap))
    cells.append(cell("code", """
        import urllib.request, urllib.error
        from kaggle_secrets import UserSecretsClient
        secrets=UserSecretsClient()
        try:
            gh_token=secrets.get_secret('GH_TOKEN')
        except Exception:
            raise RuntimeError('Enable GH_TOKEN in Add-ons > Secrets for THIS notebook, then rerun.') from None
        def github_request(url):
            return urllib.request.urlopen(urllib.request.Request(url,headers={
                'Authorization':'Bearer '+gh_token,'Accept':'application/vnd.github+json',
                'User-Agent':'linkedin-batch256'}),timeout=120)
        try:
            with github_request('https://api.github.com/repos/diyarh26/text-tabular-curation/commits/'+EXPECTED_SOURCE) as response:
                assert json.load(response)['sha']==EXPECTED_SOURCE
            archive_path=SCRATCH/'source.zip'
            with github_request('https://api.github.com/repos/diyarh26/text-tabular-curation/zipball/'+EXPECTED_SOURCE) as response:
                with archive_path.open('wb') as output:
                    shutil.copyfileobj(response,output)
        except urllib.error.HTTPError as exc:
            raise RuntimeError('GitHub HTTP %s; verify enabled GH_TOKEN read access and Internet.' % exc.code) from None
        finally:
            del gh_token
        parent=SCRATCH/'source'
        parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            names=set()
            for info in archive.infolist():
                if (info.filename in names or '\\\\' in info.filename or
                        not (parent/info.filename).resolve().is_relative_to(parent.resolve()) or
                        stat.S_ISLNK(info.external_attr>>16)):
                    raise ValueError('Unsafe source archive member')
                names.add(info.filename)
            prefix=archive.namelist()[0].split('/')[0]
            archive.extractall(parent)
        PROJECT=parent/prefix
        assert (PROJECT/'scripts/kaggle_linkedin_batch256.py').is_file()
        (PROJECT/'SOURCE_REVISION').write_text(EXPECTED_SOURCE+'\\n')
        (RUN_ROOT/'source_revision.txt').write_text(EXPECTED_SOURCE+'\\n')
        ENV={**os.environ,'MPLBACKEND':'Agg','HF_HOME':str(SCRATCH/'huggingface'),
             'TORCH_HOME':str(SCRATCH/'torch'),'XDG_CACHE_HOME':str(SCRATCH/'cache'),
             'UV_CACHE_DIR':str(SCRATCH/'uv_cache'),'TOKENIZERS_PARALLELISM':'false',
             'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','PYTHONUNBUFFERED':'1',
             'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
        for label in ('HF_TOKEN','TABPFN_TOKEN'):
            try: ENV[label]=secrets.get_secret(label)
            except Exception: print('No optional',label,'; GPU model-access checks will verify availability.')
        def run_checked(command,*,env=None,cwd=PROJECT):
            process=subprocess.Popen([str(x) for x in command],cwd=cwd,env=env or ENV,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in process.stdout: print(line,end='',flush=True)
                code=process.wait()
            except KeyboardInterrupt:
                import signal
                process.send_signal(signal.SIGINT)
                process.wait()
                raise
            if code: raise RuntimeError('Command failed; inspect traceback above. Exit code %s' % code)
    """))
    cells.append(cell("code", """
        uv=shutil.which('uv')
        if not uv:
            run_checked([sys.executable,'-m','pip','install','--quiet','uv'])
            uv=shutil.which('uv')
        assert uv
        PYTHON=SCRATCH/'venv/bin/python'
        if not PYTHON.exists(): run_checked([uv,'venv','--python','3.11',PYTHON.parent.parent])
        run_checked([uv,'pip','install','--python',PYTHON,'-c',PROJECT/'requirements/constraints.txt',
                     '-e',str(PROJECT)+'[bench,dev]','autogluon-features==1.5.0'])
        run_checked([uv,'pip','check','--python',PYTHON])
        run_checked([PYTHON,'-m','pytest','-q','tests/test_linkedin_vm.py','tests/test_full_linkedin.py',
                     'tests/test_linkedin_batch256_kaggle.py'],env={**ENV,'CUDA_VISIBLE_DEVICES':''})
        run_checked([PYTHON,'-u',PROJECT/'scripts/run_full_linkedin_vm.py',
                     '--results_root',RUN_ROOT,'--prepare_only'],env={**ENV,'CUDA_VISIBLE_DEVICES':'0'})
    """))
    cells.append(cell("code", """
        # Verify real native batch 256 on BOTH devices. Production does not reuse these smoke weights.
        output=RUN_ROOT/'linkedin_salary'
        for gpu in range(2):
            saved=output/('gpu_preflight_gpu%d.json' % gpu)
            if saved.exists() and json.loads(saved.read_text()).get('passed'):
                print('Preserving successful GPU',gpu,'preflight from this experiment.')
                continue
            run_checked([PYTHON,'-u',PROJECT/'scripts/run_full_linkedin_vm.py',
                         '--results_root',RUN_ROOT,'--gpu_smoke'],env={**ENV,'CUDA_VISIBLE_DEVICES':str(gpu)})
            shutil.copy2(output/'gpu_preflight.json',saved)
        print('Both GPUs passed. Physical train/eval batch 256; accumulation 1. No fallback.')
    """))
    cells.append(cell("code", """
        from IPython.display import display, FileLink
        print('Starting all 120 evaluations: two GPUs, shared fold queue, one GPU per fold.',flush=True)
        print('Fold R2 tables print LIVE in THIS cell. No notebook runtime cutoff.',flush=True)
        try:
            run_checked([PYTHON,'-u',PROJECT/'scripts/kaggle_linkedin_batch256.py','--results_root',RUN_ROOT])
        finally:
            for suffix in ('results','checkpoint'):
                path=WORK/(RUN_NAME+'_'+suffix+'.zip')
                if path.exists(): display(FileLink(str(path)))
        print((RUN_ROOT/'linkedin_salary/summary.txt').read_text())
    """))
    for i, entry in enumerate(cells):
        entry["id"] = f"linkedin256-{i}"
        if entry["cell_type"] == "code":
            compile(entry["source"], f"linkedin256_cell_{i}", "exec")
    return dict(nbformat=4, nbformat_minor=5, metadata={
        "kernelspec":dict(display_name="Python 3",language="python",name="python3"),
        "language_info":dict(name="python")},cells=cells)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_revision',required=True)
    args=parser.parse_args()
    path=Path(__file__).resolve().parents[1]/'notebooks/run_linkedin_full_kaggle_batch256.ipynb'
    path.write_text(json.dumps(build_notebook(args.source_revision),indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':
    main()
