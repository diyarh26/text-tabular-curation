"""Generate the private-repo, two-T4 LinkedIn notebook without bundling credentials/data."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    cells = [cell("markdown", """
        # LinkedIn: full MulTaBench evaluation on two T4 GPUs

        **15,000 prepared rows · five learners · four conditions · six folds = 120 evaluations.**
        GPU 0 runs folds 0/2/4; GPU 1 runs 1/3/5. Each process sees one GPU.
        Approved Kaggle memory adjustment: micro-batch 16, accumulation 16, effective batch 256.
        Maximum 50 epochs with early stopping. Model, learning rate, LoRA settings and float32 remain unchanged.
        Set `MICRO_BATCH = 256` to use the reference physical batch on a GPU where it fits.

        Before starting:
        1. Keep this notebook private. Set **Accelerator → GPU T4 x2**, and **Internet → On**.
        2. In **Add-ons → Secrets**, add `GH_TOKEN` with read access to the private
           `diyarh26/text-tabular-curation` repository and enable it for this notebook.
           Do not paste a token into a cell. For TabPFN-v2.5, accept the model license at
           https://ux.priorlabs.ai and add its account API key as the `TABPFN_TOKEN` secret.
           `HF_TOKEN` is a separate, optional credential.
        3. Use **Save Version → Save & Run All** for a saved background run. Do not also launch an interactive
           full run at the same time. Check your account's remaining GPU quota first.

        The supervisor stops training after a conservative 10-hour budget from the first cell,
        then exports reports and epoch checkpoints to `/kaggle/working`. Multiple sessions may be needed.
        A platform termination before export can still lose current-session work.
        **Two T4s provide two separate 16 GB devices; GPU memory is not pooled.**

        Resume: attach the previous version's notebook output using **Add Input**, then rerun this notebook.
        Its checkpoint ZIP is detected below. The saved source revision and experiment manifest are checked;
        completed evaluations are skipped, and TAR resumes from its saved epoch checkpoint.
        Only a fully audited set of 120 evaluations produces a final verdict.

        References: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks),
        [GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage),
        [T4 x2 announcement](https://www.kaggle.com/discussions/product-announcements/735239).
    """), cell("code", """
        import os, sys, time, json, subprocess, shutil, zipfile, re
        from pathlib import Path
        SESSION_START = globals().get('SESSION_START', time.monotonic())
        SESSION_WORK_HOURS = 10
        MICRO_BATCH = 16  # Approved: accumulation 16, effective batch 256. Reference physical batch: 256.
        # Leave blank on first run; on resume auto-detect one checkpoint under /kaggle/input.
        # If you attach several older versions, specify the desired checkpoint ZIP explicitly.
        RESUME_ZIP = ""
        REPO_REF = "main"  # On resume, the saved exact commit overrides this.
        WORK = Path('/kaggle/working')
        RUN_ROOT = WORK / 'linkedin_full'
        SCRATCH = Path('/kaggle/temp/linkedin_runtime')
        SCRATCH.mkdir(parents=True, exist_ok=True)
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        print(subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                       '--format=csv'], text=True))
        gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
                                            '--format=csv,noheader,nounits'], text=True).strip().splitlines()
        assert len(gpu_lines) == 2 and all('T4' in row for row in gpu_lines), 'Select GPU T4 x2 first.'
        print('Free working space: %.1f GiB' % (shutil.disk_usage(WORK).free / 2**30))
        print('CPU cores:', os.cpu_count())
    """), cell("code", """
        candidates = sorted(Path('/kaggle/input').rglob('linkedin_full_checkpoint.zip'))
        if RESUME_ZIP:
            checkpoint_zip = Path(RESUME_ZIP)
        elif len(candidates) == 1:
            checkpoint_zip = candidates[0]
        elif len(candidates) > 1:
            raise RuntimeError('Multiple checkpoints found. Set RESUME_ZIP to the desired version: ' + str(candidates))
        else:
            checkpoint_zip = None
        if checkpoint_zip and not (RUN_ROOT / 'source_revision.txt').exists():
            with zipfile.ZipFile(checkpoint_zip) as archive:
                # Validate all names before extracting any file.
                for info in archive.infolist():
                    destination = (WORK / info.filename).resolve()
                    if not destination.is_relative_to(RUN_ROOT.resolve()):
                        raise ValueError('Unexpected checkpoint member: ' + info.filename)
                archive.extractall(WORK)
            print('Restored', checkpoint_zip)
        elif (RUN_ROOT / 'source_revision.txt').exists():
            print('Reusing existing working output.')
        else:
            print('Starting a fresh experiment.')
        revision_file = RUN_ROOT / 'source_revision.txt'
        if revision_file.exists() and (RUN_ROOT / 'linkedin_salary/manifest.json').exists():
            saved_manifest = json.loads((RUN_ROOT / 'linkedin_salary/manifest.json').read_text())
            if saved_manifest['micro_batch'] == MICRO_BATCH:
                REPO_REF = revision_file.read_text().strip()
                assert re.fullmatch(r'[0-9a-f]{40}', REPO_REF), 'Invalid saved source revision.'
            elif saved_manifest['micro_batch'] == 256 and MICRO_BATCH == 16:
                print('Approved 256 -> 16 adjustment: fetch updated source, then validate baseline migration.')
            else:
                raise RuntimeError('Unsupported change to an existing experiment; keep the saved settings.')
        elif revision_file.exists():
            print('Previous setup did not initialize a full experiment; fetching the requested source revision.')
    """), cell("code", """
        import urllib.request, urllib.error
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        gh_token = secrets.get_secret('GH_TOKEN')
        def github_request(url):
            return urllib.request.urlopen(urllib.request.Request(url, headers={
                'Authorization': 'Bearer ' + gh_token,
                'Accept': 'application/vnd.github+json', 'User-Agent': 'linkedin-kaggle-runner'}), timeout=120)
        try:
            with github_request('https://api.github.com/repos/diyarh26/text-tabular-curation/commits/' + REPO_REF) as response:
                source_sha = json.load(response)['sha']
            archive_path = SCRATCH / 'source.zip'
            with github_request('https://api.github.com/repos/diyarh26/text-tabular-curation/zipball/' + source_sha) as response:
                with archive_path.open('wb') as output:
                    shutil.copyfileobj(response, output)
        except urllib.error.HTTPError as exc:
            raise RuntimeError('GitHub returned HTTP %s. Check GH_TOKEN repository access and Internet setting.' % exc.code) from None
        finally:
            del gh_token
        PROJECT_PARENT = SCRATCH / 'source'
        PROJECT_PARENT.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if not (PROJECT_PARENT / info.filename).resolve().is_relative_to(PROJECT_PARENT.resolve()):
                    raise ValueError('Unsafe repository archive member')
            prefix = archive.namelist()[0].split('/')[0]
            archive.extractall(PROJECT_PARENT)
        PROJECT = PROJECT_PARENT / prefix
        assert (PROJECT / 'scripts/run_full_linkedin.py').exists(), 'Repository revision lacks the Kaggle launcher.'
        revision_file.write_text(source_sha + '\\n')
        (PROJECT / 'SOURCE_REVISION').write_text(source_sha + '\\n')
        print('Source commit:', source_sha)
        ENV = {**os.environ, 'MPLBACKEND': 'Agg', 'HF_HOME': str(SCRATCH / 'huggingface'),
               'TORCH_HOME': str(SCRATCH / 'torch'), 'XDG_CACHE_HOME': str(SCRATCH / 'cache'),
               'UV_CACHE_DIR': str(SCRATCH / 'uv_cache'), 'TOKENIZERS_PARALLELISM': 'false',
               'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
        try:
            ENV['HF_TOKEN'] = secrets.get_secret('HF_TOKEN')
        except Exception:
            print('No optional HF_TOKEN; attempting public model access.')
        try:
            ENV['TABPFN_TOKEN'] = secrets.get_secret('TABPFN_TOKEN')
        except Exception:
            print('No TABPFN_TOKEN; the model-access check below will verify whether weights are available.')
        def run_checked(command, *, env=None, cwd=PROJECT):
            process = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env or ENV,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
            if process.wait():
                raise RuntimeError('Command failed (see preceding output); exit code %s' % process.returncode)
    """), cell("code", """
        # Isolated Python 3.11 and pinned full benchmark dependencies; no Kaggle package upgrades.
        uv = shutil.which('uv')
        if not uv:
            run_checked([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv'])
            uv = shutil.which('uv')
        assert uv, 'uv installation did not expose its executable.'
        VENV = SCRATCH / 'venv'
        PYTHON = VENV / 'bin/python'
        if not PYTHON.exists():
            run_checked([uv, 'venv', '--python', '3.11', VENV])
        run_checked([uv, 'pip', 'install', '--python', PYTHON,
                     '-c', PROJECT / 'requirements/constraints.txt',
                     '-e', str(PROJECT) + '[bench,dev]', 'autogluon-features==1.5.0'])
        # Verify CUDA independently with the same visibility masks used by the workers.
        for gpu in range(2):
            run_checked([PYTHON, '-c',
                "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; "
                "x=torch.ones(4,device='cuda'); assert x.sum().item()==4; "
                "print('PyTorch',torch.__version__,'GPU',torch.cuda.get_device_name(0)); "
                "from tt_curation.runner import _model_class; "
                "[_model_class(m) for m in ['tabm','cat','light','tabpfnv2','tabpfnv2p5']]; print('All learners imported')"],
                env={**ENV, 'CUDA_VISIBLE_DEVICES': str(gpu)})
        # Lightweight local tests; these do not download models or launch experiments.
        run_checked([PYTHON, '-m', 'pytest', '-q', '-m', 'not bench'])
    """), cell("code", """
        # Preserve previous batch-256 results and import only unchanged, validated baseline runs.
        # TAR adapters/checkpoints/results are never reused across micro-batch settings.
        manifest_path = RUN_ROOT / 'linkedin_salary/manifest.json'
        if manifest_path.exists():
            saved_manifest = json.loads(manifest_path.read_text())
            if saved_manifest['micro_batch'] == 256 and MICRO_BATCH == 16:
                run_checked([PYTHON, PROJECT / 'scripts/migrate_linkedin_microbatch.py',
                             '--results_root', RUN_ROOT], env={**ENV, 'CUDA_VISIBLE_DEVICES': '0'})
            else:
                assert saved_manifest['micro_batch'] == MICRO_BATCH, 'Existing micro-batch differs.'
                print('Existing experiment uses the selected micro-batch; no migration needed.')
        else:
            print('Fresh experiment: no previous baseline results to migrate.')
    """), cell("code", """
        # Check BOTH classification and regression checkpoints before spending time on TAR.
        smoke_env = {**ENV, 'CUDA_VISIBLE_DEVICES': '0'}
        run_checked([PYTHON, '-c',
            "import numpy as np; from tabpfn import TabPFNClassifier,TabPFNRegressor; "
            "from tabpfn.constants import ModelVersion; "
            "rng=np.random.default_rng(42); X=rng.normal(size=(64,4)); "
            "yc=np.arange(64)%2; yr=X[:,0]+rng.normal(size=64)*.1; "
            "[(lambda m: (m.fit(X,y), print(cls.__name__,v,m.predict(X[:2]))))(cls.create_default_for_version("
            "version=ModelVersion(v),device='cuda',random_state=42,ignore_pretraining_limits=True)) "
            "for cls,y in [(TabPFNClassifier,yc),(TabPFNRegressor,yr)] for v in ['v2','v2.5']]"], env=smoke_env)
        print('Both TabPFN versions: classification/regression model access verified.')
    """), cell("code", """
        # Smoke test ALL 20 combinations on GPU 0, including official-budget TAR.
        # Kept in scratch; these synthetic results never enter the LinkedIn reports.
        smoke_env = {**ENV, 'CUDA_VISIBLE_DEVICES': '0'}
        run_checked([PYTHON, PROJECT / 'prep/make_toy_dataset.py'], env=smoke_env)
        # The same persistent helper and chosen micro-batch are used in smoke and full runs.
        # The selected micro-batch is explicit; there is no automatic fallback on OOM.
        smoke_root = SCRATCH / ('smoke_micro' + str(MICRO_BATCH))
        smoke_program = "\\n".join([
            "import sys; from pathlib import Path",
            "sys.path.insert(0, str(Path('scripts').resolve()))",
            "from run_full_linkedin import full_tar",
            "from tt_curation.spec import load_spec",
            "from tt_curation.data import load_dataset",
            "from tt_curation.conditions import MODELS,Condition",
            "from tt_curation.runner import run_all",
            "root=Path(sys.argv[1]); micro=int(sys.argv[2])",
            "data=load_dataset(load_spec('configs/toy.yaml'))",
            "with full_tar(root/'checkpoints',micro_batch=micro):",
            "    run_all(data,root/'toy',models=list(MODELS),conditions=list(Condition),folds=[0],device='cuda')",
        ])
        run_checked([PYTHON, '-u', '-c', smoke_program, smoke_root, str(MICRO_BATCH)], env=smoke_env)
        smoke_rows = [json.loads(p.read_text()) for p in (smoke_root / 'toy/runs').glob('*.json')]
        expected = {(m, c, 0) for m in ['tabm','cat','light','tabpfnv2','tabpfnv2p5']
                    for c in ['structured','unstructured','joint_frozen','joint_tar']}
        assert {(r['short_name'], r['condition'], r['fold']) for r in smoke_rows} == expected
        assert all(r['status'] == 'ok' for r in smoke_rows), 'Smoke failure: inspect traceback above. Full run not started.'
        print('20/20 smoke evaluations passed.')
    """), cell("code", """
        remaining = SESSION_WORK_HOURS * 3600 - (time.monotonic() - SESSION_START)
        print('Launching GPU 0: folds 0,2,4 | GPU 1: folds 1,3,5')
        print('Remaining training budget: %.2f hours; progress prints every minute.' % (remaining / 3600))
        # The supervisor exports partial work even on a worker error, then reports failure.
        # Reports/checkpoints go to working; environment and downloaded weights stay in scratch.
        run_checked([PYTHON, '-u', PROJECT / 'scripts/kaggle_linkedin.py',
                     '--results_root', RUN_ROOT, '--max_seconds', str(remaining),
                     '--micro_batch', str(MICRO_BATCH)])
    """), cell("code", """
        from IPython.display import display, FileLink
        summary = RUN_ROOT / 'linkedin_salary/summary.txt'
        if summary.exists():
            print(summary.read_text())
        for name in ['linkedin_full_results.zip', 'linkedin_full_checkpoint.zip']:
            path = WORK / name
            if path.exists():
                display(FileLink(str(path)))
        print('Keep the checkpoint ZIP to resume. Share the smaller results ZIP for analysis.')
    """)]
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "cells": cells}
    for i, entry in enumerate(cells):
        entry["id"] = f"linkedin-kaggle-{i}"
        if entry["cell_type"] == "code":
            compile(entry["source"], f"kaggle_cell_{i}", "exec")
    destination = ROOT / "notebooks/run_linkedin_full_kaggle.ipynb"
    destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
