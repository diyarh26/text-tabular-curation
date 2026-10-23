"""Build separate, revision-pinned LendingClub and Food Kaggle notebooks."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import re
import textwrap

from kaggle_candidates import restore_archive
from run_full_candidate import CANDIDATES

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=textwrap.dedent(source).strip()+"\n")
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook(candidate, source_revision):
    micro_batch = 256 if candidate.key == "device" else 64
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("Use an exact published 40-character source commit")
    description = ("15,000 complete loans; description content length >50 words; purpose removed; "
                   "title retained. Target 0=Fully Paid (12,822), 1=Charged Off (2,178). Primary metric: AUROC."
                   if candidate.key == "lendingclub" else
                   "7,918 recall events; one product per event; three original severity classes: "
                   "Class I=3,182, Class II=4,065, Class III=671. Primary metric: log loss, stored as "
                   "negative log loss. Macro OvR AUROC is reported separately.")
    if candidate.key == "device":
        description = ("15,000 fixed recall events; Class I=988, Class II=13,426, Class III=586. "
                       "Three text inputs and eight structured inputs; intentional missing values retained. "
                       "Primary metric: NEGATIVE LOG LOSS. Macro OvR AUROC is supplementary.")
    execution = ("The Device execution uses **physical batch 256, accumulation 1, effective batch 256**, "
                 "float32 and the official 50-epoch cap/early stopping. No micro-batch fallback is authorized."
                 if candidate.key == "device" else
                 "The approved Kaggle execution uses **micro-batch 64, accumulation 4, effective batch 256**, "
                 "with corrected loss normalization, float32 and the official 50-epoch cap/early stopping. "
                 "It can differ numerically from physical batch 256; the execution variant is recorded.")
    cells = [cell("markdown", f"""
        # {candidate.title}: full evaluation on Kaggle T4 x2

        **Five learners x four conditions x six folds = 120 evaluations.** {description}
        Use the exact prepared CSV; this notebook does not resample or change missing-value rules.

        **Before running:**
        1. Keep the notebook **private**. Enable **GPU T4 x2** and **Internet**.
        2. Add and ENABLE the Kaggle secret **GH_TOKEN** for this notebook, with Contents read access
           to private repository `diyarh26/text-tabular-curation`. Tokens must not appear in cells.
        3. For TabPFN-v2.5, accept the model license at https://ux.priorlabs.ai and enable the
           **TABPFN_TOKEN** secret if required. `HF_TOKEN` is a separate optional credential.
        4. Use **Save Version > Save & Run All**. Do not simultaneously launch an interactive copy.
           Check remaining GPU quota; both candidate notebooks consume the same account quota.

        GPU 0 owns folds 0/2/4; GPU 1 owns folds 1/3/5. Each worker sees one T4; memory is not pooled.
        {execution}
        Frozen conditions use E5-small-v2, not TF-IDF. There is no additional wall-time cutoff or pilot cap.

        **LIVE RESULTS:** the running cell prints a complete five-learner result table after EACH fold,
        including Joint Signal/TAR deltas and independent passing counts. It also prints cumulative
        averages over fully completed folds only. No need to wait for the cell or ZIP export to finish.
        Fold tables are preliminary; all six folds are needed for the final verdict.

        Results/checkpoints are isolated in `/kaggle/working/{candidate.run_name}`. Rolling result and
        checkpoint ZIPs refresh every 15 minutes. Kaggle's own session/quota limits still apply;
        preserve the checkpoint ZIP or saved notebook output to resume after interruption.
        Attach the previous saved output via **Add Input**; only this candidate's checkpoint is detected.
        Resume pins the original source commit and verifies data, packages, code, splits and settings.
        No Steam, LinkedIn, CPU-screen or other candidate results are imported.
        This notebook starts a fresh batch-{micro_batch} experiment. Attach only matching checkpoints;
        settings mismatches are rejected rather than mixing TAR protocols.

        Sources: [Kaggle notebook execution/output limits](https://www.kaggle.com/docs/notebooks),
        [GPU quota](https://www.kaggle.com/docs/efficient-gpu-usage).
    """), cell("code", f"""
        import os, sys, time, json, subprocess, shutil, zipfile, re, stat, tempfile
        from pathlib import Path
        DATASET = {candidate.key!r}
        DATASET_NAME = {candidate.name!r}
        RUN_NAME = {candidate.run_name!r}
        MICRO_BATCH = {micro_batch}  # Effective batch 256; no automatic fallback.
        REPO_REF = {source_revision!r}  # Exact published source; resume uses its saved exact commit.
        RESUME_ZIP = ''  # Auto-detect one matching checkpoint in attached input; set if several exist.
        WORK = Path('/kaggle/working')
        RUN_ROOT = WORK / RUN_NAME
        SCRATCH = Path('/kaggle/temp') / (RUN_NAME + '_runtime')
        SCRATCH.mkdir(parents=True, exist_ok=True)
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        assert MICRO_BATCH in (16, 32, 64, 256), 'Unsupported execution batch.'
        print(subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                       '--format=csv'], text=True))
        gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=name',
                                            '--format=csv,noheader'], text=True).strip().splitlines()
        assert len(gpu_lines) == 2 and all('T4' in row for row in gpu_lines), 'Select GPU T4 x2.'
        print('Free working space: %.1f GiB' % (shutil.disk_usage(WORK).free / 2**30))
        print('Dataset:', DATASET_NAME, '| micro batch:', MICRO_BATCH,
              '| accumulation steps:', 256 // MICRO_BATCH, '| effective batch: 256')
    """)]
    restore_source = inspect.getsource(restore_archive)
    cells.append(cell("code", restore_source + "\n" + textwrap.dedent("""
        candidates = sorted(Path('/kaggle/input').rglob(RUN_NAME + '_checkpoint.zip'))
        if RESUME_ZIP:
            checkpoint_zip = Path(RESUME_ZIP)
        elif len(candidates) == 1:
            checkpoint_zip = candidates[0]
        elif len(candidates) > 1:
            raise RuntimeError('Multiple matching checkpoints found. Set RESUME_ZIP: ' + str(candidates))
        else:
            checkpoint_zip = None
        revision_file = RUN_ROOT / 'source_revision.txt'
        if revision_file.exists():
            REPO_REF = revision_file.read_text().strip()
            print('Reusing current working output; attached ZIP will not overwrite it.')
        elif checkpoint_zip:
            REPO_REF = restore_archive(checkpoint_zip, RUN_ROOT, DATASET_NAME)
            print('Restored checkpoint:', checkpoint_zip)
        elif any(RUN_ROOT.iterdir()):
            raise RuntimeError('Existing output has no source revision. Preserve it and investigate before proceeding.')
        else:
            print('Starting a fresh experiment.')
        assert re.fullmatch(r'[0-9a-f]{40}', REPO_REF), 'Expected an exact source revision.'
        saved_manifest = RUN_ROOT / DATASET_NAME / 'manifest.json'
        if saved_manifest.exists():
            saved_settings = json.loads(saved_manifest.read_text())
            if saved_settings.get('micro_batch') != MICRO_BATCH:
                raise RuntimeError('Checkpoint batch differs from this notebook. Preserve the old output; '
                                   'resume it with its original notebook or start this run fresh.')
    """)))
    cells.append(cell("code", """
        import urllib.request, urllib.error
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        try:
            gh_token = secrets.get_secret('GH_TOKEN')
        except Exception:
            raise RuntimeError('Add GH_TOKEN in Add-ons > Secrets and ENABLE it for THIS notebook, then rerun.') from None
        def github_request(url):
            return urllib.request.urlopen(urllib.request.Request(url, headers={
                'Authorization': 'Bearer ' + gh_token, 'Accept': 'application/vnd.github+json',
                'User-Agent': 'candidate-kaggle-runner'}), timeout=120)
        try:
            with github_request('https://api.github.com/repos/diyarh26/text-tabular-curation/commits/' + REPO_REF) as response:
                source_sha = json.load(response)['sha']
            assert source_sha == REPO_REF, 'Source revision mismatch.'
            archive_path = SCRATCH / 'source.zip'
            with github_request('https://api.github.com/repos/diyarh26/text-tabular-curation/zipball/' + source_sha) as response:
                with archive_path.open('wb') as output:
                    shutil.copyfileobj(response, output)
        except urllib.error.HTTPError as exc:
            raise RuntimeError('GitHub HTTP %s. Check GH_TOKEN Contents read access and Internet.' % exc.code) from None
        finally:
            del gh_token
        project_parent = SCRATCH / 'source'
        project_parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if ('\\\\' in info.filename or
                        not (project_parent / info.filename).resolve().is_relative_to(project_parent.resolve()) or
                        stat.S_ISLNK(info.external_attr >> 16)):
                    raise ValueError('Unsafe source archive member')
            prefix = archive.namelist()[0].split('/')[0]
            archive.extractall(project_parent)
        PROJECT = project_parent / prefix
        assert (PROJECT / 'scripts/run_full_candidate.py').exists(), 'Source lacks the candidate launcher.'
        assert (PROJECT / 'configs' / (DATASET_NAME + '.yaml')).exists(), 'Source lacks the selected config.'
        revision_file.write_text(source_sha + '\\n')
        (PROJECT / 'SOURCE_REVISION').write_text(source_sha + '\\n')
        print('Exact source commit:', source_sha)
        ENV = {**os.environ, 'MPLBACKEND': 'Agg', 'HF_HOME': str(SCRATCH / 'huggingface'),
               'TORCH_HOME': str(SCRATCH / 'torch'), 'XDG_CACHE_HOME': str(SCRATCH / 'cache'),
               'UV_CACHE_DIR': str(SCRATCH / 'uv_cache'), 'TOKENIZERS_PARALLELISM': 'false',
               'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1',
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
        for label in ('HF_TOKEN', 'TABPFN_TOKEN'):
            try:
                ENV[label] = secrets.get_secret(label)
            except Exception:
                print('No optional', label, '; model-access check will verify availability.')
        def run_checked(command, *, env=None, cwd=PROJECT):
            process = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env or ENV,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                code = process.wait()
            except KeyboardInterrupt:
                import signal
                process.send_signal(signal.SIGINT)
                process.wait()
                raise
            if code:
                raise RuntimeError('Command failed; inspect preceding output. Exit code %s' % code)
    """))
    cells.append(cell("code", """
        # Isolated Python 3.11 with pinned full dependencies; keep environment/caches outside saved output.
        uv = shutil.which('uv')
        if not uv:
            run_checked([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv'])
            uv = shutil.which('uv')
        assert uv, 'uv executable not available.'
        VENV = SCRATCH / 'venv'
        PYTHON = VENV / 'bin/python'
        if not PYTHON.exists():
            run_checked([uv, 'venv', '--python', '3.11', VENV])
        run_checked([uv, 'pip', 'install', '--python', PYTHON, '-c', PROJECT / 'requirements/constraints.txt',
                     '-e', str(PROJECT) + '[bench,dev]', 'autogluon-features==1.5.0'])
        for gpu in range(2):
            run_checked([PYTHON, '-c',
                "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; "
                "x=torch.ones(4,device='cuda'); assert x.sum().item()==4; "
                "print('PyTorch',torch.__version__,'GPU',torch.cuda.get_device_name(0)); "
                "from tt_curation.runner import _model_class; "
                "[_model_class(m) for m in ['tabm','cat','light','tabpfnv2','tabpfnv2p5']]; "
                "print('All five learners imported')"], env={**ENV, 'CUDA_VISIBLE_DEVICES': str(gpu)})
        run_checked([PYTHON, '-m', 'pytest', '-q', 'tests/test_full_candidates.py', '-m', 'not bench'])
        # Tiny CPU checks of the ACTUAL pinned Trainer loss, updates and checkpoint resumption.
        # No dataset screen, no 20-case toy GPU experiment, no TAR pilot.
        run_checked([PYTHON, '-m', 'pytest', '-q', 'tests/test_full_linkedin.py', '-m', 'bench'],
                    env={**ENV, 'CUDA_VISIBLE_DEVICES': ''})
        run_checked([PYTHON, '-u', PROJECT / 'scripts/run_full_candidate.py', '--dataset', DATASET,
                     '--results_root', RUN_ROOT, '--micro_batch', str(MICRO_BATCH), '--prepare_only'],
                    env={**ENV, 'CUDA_VISIBLE_DEVICES': '0'})
    """))
    cells.append(cell("code", f"""
        # Verify both classification checkpoints for THIS task before the long run.
        # No regression checkpoint downloads are needed for these two classification datasets.
        model_check = '\\n'.join([
            'import gc, numpy as np, torch',
            'from tabpfn import TabPFNClassifier',
            'from tabpfn.constants import ModelVersion',
            'rng=np.random.default_rng(42); X=rng.normal(size=(96,4)); y=np.arange(96)%{len(candidate.counts)}',
            "for version in ['v2','v2.5']:",
            "    model=TabPFNClassifier.create_default_for_version(version=ModelVersion(version), device='cuda', random_state=42, ignore_pretraining_limits=True)",
            '    model.fit(X,y); probabilities=model.predict_proba(X[:4])',
            '    assert probabilities.shape == (4,{len(candidate.counts)}) and np.isfinite(probabilities).all()',
            "    print('TabPFN',version,'classification CUDA access verified',flush=True)",
            '    del model; gc.collect(); torch.cuda.empty_cache()',
        ])
        run_checked([PYTHON, '-u', '-c', model_check], env={{**ENV, 'CUDA_VISIBLE_DEVICES': '0'}})
    """))
    cells.append(cell("code", """
        from IPython.display import display, FileLink
        print('Launching all 120 evaluations: GPU 0 folds 0/2/4; GPU 1 folds 1/3/5.', flush=True)
        print('LIVE fold tables appear in THIS cell when each fold finishes. No notebook time cutoff.', flush=True)
        try:
            run_checked([PYTHON, '-u', PROJECT / 'scripts/kaggle_candidates.py', '--dataset', DATASET,
                         '--results_root', RUN_ROOT, '--micro_batch', str(MICRO_BATCH)])
        finally:
            for suffix in ('results', 'checkpoint'):
                path = WORK / (RUN_NAME + '_' + suffix + '.zip')
                if path.exists():
                    display(FileLink(str(path)))
            print('Keep the checkpoint ZIP or saved output for resumption; share results ZIP for analysis.', flush=True)
    """))
    cells.append(cell("code", """
        summary = RUN_ROOT / DATASET_NAME / 'summary.txt'
        if summary.exists():
            print(summary.read_text())
        print('Per-fold reports:', RUN_ROOT / DATASET_NAME / 'foldN_report.txt')
        print('Cumulative completed-fold comparison:', RUN_ROOT / DATASET_NAME / 'progress_report.txt')
    """))
    for index, entry in enumerate(cells):
        entry["id"] = f"{candidate.key}-kaggle-{index}"
        if entry["cell_type"] == "code":
            compile(entry["source"], f"{candidate.key}_cell_{index}", "exec")
    return dict(nbformat=4, nbformat_minor=5, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, cells=cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_revision", required=True)
    args = parser.parse_args()
    for candidate in CANDIDATES.values():
        notebook = build_notebook(candidate, args.source_revision)
        destination = ROOT / "notebooks" / f"run_{candidate.key}_full_kaggle.ipynb"
        destination.write_text(json.dumps(notebook, indent=1)+"\n", encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
