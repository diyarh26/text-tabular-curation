"""Protect original result provenance and reject premature submission packages."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import runpy
import zipfile

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import report_linkedin as reporting
from build_submission import build, build_code


@pytest.fixture(autouse=True)
def synthetic_original_index(tmp_path,monkeypatch):
    monkeypatch.setattr(reporting,'ORIGINAL_RESULTS_INDEX',tmp_path/'original_run_sha256.json')


def fixture(output,complete=False):
    manifest=json.loads((ROOT/"reproduction/original_vm256/experiment_manifest.json").read_text())
    output.mkdir()
    (output/"manifest.json").write_text(json.dumps(manifest))
    for fold in range(6):
        for short,name in reporting.MODELS.items():
            for condition in reporting.CONDITIONS:
                if not complete and (fold==5 or (fold==4 and condition=="joint_tar")): continue
                score={"structured":.4,"unstructured":.6,"joint_frozen":.65,"joint_tar":.7}[condition]
                row=dict(dataset="linkedin_salary",short_name=short,model=name,fold=fold,condition=condition,
                    protocol=reporting.PROTOCOL,git=reporting.SOURCE,multabench_commit=manifest['multabench_commit'],
                    task="regression",size="10k",n_train=10000,n_test=5000,n_features=12 if condition=="structured" else 2 if condition=="unstructured" else 14,
                    metric="root_mean_squared_error",text_encoder=None if condition=="structured" else "e5-small-tar" if condition=="joint_tar" else "e5-small",
                    r2=score,test_error=.3,test_score=-.3,runtime_s=1.,device="cuda",status="ok")
                if condition=="joint_tar": row.update(tar_parameters=manifest['tar_parameters'],training_arguments=manifest['training_arguments'],tar_finetune=dict(rows=10000,epochs=10.,best_val_loss=.5))
                path=output/f'folds/fold{fold}/runs/{short}_{condition}_10k_fold{fold}.json'
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(row))
    original={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.glob('folds/*/runs/*.json')
              if int(p.parent.parent.name.removeprefix('fold'))<4 or
                 (p.parent.parent.name=='fold4' and '_joint_tar_' not in p.name)}
    reporting.ORIGINAL_RESULTS_INDEX.write_text(json.dumps(original))
    return output


def test_partial_report_excludes_partial_fold_and_preserves_all_original_bytes(tmp_path):
    output=fixture(tmp_path/'results')
    before={p:p.read_bytes() for p in [output/'manifest.json',*output.glob('folds/*/runs/*.json')]}
    (output/'curation_verdict.csv').write_text('STALE')
    status=reporting.report(output)
    assert status['successful_evaluations']==95 and status['completed_folds']==[0,1,2,3]
    assert not status['complete'] and not (output/'curation_verdict.csv').exists()
    table=pd.read_csv(output/'evaluation_report.csv')
    assert len(table)==20 and set(table.n_folds)=={4}
    assert len(pd.read_csv(output/'per_fold_results.csv'))==95
    assert all(p.read_bytes()==data for p,data in before.items())


@pytest.mark.parametrize('update',[{'git':'wrong'},{'protocol':'16x16'},{'r2':float('nan')},{'r2':.55},{'n_features':13},{'text_encoder':'tfidf'},{'training_arguments':{'per_device_train_batch_size':16}}])
def test_mixed_or_invalid_tar_records_are_rejected(tmp_path,update):
    output=fixture(tmp_path/'results')
    p=output/'folds/fold0/runs/cat_joint_tar_10k_fold0.json'
    row=json.loads(p.read_text());row.update(update);p.write_text(json.dumps(row))
    with pytest.raises(ValueError): reporting.audit(output)


def test_final_report_requires_every_identity_and_uses_r2(tmp_path):
    output=fixture(tmp_path/'results',complete=True)
    status=reporting.report(output)
    assert status['complete'] and 'Final verdict: PASS' in (output/'summary.txt').read_text()
    assert len(pd.read_csv(output/'curation_verdict.csv'))==5
    assert set(pd.read_csv(output/'evaluation_report.csv').n_folds)=={6}
    # RMSE is identical in the fixture; only the requested R2 metric has positive deltas.
    assert pd.read_csv(output/'curation_verdict.csv').joint_signal.all()
    (output/'folds/fold5/runs/cat_joint_tar_10k_fold5.json').unlink()
    reporting.report(output)
    assert not (output/'curation_verdict.csv').exists()


def test_submission_rejects_missing_paper_and_incomplete_results(tmp_path):
    output=fixture(tmp_path/'results')
    with pytest.raises(ValueError,match='final LinkedIn paper'): build(output,tmp_path/'missing.pdf',tmp_path/'a.zip')
    paper=tmp_path/'test.pdf';paper.write_bytes(b'%PDF-1.4\nunit-test stub')
    with pytest.raises(ValueError,match='all 120'): build(output,paper,tmp_path/'a.zip')
    assert not (tmp_path/'a.zip').exists()


def test_complete_package_preserves_layout_hashes_and_excludes_archives(tmp_path):
    output=fixture(tmp_path/'results',complete=True)
    paper=tmp_path/'unit-test.pdf';paper.write_bytes(b'%PDF-1.4\nunit-test stub, not a deliverable')
    destination=build(output,paper,tmp_path/'package.zip')
    with zipfile.ZipFile(destination) as z:
        assert set(z.namelist())=={'track2_submission/'+name for name in (
            'data/dataset.csv','src/preprocess.py','src/evaluate.py',
            'evaluation_results.csv','final_report.pdf','README.md')}
        assert z.read('track2_submission/data/dataset.csv')==(ROOT/'data/processed/linkedin_salary.csv').read_bytes()
        assert z.read('track2_submission/evaluation_results.csv')==(output/'evaluation_report.csv').read_bytes()
        assert z.read('track2_submission/final_report.pdf')==paper.read_bytes()
        assert 'All 120 original' in z.read('track2_submission/README.md').decode()
        assert '95/120' not in z.read('track2_submission/README.md').decode()


def test_two_file_bundle_imports_without_repository_helpers(tmp_path):
    destination=build_code(tmp_path/'code.zip')
    extracted=tmp_path/'standalone'
    with zipfile.ZipFile(destination) as z:
        assert set(z.namelist())=={'track2_submission/'+name for name in (
            'data/dataset.csv','src/preprocess.py','src/evaluate.py','evaluation_results.csv','README.md')}
        z.extractall(extracted)
    command="""import hashlib, runpy, sys
from pathlib import Path
root=Path('track2_submission').resolve()
preprocess=runpy.run_path(str(root/'src/preprocess.py'))
evaluation=runpy.run_path(str(root/'src/evaluate.py'))
assert preprocess['OUT']==root/'data/dataset.csv'
assert '[amount]' in preprocess['mask_money']('Salary: $75,000 per year')
assert evaluation['ROOT']==root
assert evaluation['TAR']['batch_size']==256
assert hashlib.sha256((root/'data/dataset.csv').read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest()==evaluation['CSV_SHA256_LF']
assert 'tt_curation' not in sys.modules
print('Standalone two-file imports passed')
"""
    result=subprocess.run([sys.executable,'-I','-c',command],cwd=extracted,
                          capture_output=True,text=True,check=False)
    assert result.returncode==0, result.stdout+result.stderr
    assert 'Standalone two-file imports passed' in result.stdout


def test_standalone_native_report_accepts_official_short_git_hash(tmp_path):
    pytest.importorskip('multabench')
    import numpy as np
    from multabench.benchmark.splits import get_split
    code=runpy.run_path(str(ROOT/'track2_submission/src/evaluate.py'))
    csv=ROOT/'track2_submission/data/dataset.csv'
    y=pd.read_csv(csv,usecols=['log_salary']).log_salary.to_numpy()
    output=tmp_path/'synthetic_native_results'
    (output/'runs').mkdir(parents=True)
    for fold in range(6):
        variance=np.var(y[get_split(y,False,fold,size='10k')[1]],ddof=0)
        for model in code['MODELS'].values():
            for condition,score in {'structured':.4,'unstructured':.6,'joint_frozen':.65,'joint_tar':.7}.items():
                dataset={'structured':code['STRUCTURED'],'unstructured':code['TEXT']}.get(condition,code['FULL'])
                row=dict(status='ok',fold=fold,n_train=10000,n_test=5000,metric='root_mean_squared_error',
                    size='10k',model=model,dataset=dataset,git=code['OFFICIAL_COMMIT'][:7],
                    text_encoder='e5-small-tar' if condition=='joint_tar' else 'e5-small',
                    test_error=float(np.sqrt((1-score)*variance)))
                (output/'runs'/f'{fold}_{model}_{condition}.json').write_text(json.dumps(row))
    destination=tmp_path/'table.csv'
    assert code['report_results'](output,csv,destination)
    table=pd.read_csv(destination)
    assert len(table)==20 and set(table.n_folds)=={6} and set(table.status)=={'final'}
    assert np.allclose(table[table.condition=='joint_tar'].r2_mean,.7)
    assert np.allclose(table[table.condition=='joint_frozen'].r2_mean,.65)
