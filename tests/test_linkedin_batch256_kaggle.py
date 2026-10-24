"""Protect the native-256 Kaggle notebook, restoration, two-GPU scheduler and live tables."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import kaggle_linkedin_batch256 as supervisor
from build_kaggle_linkedin_batch256_notebook import build_notebook
from run_full_linkedin_vm import PROTOCOL, TRAINING

REVISION='a'*40


def checkpoint(path, *, old=False, extra=None):
    prefix=supervisor.RUN_NAME
    manifest=dict(protocol='old' if old else PROTOCOL,source_revision=REVISION,dataset='linkedin_salary',
        training_arguments=dict(TRAINING,per_device_train_batch_size=16 if old else 256))
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr(prefix+'/source_revision.txt',REVISION+'\n')
        archive.writestr(prefix+'/linkedin_salary/manifest.json',json.dumps(manifest))
        archive.writestr(prefix+'/linkedin_salary/folds/fold0/checkpoints/trainer/checkpoint-2/optimizer.pt',b'optimizer')
        if extra: archive.writestr(extra,b'bad')


def test_notebook_compiles_and_pins_source_native_training_and_two_gpu_smokes():
    notebook=build_notebook(REVISION)
    assert notebook['nbformat']==4
    code='\n'.join(c['source'] for c in notebook['cells'] if c['cell_type']=='code')
    for c in notebook['cells']:
        if c['cell_type']=='code': compile(c['source'],'cell','exec')
    assert repr(REVISION) in code
    assert 'range(2)' in code and "'--gpu_smoke'" in code
    assert 'kaggle_linkedin_batch256.py' in code
    assert 'run_full_linkedin_vm.py' in code
    assert 'migrate' not in code and '--micro_batch' not in code
    assert 'time_budget' not in code and 'max_runtime' not in code
    assert 'EXPECTED_SOURCE+\'\\n\'' in code
    assert TRAINING['per_device_train_batch_size']==TRAINING['per_device_eval_batch_size']==256
    assert TRAINING['gradient_accumulation_steps']==1 and TRAINING['dataloader_num_workers']==4
    assert not TRAINING['fp16'] and not TRAINING['bf16']
    with pytest.raises(ValueError): build_notebook('main')


def test_restore_new_protocol_zip_then_unpacked_and_preserve_existing(tmp_path):
    archive=tmp_path/'snapshot.zip'
    checkpoint(archive)
    root=tmp_path/'working'/supervisor.RUN_NAME
    supervisor.restore_input(archive,root,REVISION)
    assert (root/'linkedin_salary/manifest.json').exists()
    assert next(root.rglob('optimizer.pt')).read_bytes()==b'optimizer'
    second=tmp_path/'another'/supervisor.RUN_NAME
    supervisor.restore_input(root,second,REVISION)
    assert (second/'source_revision.txt').read_text().strip()==REVISION
    with pytest.raises(ValueError,match='overwrite'):
        supervisor.restore_input(archive,root,REVISION)


@pytest.mark.parametrize('bad',['old','source','traversal'])
def test_restore_rejects_old_protocol_other_source_and_path_traversal(tmp_path,bad):
    archive=tmp_path/'snapshot.zip'
    checkpoint(archive,old=bad=='old',extra='../escape.txt' if bad=='traversal' else None)
    root=tmp_path/'working'/supervisor.RUN_NAME
    with pytest.raises(ValueError):
        supervisor.restore_input(archive,root,'b'*40 if bad=='source' else REVISION)
    assert not (root/'linkedin_salary/manifest.json').exists()
    assert not (tmp_path/'escape.txt').exists()


def test_autosave_keeps_ready_optimizer_excludes_partial_and_never_exports_secrets(tmp_path,monkeypatch):
    root=tmp_path/supervisor.RUN_NAME
    ready=root/'linkedin_salary/folds/fold0/checkpoints/trainer/checkpoint-1'
    partial=ready.parent/'checkpoint-2'
    ready.mkdir(parents=True)
    partial.mkdir()
    (ready/'trainer_state.json').write_text(json.dumps(dict(global_step=1)))
    (ready/'optimizer.pt').write_bytes(b'optimizer')
    (partial/'optimizer.pt').write_bytes(b'partial')
    (root/'source_revision.txt').write_text(REVISION)
    (tmp_path/'secret.txt').write_text('private')
    assert supervisor.export_archives(root,live=True)
    result=tmp_path/(supervisor.RUN_NAME+'_results.zip')
    saved=tmp_path/(supervisor.RUN_NAME+'_checkpoint.zip')
    with zipfile.ZipFile(result) as z:
        assert not any('checkpoints/' in n for n in z.namelist())
    with zipfile.ZipFile(saved) as z:
        assert any('checkpoint-1/optimizer.pt' in n for n in z.namelist())
        assert not any('checkpoint-2/' in n or 'secret' in n for n in z.namelist())
    previous=saved.read_bytes()
    original=zipfile.ZipFile.write
    def changed(self,filename,*args,**kwargs):
        result=original(self,filename,*args,**kwargs)
        if Path(filename).name=='optimizer.pt': Path(filename).write_bytes(b'changed')
        return result
    monkeypatch.setattr(zipfile.ZipFile,'write',changed)
    assert not supervisor.export_archives(root,live=True)
    assert saved.read_bytes()==previous


def test_live_reports_are_per_fold_and_cumulative_without_waiting_for_other_fold(tmp_path,monkeypatch,capsys):
    folder=tmp_path/'folds/fold1'
    (folder/'runs').mkdir(parents=True)
    (folder/'r2_summary.txt').write_text('R2 table: joint delta +0.03; TAR delta +0.04')
    (folder/'runs/cat_joint_tar_10k_fold1.json').write_text(json.dumps(dict(tar_finetune=dict(epochs=7))))
    monkeypatch.setattr(supervisor,'audit_runs',lambda output,folds: [] if folds==[1] else ['missing'])
    calls=[]
    monkeypatch.setattr(supervisor,'write_reports',lambda output:calls.append(output))
    announced=set()
    supervisor.emit_reports(tmp_path,announced)
    supervisor.emit_reports(tmp_path,announced)
    text=capsys.readouterr().out
    assert text.count('FOLD 1 COMPLETE')==1 and 'R2 table' in text
    assert announced=={1} and len(calls)==1


def test_shared_queue_reassigns_to_free_gpu_without_duplicate_folds_or_ten_hour_cutoff(tmp_path,monkeypatch,capsys):
    root=tmp_path/supervisor.RUN_NAME
    output=root/'linkedin_salary'
    output.mkdir(parents=True)
    (root/'source_revision.txt').write_text(REVISION)
    (output/'manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL,source_revision=REVISION,training_arguments=TRAINING)))
    for gpu in range(2):
        (output/f'gpu_preflight_gpu{gpu}.json').write_text(json.dumps(dict(passed=True,training_arguments=TRAINING)))
    clock=[0]
    started=[]
    completed=set()
    class Process:
        def __init__(self,fold):
            self.fold=fold
            self.ends=clock[0]+(33 if fold==0 else 11)*3600
        def poll(self):
            if clock[0]<self.ends: return None
            completed.add(self.fold)
            return 0
    def start(command,**kwargs):
        started.append((command,kwargs['env']['CUDA_VISIBLE_DEVICES']))
        return Process(int(command[command.index('--folds')+1]))
    def audit(output,folds=range(6),**kwargs):
        return [] if kwargs.get('allow_missing') or set(folds)<=completed else ['pending']
    monkeypatch.setattr(sys,'argv',['supervisor','--results_root',str(root)])
    monkeypatch.setattr(supervisor.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='GPU healthy'))
    monkeypatch.setattr(supervisor.subprocess,'Popen',start)
    monkeypatch.setattr(supervisor,'audit_runs',audit)
    monkeypatch.setattr(supervisor,'emit_reports',lambda *a:None)
    monkeypatch.setattr(supervisor,'progress',lambda *a:(95,[]))
    monkeypatch.setattr(supervisor,'write_reports',lambda *a:True)
    monkeypatch.setattr(supervisor,'export_archives',lambda *a,**k:True)
    monkeypatch.setattr(supervisor,'stop_workers',lambda *a:None)
    monkeypatch.setattr(supervisor,'tail',lambda *a:None)
    monkeypatch.setattr(supervisor.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(supervisor.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+11*3600))
    supervisor.main()
    text=capsys.readouterr().out
    assert 'elapsed 11.00 h' in text and 'COMPLETE: all 120' in text
    assert len(started)==6
    assigned=[(int(command[command.index('--folds')+1]),gpu) for command,gpu in started]
    assert [fold for fold,gpu in assigned]==list(range(6))
    assert assigned[:4]==[(0,'0'),(1,'1'),(2,'1'),(3,'1')]
    for command,gpu in started:
        assert '--fold_worker' in command


def test_downloaded_source_provenance_and_git_mismatch(tmp_path,monkeypatch):
    import tt_curation
    from tt_curation import runner
    import run_full_linkedin_vm as vm
    monkeypatch.setattr(tt_curation,'REPO_ROOT',tmp_path)
    monkeypatch.setattr(runner,'_git_commit',lambda:None)
    with pytest.raises(ValueError,match='exact source'): vm.source_revision()
    (tmp_path/'SOURCE_REVISION').write_text(REVISION+'\n')
    assert vm.source_revision()==REVISION
    monkeypatch.setattr(runner,'_git_commit',lambda:'b'*40)
    with pytest.raises(ValueError,match='disagree'): vm.source_revision()


def test_downloaded_run_records_verified_source_and_worker_does_not_write_global_report(tmp_path,monkeypatch):
    from tt_curation import runner
    import run_full_linkedin_vm as vm
    from test_linkedin_vm import saved_runs
    saved_runs(tmp_path,folds=[0])
    assert vm.write_reports(tmp_path,just_finished=0,fold_only=True)
    assert (tmp_path/'folds/fold0/r2_summary.txt').exists()
    assert not (tmp_path/'summary.txt').exists()
    monkeypatch.setattr(runner,'run_one',lambda:dict(status='ok',condition='structured',git=None))
    with vm.fail_fast(tmp_path/'folds/fold0'):
        row=runner.run_one()
        assert row['git']==REVISION and row['protocol']==vm.PROTOCOL
