"""Integration tests execute the real dataset-patched upstream package in isolation."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "artifacts/multabench-native"
DATA = ROOT / "artifacts/native_linkedin_data"


@pytest.fixture(scope="module")
def native_env():
    pytest.importorskip("multabench")
    if not (NATIVE / "multabench/datasets/linkedin_salary.py").exists() or not DATA.exists():
        pytest.skip("Run setup_native_multabench.py and native prepare_linkedin before integration tests")
    return {**os.environ, "PYTHONPATH": str(NATIVE), "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8",
            "MULTABENCH_LINKEDIN_DATA_DIR": str(DATA)}


def execute(code, env):
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True,
                          text=True, capture_output=True).stdout


def test_fresh_patch_application_is_idempotent_and_protects_upstream_code(native_env,tmp_path):
    checkout=tmp_path/'upstream'
    subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(NATIVE),str(checkout)],check=True)
    subprocess.run(['git','-C',str(checkout),'-c','core.autocrlf=false','checkout','--quiet','--detach',
        'ff0c9c8b40a2085a06a7771190e53a261341802f'],check=True)
    command=[sys.executable,str(ROOT/'scripts/setup_native_multabench.py'),'--checkout',str(checkout)]
    first=subprocess.run(command,check=True,capture_output=True,text=True).stdout
    second=subprocess.run(command,check=True,capture_output=True,text=True).stdout
    assert json.loads(first)==json.loads(second)
    (checkout/'benchmark.py').write_text('changed')
    bad=subprocess.run(command,capture_output=True,text=True)
    assert bad.returncode!=0 and 'Unexpected upstream modifications' in bad.stderr


def test_native_loader_exact_parity_for_all_views_and_six_splits(native_env):
    output = execute("from scripts.verify_native_linkedin import verify; verify(r'"+str(DATA)+"')", native_env)
    assert '"loader_parity_passed": true' in output


def test_bad_csv_and_modified_prepared_data_are_rejected(native_env, tmp_path):
    code = f"""
from pathlib import Path
import shutil
from multabench.datasets.linkedin_salary import typed_csv, dataset_dir, FULL, prepare
root=Path({str(tmp_path)!r})
bad=root/'bad.csv'; bad.write_text('title,log_salary\\na,1\\n')
try: typed_csv(bad)
except ValueError: pass
else: raise AssertionError('Wrong sample accepted')
copy=root/'data'; shutil.copytree({str(DATA)!r},copy)
p=copy/FULL.value/'data.parquet'; p.write_bytes(p.read_bytes()+b'changed')
try: dataset_dir(FULL,copy)
except ValueError: pass
else: raise AssertionError('Changed table accepted')
try: prepare({str(ROOT/'data/processed/linkedin_salary.csv')!r},copy)
except ValueError: pass
else: raise AssertionError('Modified output overwritten')
"""
    execute(code, native_env)


def test_native_cache_route_and_text_alias_use_real_upstream_cache_reader(native_env, tmp_path):
    code = f"""
import json,shutil,os
from pathlib import Path
import numpy as np
from multabench.datasets.linkedin_salary import FULL,TEXT,STRUCTURED,ENV
from multabench.e5.constants import TextEncoder
from multabench.embeddings.hub import write_embeddings,cached_embeddings,embeddings_dir,embeds_dataset
root=Path({str(tmp_path)!r}); os.environ[ENV]=str(root)
keys=['passage: description: example','passage: title: engineer']
values=np.arange(768,dtype=np.float32).reshape(2,384)
for encoder,fold in [(TextEncoder.E5_SMALL,None),(TextEncoder.E5_SMALL_TAR,0)]:
 path=root/('embeddings-'+encoder)/embeddings_dir(FULL,size='10k',fold=fold)
 write_embeddings(str(path),keys,values,dict(features=['description','title']))
 cache=cached_embeddings(encoder,TEXT,size='10k' if fold is not None else None,fold=fold)
 assert np.array_equal(cache.lookup(keys,feature='title'),values)
 assert embeddings_dir(TEXT,size='10k',fold=fold)==embeddings_dir(FULL,size='10k',fold=fold)
assert not embeds_dataset(TextEncoder.E5_SMALL,STRUCTURED)
try: cached_embeddings(TextEncoder.E5_SMALL_TAR,FULL,size='10k',fold=1)
except FileNotFoundError: pass
else: raise AssertionError('Missing TAR fold silently accepted')
"""
    execute(code, native_env)


def test_unchanged_native_evaluation_selects_all_four_condition_inputs(native_env):
    code = """
import torch
from multabench.baselines.benchmarks.evaluate import evaluate_on_dataset
from multabench.datasets.linkedin_salary import FULL,STRUCTURED,TEXT
from multabench.e5.constants import TextEncoder
from multabench.result_keys import ResultKey
class Capture:
 MODEL_NAME='capture'; last=None
 def __init__(self,**kwargs): self.kwargs=kwargs; Capture.last=self
 def fit(self,x,y): self.x,self.y=x,y
 def score_all_metrics(self,X,y):
  assert list(X)==list(self.x) and len(X)==5000 and len(self.x)==10000
  return {ResultKey.METRIC:'root_mean_squared_error',ResultKey.TEST_ERROR:1.,ResultKey.TEST_SCORE:-1.}
for dataset,encoder,columns in [(STRUCTURED,TextEncoder.E5_SMALL,12),(TEXT,TextEncoder.E5_SMALL,2),
 (FULL,TextEncoder.E5_SMALL,14),(FULL,TextEncoder.E5_SMALL_TAR,14)]:
 row=evaluate_on_dataset(Capture,dataset,fold=0,device=torch.device('cpu'),text_encoder=encoder)
 assert Capture.last.x.shape[1]==columns
 assert Capture.last.kwargs['text_encoder']==encoder
 assert row[ResultKey.N_TRAIN]==10000 and row[ResultKey.N_TEST]==5000
"""
    execute(code, native_env)


@pytest.mark.bench
def test_real_native_cli_lightgbm_matches_existing_runner(native_env, tmp_path):
    subprocess.run([sys.executable, str(NATIVE / "benchmark.py"), "--model", "light", "--dataset_name",
        "REG_TEXT_LINKEDIN_SALARY_STRUCTURED", "--fold", "0", "--device", "cpu", "--output_dir", str(tmp_path)],
        env=native_env, cwd=NATIVE, check=True, capture_output=True, text=True)
    row=json.loads(next(tmp_path.glob("*.json")).read_text())
    assert row["status"] == "ok"
    code="""
import json,torch
from tt_curation.data import load_dataset
from tt_curation.spec import load_spec
from tt_curation.runner import run_one
from tt_curation.conditions import Condition
from multabench.datasets.objects import MultimodalDataset,SupervisedTask
from multabench.benchmark.load import split_dataset
d=load_dataset(load_spec('configs/linkedin_salary.yaml'))
dataset=MultimodalDataset(d.x,d.y,SupervisedTask.REGRESSION,None)
r=run_one('light',split_dataset(dataset,0,'10k'),SupervisedTask.REGRESSION,d,Condition.STRUCTURED,0,'10k',torch.device('cpu'))
assert r['status']=='ok'
print('PARITY_JSON='+json.dumps(r))
"""
    output=execute(code,native_env)
    old=json.loads(output.split("PARITY_JSON=")[-1])
    assert row["test_error"] == pytest.approx(old["test_error"],abs=1e-12)
    assert row["n_train"] == old["n_train"] and row["n_test"] == old["n_test"]
