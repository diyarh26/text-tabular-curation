"""Protect the multiclass Device run from incorrect score audits and reuse."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_full_device import audit_runs, validate_resume, fold_dir, MODELS, CONDITIONS, make_report
from tt_curation.conditions import MODELS as NAMES


def populate(output):
    for fold in range(6):
        runs = fold_dir(output, fold) / 'runs'
        runs.mkdir(parents=True)
        for model in MODELS:
            for condition in CONDITIONS:
                row = dict(dataset='fda_device_severity', model=NAMES[model], short_name=model,
                    condition=condition, fold=fold, size='10k', task='multiclass', status='ok',
                    test_score=-.3, test_error=.3, auroc=.8, metric='log_loss', n_train=10000, n_test=5000,
                    multabench_commit='ff0c9c8b40a2085a06a7771190e53a261341802f',
                    text_encoder='e5-small-tar' if condition=='joint_tar' else None if condition=='structured' else 'e5-small')
                (runs/f'{model}_{condition}_10k_fold{fold}.json').write_text(json.dumps(row))


def test_complete_negative_scores_and_reports(tmp_path):
    populate(tmp_path)
    assert audit_runs(tmp_path)==[]
    validate_resume(tmp_path,0)
    assert make_report(tmp_path)
    assert 'COMPLETE:' in (tmp_path/'summary.txt').read_text()


@pytest.mark.parametrize('change',[{'dataset':'steam_recommended'},{'text_encoder':'tfidf'},
    {'status':'error'}, {'test_score':.3}, {'task':'binary'}, {'size':'1k'}, {'auroc':1.1}, {'model':'wrong'}])
def test_invalid_results_cannot_be_skipped(tmp_path,change):
    populate(tmp_path)
    path=fold_dir(tmp_path,0)/'runs/tabm_unstructured_10k_fold0.json'
    row=json.loads(path.read_text()); row.update(change); path.write_text(json.dumps(row))
    assert len(audit_runs(tmp_path))==1
    with pytest.raises(ValueError): validate_resume(tmp_path,0)


def test_missing_failed_and_stale_verdict(tmp_path):
    populate(tmp_path)
    assert make_report(tmp_path)
    (fold_dir(tmp_path,0)/'runs/tabm_structured_10k_fold0.json').unlink()
    assert len(audit_runs(tmp_path))==1
    assert not make_report(tmp_path)
    assert not (tmp_path/'curation_verdict.csv').exists()
