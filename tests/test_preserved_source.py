"""The original experiment code and already-published result records must stay intact."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_original_vm_code_matches_preserved_git_blob_hashes():
    root=ROOT/'reproduction/original_vm256'
    manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    assert manifest['source_revision']=='bfe0ba3b846cb8bd719d43a586dc4d25b36088cd'
    for relative,digest in manifest['raw_git_blob_sha256'].items():
        assert hashlib.sha256((root/'code'/relative).read_bytes()).hexdigest()==digest,relative


def test_recorded_original_result_bytes_are_preserved():
    root=ROOT/'results/linkedin_batch256/linkedin_salary'
    index=json.loads((ROOT/'reproduction/original_vm256/original_run_sha256.json').read_text(encoding='utf-8'))
    for relative,digest in index.items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest,relative
