"""Generate only the FDA Device notebook against an exact runtime source revision."""
import argparse
import json
from pathlib import Path
from build_kaggle_candidate_notebooks import build_notebook
from run_full_candidate import CANDIDATES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_revision', required=True)
    args = parser.parse_args()
    notebook = build_notebook(CANDIDATES['device'], args.source_revision)
    path = Path(__file__).resolve().parents[1] / 'notebooks/run_device_full_kaggle.ipynb'
    path.write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
