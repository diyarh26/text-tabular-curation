"""Command line: validate a dataset spec, run the curation pipeline, and build the evaluation report.

    python -m tt_curation validate configs/my_dataset.yaml
    python -m tt_curation run configs/my_dataset.yaml                    # full: 5 learners x 4 conditions x 6 folds
    python -m tt_curation run configs/my_dataset.yaml --preset screen    # quick triage of a candidate dataset
    python -m tt_curation report configs/my_dataset.yaml
"""
import argparse
import sys
from pathlib import Path

# MulTaBench prints emoji, which the Windows console's default codepage can't encode.
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

from tt_curation import REPO_ROOT
from tt_curation.conditions import Condition, MODELS

PRESETS = {
    # Fast check of the joint signal on CPU-friendly learners, before paying for the full run.
    "screen": dict(models="light,cat", conditions="structured,unstructured,joint_frozen", folds="0,1,2"),
    "full": dict(models=",".join(MODELS), conditions=",".join(c.value for c in Condition), folds="0,1,2,3,4,5"),
}


def results_dir_for(spec, results_root: str | None) -> Path:
    return Path(results_root or REPO_ROOT / "results") / spec.name


def main():
    parser = argparse.ArgumentParser(prog="tt_curation")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("config", help="the dataset's YAML spec, e.g. configs/my_dataset.yaml")
        p.add_argument("--results_root", default=None, help="default: results/ in the repo")
    run = sub.choices["run"]
    run.add_argument("--preset", choices=PRESETS, default="full")
    run.add_argument("--models", help=f"comma-separated, from {list(MODELS)}")
    run.add_argument("--conditions", help=f"comma-separated, from {[c.value for c in Condition]}")
    run.add_argument("--folds", help="comma-separated, from 0-5")
    run.add_argument("--size", default="10k", choices=["10k", "full"])
    run.add_argument("--device", default=None, help="e.g. cuda, cpu; default: cuda if available")
    run.add_argument("--overwrite", action="store_true", help="rerun runs whose JSON already exists")
    args = parser.parse_args()

    from tt_curation.spec import load_spec
    spec = load_spec(args.config)
    results_dir = results_dir_for(spec, args.results_root)

    if args.command in ("validate", "run"):
        from tt_curation.data import describe, load_dataset
        data = load_dataset(spec)
        print("\n".join(describe(data)))
        if args.command == "run":
            from tt_curation.runner import run_all
            preset = PRESETS[args.preset]
            models = (args.models or preset["models"]).split(",")
            unknown = [m for m in models if m not in MODELS]
            if unknown:
                parser.error(f"unknown models {unknown}; choose from {list(MODELS)}")
            conditions = [Condition(c) for c in (args.conditions or preset["conditions"]).split(",")]
            folds = [int(f) for f in (args.folds or preset["folds"]).split(",")]
            run_all(data, results_dir, models=models, conditions=conditions, folds=folds, size=args.size,
                    device=args.device, overwrite=args.overwrite)
            report(results_dir)
    else:
        report(results_dir)


def report(results_dir: Path):
    from tt_curation.criteria import curation_verdict, evaluation_report, load_runs, summary_lines
    runs = load_runs(results_dir)
    evaluation_report(runs).to_csv(results_dir / "evaluation_report.csv", index=False)
    verdict = curation_verdict(runs)
    verdict.to_csv(results_dir / "curation_verdict.csv", index=False)
    lines = summary_lines(verdict, runs)
    (results_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n📄 {results_dir / 'evaluation_report.csv'}\n📄 {results_dir / 'curation_verdict.csv'}")


if __name__ == "__main__":
    main()
