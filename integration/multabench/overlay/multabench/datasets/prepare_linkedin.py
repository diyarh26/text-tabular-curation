"""Export the selected LinkedIn CSV into native typed dataset views."""
import argparse
from multabench.datasets.linkedin_salary import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.csv, args.output)
