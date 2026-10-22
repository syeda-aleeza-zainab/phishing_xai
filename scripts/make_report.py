"""CLI entry point to generate consolidated RESULTS.md and summary.json."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phishing_xai.report import generate_consolidated_report
from phishing_xai.utils import get_logger

logger = get_logger("make_report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate consolidated experimental results report")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing experiment artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    md_path, json_path = generate_consolidated_report(args.results_dir)
    print(f"Report generated successfully:\n  Markdown: {md_path}\n  JSON:     {json_path}")


if __name__ == "__main__":
    main()
