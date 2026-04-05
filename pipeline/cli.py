"""
CLI entry point for the Governed Decomposition Pipeline.

Usage:
  gov-pipeline run --corpus-dir <dir> --output-dir <dir>
  gov-pipeline stats --corpus-dir <dir>
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Governed Decomposition Pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run the full S1-S5 pipeline")
    run_parser.add_argument("--corpus-dir", required=True, help="Input corpus directory")
    run_parser.add_argument("--output-dir", required=True, help="Output directory for CORPUS-SEMI-001")

    stats_parser = sub.add_parser("stats", help="Compute corpus statistics")
    stats_parser.add_argument("--corpus-dir", required=True, help="CORPUS-SEMI-001 directory")

    args = parser.parse_args()

    if args.command == "run":
        print(f"Pipeline run: {args.corpus_dir} -> {args.output_dir}")
        print("Not yet implemented — stages S1-S5 are scaffolded, awaiting corpus acquisition.")
        sys.exit(0)
    elif args.command == "stats":
        print(f"Statistics for: {args.corpus_dir}")
        print("Not yet implemented.")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
