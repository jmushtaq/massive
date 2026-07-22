"""
Find tickers present in one directory but missing in another.

Extracts ticker names from CSV filenames and reports which tickers
exist in the reference dir but not in the target dir.

Usage:
    python scripts/find_missing_tickers.py --reference data/SPY/1min/2022 --target data/quotes/1min/2022 --output missing_tickers.txt
"""

import argparse
import sys
from pathlib import Path


def get_tickers(directory: Path) -> set[str]:
    tickers: set[str] = set()
    for f in directory.iterdir():
        if f.suffix != ".csv":
            continue
        name = f.stem
        ticker = name.split("_")[0]
        tickers.add(ticker)
    return tickers


def main():
    parser = argparse.ArgumentParser(
        description="Find tickers in reference dir that are missing from target dir"
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference directory (source of truth for tickers)",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to target directory to check for missing tickers",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    ref_path = Path(args.reference)
    target_path = Path(args.target)

    if not ref_path.is_dir():
        print(f"Error: reference directory not found: {ref_path}", file=sys.stderr)
        sys.exit(1)
    if not target_path.is_dir():
        print(f"Error: target directory not found: {target_path}", file=sys.stderr)
        sys.exit(1)

    ref_tickers = get_tickers(ref_path)
    target_tickers = get_tickers(target_path)
    missing = sorted(ref_tickers - target_tickers)

    out = sys.stdout if args.output is None else open(args.output, "w")
    with out:
        print("ticker", file=out)
        for t in missing:
            print(t, file=out)

    if args.output:
        print(f"Wrote {len(missing)} missing tickers to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
