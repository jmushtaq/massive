"""
For each year in data/universes/<year>/, create a combined_unique.csv containing
all tickers from top1000_liquidity, large_cap, mid_cap, small_cap, mini_cap,
sorted alphabetically, excluding any ticker that already has data in a target directory.

Typical use: generate a ticker list for downloading data that's not yet saved.
Feed the output into any download script via --tickers_file to fill gaps.

Usage:
    # basic — processes all <year>/ dirs under data/universes/
    python scripts/generate_missing_tickers.py --target 'data/SPY/1min/<year>/'

    # specific years
    python scripts/generate_missing_tickers.py \
        --target 'data/SPY/1min/<year>/' \
        --years data/universes/2003 data/universes/2004

    # feed into download script
    python scripts/stocks_aggs_download.py \
        --tickers_file data/universes/2025/combined_unique.csv \
        --year 2025 --aggregate 1min

Output: <universe_dir>/combined_unique.csv  (header: ticker)
"""

import argparse
import csv
import sys
from pathlib import Path

UNIVERSE_FILES = [
    "top1000_liquidity.csv",
    "large_cap_liquidity.csv",
    "mid_cap_liquidity.csv",
    "small_cap_liquidity.csv",
    "mini_cap_liquidity.csv",
]


def load_existing(target_dir: Path | str) -> set[str]:
    existing: set[str] = set()
    target = Path(target_dir)
    if not target.exists():
        return existing
    if target.is_file():
        return existing
    for f in target.iterdir():
        if f.suffix == ".csv":
            ticker = f.stem.split("_")[0]
            existing.add(ticker)
    return existing


def load_universe_tickers(universe_dir: Path) -> set[str]:
    tickers: set[str] = set()
    for name in UNIVERSE_FILES:
        path = universe_dir / name
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                t = row.get("ticker", "").strip()
                if t:
                    tickers.add(t)
    return tickers


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined_unique.csv per universe year, excluding existing tickers"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target directory pattern to check for existing files (e.g. data/SPY/1min/<year>/)",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        default=None,
        help="Specific universe directories (default: all under data/universes/)",
    )
    args = parser.parse_args()

    target_pattern = args.target

    if args.years:
        universe_dirs = [Path(d) for d in args.years]
    else:
        base = Path("data") / "universes"
        universe_dirs = sorted(
            base / d.name for d in base.iterdir() if d.is_dir() and d.name.isdigit()
        )

    if not universe_dirs:
        print("No universe directories found", file=sys.stderr)
        sys.exit(1)

    for u_dir in universe_dirs:
        if not u_dir.exists():
            print(f"  skipping {u_dir} (not found)", file=sys.stderr)
            continue

        year = u_dir.name
        target_dir = target_pattern.replace("<year>", year)
        existing = load_existing(target_dir)

        all_tickers = load_universe_tickers(u_dir)
        missing = sorted(all_tickers - existing)

        out_path = u_dir / "combined_unique.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker"])
            for t in missing:
                writer.writerow([t])

        total = len(all_tickers)
        skipped = len(existing & all_tickers)
        print(f"  {year}: {len(missing)} missing / {total} total, {skipped} already in {target_dir} -> {out_path}")


if __name__ == "__main__":
    main()
