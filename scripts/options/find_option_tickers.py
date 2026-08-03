"""
Scan cached options flat files to determine which tickers have tradable options data.

Reads the pre-downloaded minute_aggs files in tmp/options_cache_<year>/ and checks
for the presence of option records (O:<TICKER>...) for each requested ticker.
Outputs eligible tickers to option_tickers_avail_<year>.txt in the --output directory.

Usage:
    # Check specific tickers
    python scripts/options/find_option_tickers.py --tickers AAPL,TSLA --year 2025 --use_local_cache

    # Check all tickers from a file
    python scripts/options/find_option_tickers.py --tickers_file data/universes/2025/combined_unique.csv --year 2025 --use_local_cache

    # Check tickers with existing OHLCV data
    python scripts/options/find_option_tickers.py --ohlcv_tickers --year 2025 --use_local_cache --output data/combined

    # For .csv.gz cache files (use --use_unzipped False)
    python scripts/options/find_option_tickers.py --ohlcv_tickers --year 2025 --use_local_cache --use_unzipped False

Output:
    <output>/option_tickers_avail_<year>.txt  — tickers found (CSV with 'ticker' header)
    <output>/option_tickers_missing_<year>.txt — tickers NOT found

Note: scanning 250 .csv.gz files (~3.5 GB compressed) takes a few minutes. Unzipped
.csv files are faster (no decompression overhead).
"""

import argparse
import csv
import gzip
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

SCRIPT_NAME = Path(__file__).resolve().stem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find which tickers have tradable options in cached flat files"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker symbols (e.g. AAPL,TSLA,NVDA)",
    )
    parser.add_argument(
        "--tickers_file",
        type=str,
        default=None,
        help="Path to CSV with ticker list (header 'ticker')",
    )
    parser.add_argument(
        "--ohlcv_tickers",
        action="store_true",
        default=False,
        help="Derive ticker list from saved OHLCV files in data/SPY/1min/<year>/",
    )
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help="Year to check (e.g. 2025)",
    )
    parser.add_argument(
        "--use_local_cache",
        action="store_true",
        default=False,
        help="Read cached flat files from tmp/options_cache_<year>/",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory path. Overrides tmp/options_cache_<year>/.",
    )
    parser.add_argument(
        "--use_unzipped",
        type=lambda s: s.lower() == "true",
        default=True,
        help="Expect .csv files (True) or .csv.gz files (False) in cache dir. Default: True.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the ticker list. Default: data/",
    )
    return parser.parse_args()


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


def load_tickers(args) -> list[str]:
    tickers = []
    if args.tickers:
        tickers.extend(clean_ticker(t) for t in args.tickers.split(",") if t.strip())
    if args.tickers_file:
        with open(args.tickers_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("ticker", "").strip()
                if t:
                    tickers.append(clean_ticker(t))
    if args.ohlcv_tickers:
        year = args.year.split("-")[0]
        src_dir = Path("data") / "SPY" / "1min" / year
        if not src_dir.exists():
            raise SystemExit("Error: OHLCV directory not found: %s" % src_dir)
        for f in sorted(src_dir.glob(f"*_{year}_1min.csv*")):
            name = f.name.replace(".csv.gz", "").replace(".csv", "")
            ticker = name.split("_")[0]
            tickers.append(clean_ticker(ticker))
    if not tickers:
        raise SystemExit("Error: specify at least one of --tickers, --tickers_file, or --ohlcv_tickers")
    return list(dict.fromkeys(tickers))  # deduplicate, preserve order


def scan_cache_for_tickers(cache_dir: str, tickers: set[str], ext: str, verbose: bool = True) -> tuple[set[str], set[str]]:
    found: set[str] = set()
    missing: set[str] = set(tickers)

    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(ext))
    if not files:
        print("  No %s files found in %s" % (ext, cache_dir))
        return found, missing

    print("  Scanning %d files for %d tickers ..." % (len(files), len(tickers)))
    t0 = time.time()

    for i, filename in enumerate(files, 1):
        if not missing:
            break
        path = os.path.join(cache_dir, filename)
        try:
            opener = gzip.open if ext == ".csv.gz" else open
            with opener(path, "rt", errors="replace") as f:
                for line in f:
                    if not line.startswith("O:"):
                        continue
                    for ticker in list(missing):
                        if line.startswith("O:" + ticker):
                            found.add(ticker)
                            missing.discard(ticker)
                    if not missing:
                        break
        except (OSError, gzip.BadGzipFile) as e:
            print("  [WARN] Error reading %s: %s" % (filename, e))
        if verbose and i % 25 == 0:
            elapsed = time.time() - t0
            print("    %d/%d files | %d found, %d remaining (%.1fs)" % (
                i, len(files), len(found), len(missing), elapsed))

    elapsed = time.time() - t0
    print("  Done: %d found, %d missing (%.1fs)" % (len(found), len(missing), elapsed))
    return found, missing


def main():
    args = parse_args()
    overall_start = time.time()

    tickers = load_tickers(args)
    if not tickers:
        raise SystemExit("Error: no tickers to check")

    year = args.year.split("-")[0]
    ext = ".csv" if args.use_unzipped else ".csv.gz"
    cache_dir = args.cache_dir or f"tmp/options_cache_{year}"

    if not os.path.isdir(cache_dir):
        raise SystemExit("Error: cache directory not found: %s" % cache_dir)

    output_base = Path(args.output) if args.output else Path("data")
    out_path = output_base / f"option_tickers_avail_{year}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OPTION TICKER SCANNER  [year %s]" % year)
    print("  Tickers to check: %d" % len(tickers))
    print("  Cache dir:  %s" % cache_dir)
    print("  File ext:   %s" % ext)
    print("  Output:     %s" % out_path)
    print("=" * 60)
    print()

    found, missing = scan_cache_for_tickers(cache_dir, set(tickers), ext)

    print()
    print("=" * 60)
    print("RESULTS")
    print("  Checked:  %d" % len(tickers))
    print("  Found:    %d" % len(found))
    print("  Missing:  %d" % len(missing))
    total_elapsed = time.time() - overall_start
    print("  Duration: %.1fs" % total_elapsed)
    print("=" * 60)

    with open(out_path, "w") as f:
        f.write("ticker\n")
        for t in sorted(found):
            f.write(t + "\n")
    print()
    print("  Wrote %d tickers to %s" % (len(found), out_path))

    if missing:
        missing_path = output_base / f"option_tickers_missing_{year}.txt"
        with open(missing_path, "w") as f:
            f.write("ticker\n")
            for t in sorted(missing):
                f.write(t + "\n")
        print("  Wrote %d missing tickers to %s" % (len(missing), missing_path))


if __name__ == "__main__":
    main()
