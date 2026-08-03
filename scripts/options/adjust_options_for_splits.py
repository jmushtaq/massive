"""
Post-process options CSV files to adjust option prices and strikes for stock splits,
making them consistent with the split-adjusted underlying OHLCV data.

Reads split metadata from data/corporate_actions/splits.csv and applies cumulative
adjustment factors to option-specific columns. Underlying OHLCV columns (open, high,
low, close, vwap, underlying_price) are already adjusted by the API and are left
as-is.

Output is written to an `adjusted/` subdirectory within each year's output folder
by default, or to a custom base directory via --output.

Usage:
    # Adjust specific year (writes to adjusted/ subdir within the year folder)
    python scripts/options/adjust_options_for_splits.py --year 2023

    # Adjust multiple years
    python scripts/options/adjust_options_for_splits.py --year 2022-2023

    # Custom input directory (CSV files at <input>/<year>/<ticker>_...csv)
    python scripts/options/adjust_options_for_splits.py --year 2023 --input /tmp/options_staging

    # Write to custom output base
    python scripts/options/adjust_options_for_splits.py --year 2023 --output data/options/stocks_adjusted

    # Dry-run (no changes, report only)
    python scripts/options/adjust_options_for_splits.py --year 2023 --dry-run
"""

import argparse
import csv
import datetime
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

SCRIPT_NAME = Path(__file__).resolve().stem

AGGREGATE_MAP = {
    "1sec": "1sec",
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1H": "1H",
    "4H": "4H",
    "1D": "1D",
}

# Columns from the option flat files that contain raw (unadjusted) prices or strikes.
ADJUST_COLUMNS = [
    "atm_strike",
    "atm_call_close",
    "atm_put_close",
    "iv30d_strike",
    "iv30d_call_close",
    "iv30d_put_close",
]

# Derived columns that must be recomputed after adjusting their inputs.
DERIVED_COLUMNS = {
    "atm_straddle_price": ("atm_call_close", "atm_put_close", "sum"),
    "expected_move": ("atm_call_close", "atm_put_close", "sum"),
}

CSV_HEADERS = [
    "ticker",
    "timestamp",
    "underlying_price",
    "atm_strike",
    "atm_call_close",
    "atm_put_close",
    "atm_straddle_price",
    "expected_move",
    "put_volume",
    "call_volume",
    "put_call_ratio",
    "contract_count",
    "avg_bid_ask_spread",
    "avg_bid_size",
    "avg_ask_size",
    "quote_imbalance",
    "atm_days_to_expiry",
    "iv30d_strike",
    "iv30d_call_close",
    "iv30d_put_close",
    "iv30d_days_to_expiry",
    "open",
    "high",
    "low",
    "close",
    "vwap",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adjust options CSV files for stock splits"
    )
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help="Year to adjust (e.g. 2023)",
    )
    parser.add_argument(
        "--aggregate",
        choices=list(AGGREGATE_MAP.keys()),
        default="1min",
        help="Aggregate window size (default: 1min)",
    )
    parser.add_argument(
        "--splits_file",
        type=str,
        default="data/corporate_actions/splits.csv",
        help="Path to splits CSV (default: data/corporate_actions/splits.csv)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Directory containing option CSV files. When set, CSV files are looked up at <input>/<year>/<ticker>_<year>_<agg>_options.csv. Default: data/options/stocks/<agg>/<year>/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output directory. If omitted, writes to an 'adjusted/' subfolder within each year directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Report what would be adjusted without writing any files",
    )
    return parser.parse_args()


def load_splits(splits_path: str) -> dict[str, list[dict]]:
    splits_by_ticker: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    if not os.path.exists(splits_path):
        raise SystemExit("Error: splits file not found: %s" % splits_path)
    with open(splits_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"].strip().upper()
            execution_date = row["execution_date"].strip()
            split_from = int(row["split_from"])
            split_to = int(row["split_to"])
            key = (ticker, execution_date)
            if key in seen:
                continue
            seen.add(key)
            splits_by_ticker[ticker].append({
                "execution_date": datetime.date.fromisoformat(execution_date),
                "split_from": split_from,
                "split_to": split_to,
                "ratio": split_to / split_from,
            })
    for ticker in splits_by_ticker:
        splits_by_ticker[ticker].sort(key=lambda s: s["execution_date"], reverse=True)
    return dict(splits_by_ticker)


def cumulative_factor_for_date(splits: list[dict], row_date: datetime.date) -> float:
    factor = 1.0
    for s in splits:
        if s["execution_date"] > row_date:
            factor *= s["ratio"]
        else:
            break
    return factor


def safe_float(value: str) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def adjust_row(row: dict[str, str], factor: float) -> dict[str, str]:
    adjusted = dict(row)
    for col in ADJUST_COLUMNS:
        val = safe_float(row.get(col, ""))
        if val is not None:
            adjusted[col] = str(val * factor)
        else:
            adjusted[col] = row.get(col, "")
    call_close = safe_float(adjusted.get("atm_call_close", ""))
    put_close = safe_float(adjusted.get("atm_put_close", ""))
    straddle = ""
    if call_close is not None and put_close is not None:
        straddle = str(call_close + put_close)
    elif call_close is not None:
        straddle = str(call_close)
    elif put_close is not None:
        straddle = str(put_close)
    adjusted["atm_straddle_price"] = straddle
    adjusted["expected_move"] = straddle
    return adjusted


def validate_row(row: dict[str, str], ticker: str, row_num: int, logger: logging.Logger) -> bool:
    underlying = safe_float(row.get("underlying_price", ""))
    atm_strike = safe_float(row.get("atm_strike", ""))
    ok = True
    if underlying and atm_strike and underlying > 0:
        ratio = abs(atm_strike - underlying) / underlying
        if ratio > 0.25:
            logger.debug(
                "  [%s row %d] atm_strike (%s) far from underlying_price (%s) ratio=%.3f",
                ticker, row_num, row.get("atm_strike"), row.get("underlying_price"), ratio,
            )
    for col in ADJUST_COLUMNS:
        val = safe_float(row.get(col, ""))
        if val is not None and val < -0.0001:
            logger.warning("  [%s row %d] negative value in %s: %s", ticker, row_num, col, row.get(col))
            ok = False
    straddle = safe_float(row.get("atm_straddle_price", ""))
    expected = safe_float(row.get("expected_move", ""))
    if straddle is not None and expected is not None:
        if abs(straddle - expected) > 0.0001:
            logger.warning(
                "  [%s row %d] atm_straddle_price (%s) != expected_move (%s)",
                ticker, row_num, row.get("atm_straddle_price"), row.get("expected_move"),
            )
            ok = False
    if underlying and underlying > 0 and straddle and straddle > 0:
        if straddle > underlying * 0.5:
            logger.debug(
                "  [%s row %d] straddle/underlying ratio high: straddle=%s underlying=%s",
                ticker, row_num, straddle, underlying,
            )
    return ok


def adjust_file(input_path: Path, output_path: Path, splits: list[dict],
                logger: logging.Logger, dry_run: bool = False) -> dict:
    stats = {"rows": 0, "adjusted_rows": 0, "validation_errors": 0, "skipped": False}
    ticker = input_path.stem.split("_")[0]

    if not input_path.exists():
        logger.warning("  [%s] file not found: %s", ticker, input_path)
        stats["skipped"] = True
        return stats

    rows = []
    with open(input_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    if not rows:
        logger.info("  [%s] empty file, skipping: %s", ticker, input_path.name)
        stats["skipped"] = True
        return stats

    if not splits:
        logger.info("  [%s] no splits, copying as-is: %s", ticker, input_path.name)
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(input_path) as fin, open(output_path, "w", newline="") as fout:
                fout.write(fin.read())
        stats["rows"] = len(rows)
        stats["skipped"] = True
        return stats

    adjusted_rows = []
    adjusted_count = 0
    row_num = 0
    for row in rows:
        row_num += 1
        ts_str = row.get("timestamp", "")
        row_date = None
        try:
            dt = datetime.datetime.fromisoformat(ts_str.replace("+00:00", ""))
            row_date = dt.date()
        except (ValueError, TypeError):
            pass

        factor = 1.0
        if row_date is not None:
            factor = cumulative_factor_for_date(splits, row_date)

        if abs(factor - 1.0) > 1e-9:
            adjusted = adjust_row(row, factor)
            adjusted_rows.append(adjusted)
            adjusted_count += 1
        else:
            adjusted_rows.append(dict(row))

    if adjusted_count == 0:
        logger.info("  [%s] all splits before data range, no adjustment needed: %s", ticker, input_path.name)

    stats["rows"] = len(rows)
    stats["adjusted_rows"] = adjusted_count

    if dry_run:
        logger.info(
            "  [%s] DRY-RUN: %d rows, %d would change (%d splits affecting this file)",
            ticker, len(rows), adjusted_count,
            sum(1 for s in splits if s["execution_date"].year <= int(input_path.stem.split("_")[1])),
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for row in adjusted_rows:
                writer.writerow({k: row.get(k, "") for k in CSV_HEADERS})

        validation_errors = 0
        for i, row in enumerate(adjusted_rows, 1):
            if not validate_row(row, ticker, i, logger):
                validation_errors += 1
        stats["validation_errors"] = validation_errors

        logger.info(
            "  [%s] %d rows → %s  (validation errors: %d)",
            ticker, len(rows), output_path, validation_errors,
        )

    return stats


def main():
    args = parse_args()
    overall_start = time.time()

    log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(SCRIPT_NAME)

    splits_path = args.splits_file
    all_splits = load_splits(splits_path)
    tickers_with_splits = sorted(all_splits.keys())

    logger.info("=" * 60)
    logger.info("OPTIONS SPLIT ADJUSTMENT")
    logger.info("  Splits file:  %s", splits_path)
    logger.info("  Tickers with splits: %s", ", ".join(tickers_with_splits))
    logger.info("  Aggregate:    %s", args.aggregate)
    logger.info("  Dry-run:      %s", args.dry_run)
    logger.info("=" * 60)

    if args.input:
        year_dir_pattern = Path(args.input)
    else:
        year_dir_pattern = Path("data") / "options" / "stocks" / AGGREGATE_MAP[args.aggregate]
    if not year_dir_pattern.exists():
        raise SystemExit("Error: input directory not found: %s" % year_dir_pattern)

    years = [y.strip() for y in args.year.split(",")]
    if not years:
        raise SystemExit("Error: no valid years specified")

    logger.info("  Years:        %s", ", ".join(years))

    output_base = Path(args.output) if args.output else None

    all_stats: list[dict] = []
    total_rows = 0
    total_adjusted = 0
    total_files = 0
    total_errors = 0

    for year in years:
        input_dir = year_dir_pattern / year
        if not input_dir.exists():
            logger.warning("  Year dir not found: %s", input_dir)
            continue

        opts_files = sorted(input_dir.glob(f"*_{year}_{AGGREGATE_MAP[args.aggregate]}_options.csv"))
        if not opts_files:
            logger.warning("  No options files found in %s", input_dir)
            continue

        logger.info("")
        logger.info("--- Year %s (%d files) ---", year, len(opts_files))

        for input_path in opts_files:
            ticker = input_path.stem.split("_")[0]
            total_files += 1

            if output_base:
                out_dir = output_base / "options" / "stocks" / AGGREGATE_MAP[args.aggregate] / year
            else:
                out_dir = input_dir / "adjusted"
            output_path = out_dir / input_path.name

            ticker_splits = all_splits.get(ticker, [])

            t0 = time.time()
            stats = adjust_file(input_path, output_path, ticker_splits, logger, args.dry_run)
            stats["ticker"] = ticker
            stats["year"] = year
            stats["elapsed_s"] = round(time.time() - t0, 2)
            all_stats.append(stats)

            total_rows += stats.get("rows", 0)
            total_adjusted += stats.get("adjusted_rows", 0)
            total_errors += stats.get("validation_errors", 0)

    total_time = time.time() - overall_start
    total_would_change = sum(s.get("adjusted_rows", 0) for s in all_stats if not s.get("skipped"))

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Duration:       %.1fs", total_time)
    logger.info("  Years:          %s", ", ".join(years))
    logger.info("  Files:          %d", total_files)
    logger.info("  Total rows:     %d", total_rows)
    if args.dry_run:
        logger.info("  Would adjust:   %d rows", total_would_change)
    else:
        logger.info("  Validation err: %d", total_errors)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
