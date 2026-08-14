#!/usr/bin/env python3
"""
Split-adjust Trades and Quotes to be consistent with the split-adjusted OHLCV.

Reads the split records written by detect_split_adjustments.py
(data/corporate_actions/split_adjustments_<aggregate>.csv) and, for each record,
adjusts the ticker's trades and quotes CSV files so prices/volumes match the
adjusted OHLCV basis:

  * PRICE columns  (vwap; avg_bid/ask/spread/mid/micro/spread_volatility) are
    divided by `factor` for bars before the split date.
  * SIZE columns   (volume/buy_volume/sell_volume/delta/avg_trade_size/...;
                    bid_size/ask_size) are multiplied by `factor`.
  * Counts and ratios are left unchanged.

The original files are backed up to a `split-unadjusted/` subdirectory the first
time; re-running is idempotent (it always re-adjusts from the raw backup).

Usage:
    python scripts/options/adjust_trades_quotes_for_splits.py --year 2024-2025 --dry-run
    python scripts/options/adjust_trades_quotes_for_splits.py --year 2014-2025 --aggregate 1min
"""

import argparse
import csv
import datetime
import logging
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("adjust_tq")

DATA_TRADES = Path("data/trades")
DATA_QUOTES = Path("data/quotes")

TRADES_PRICE_COLS = ["vwap"]
TRADES_SIZE_COLS = [
    "volume", "buy_volume", "sell_volume", "delta", "cumulative_delta",
    "avg_trade_size", "median_trade_size", "largest_trade", "stddev_trade_size",
]
QUOTES_PRICE_COLS = [
    "avg_bid", "avg_ask", "avg_spread", "max_spread", "min_spread",
    "mid_price", "microprice", "spread_volatility",
]
QUOTES_SIZE_COLS = ["bid_size", "ask_size"]


def parse_years(year_arg):
    parts = year_arg.split("-")
    if len(parts) == 1:
        return [parts[0]]
    return [str(y) for y in range(int(parts[0]), int(parts[1]) + 1)]


def load_splits(path):
    if not Path(path).exists():
        raise SystemExit(f"Error: splits file not found: {path}. Run detect_split_adjustments.py first.")
    with open(path) as f:
        return list(csv.DictReader(f))


def _resolve(base, agg, year, name):
    p = base / agg / year / name
    if p.exists():
        return p
    gz = base / agg / year / (name + ".gz")
    return gz if gz.exists() else None


def adjust_file(src_path, split_date, factor, price_cols, size_cols, dry_run):
    """Adjust a single CSV (or .csv.gz) in place; returns (rows_before, rows_adjusted)."""
    gzip = str(src_path).endswith(".gz")
    df = pd.read_csv(src_path)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    mask = ts < pd.Timestamp(split_date, tz="UTC")
    n = int(mask.sum())
    if n == 0:
        return len(df), 0
    if dry_run:
        return len(df), n
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            df.loc[mask, col] = df.loc[mask, col] / factor
    for col in size_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            df.loc[mask, col] = df.loc[mask, col] * factor
    if gzip:
        df.to_csv(src_path, index=False, compression=dict(method="gzip", compresslevel=1))
    else:
        df.to_csv(src_path, index=False)
    return len(df), n


def main():
    p = argparse.ArgumentParser(description="Split-adjust trades and quotes")
    p.add_argument("--splits", default=None, help="path to the detected splits CSV")
    p.add_argument("--aggregate", default="1sec")
    p.add_argument("--year", default=None, help="restrict to years, e.g. 2024-2025")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    agg = args.aggregate
    splits_path = args.splits or f"data/corporate_actions/split_adjustments_{agg}.csv"
    splits = load_splits(splits_path)
    years = parse_years(args.year) if args.year else None

    for rec in splits:
        if years and rec["year"] not in years:
            continue
        ticker, year = rec["ticker"], rec["year"]
        split_date = rec["split_date"]
        factor = float(rec["factor"])

        tq = [
            (_resolve(DATA_TRADES, agg, year, f"{ticker}_{year}_{agg}_trades.csv"),
             TRADES_PRICE_COLS, TRADES_SIZE_COLS, "trades"),
            (_resolve(DATA_QUOTES, agg, year, f"{ticker}_{year}_{agg}_quotes.csv"),
             QUOTES_PRICE_COLS, QUOTES_SIZE_COLS, "quotes"),
        ]

        for path, price_cols, size_cols, kind in tq:
            if path is None:
                continue
            backup_dir = path.parent / "split-unadjusted"
            backup_path = backup_dir / path.name

            if not backup_path.exists():
                if args.dry_run:
                    log.info("[dry-run] %s %s: would back up %s", ticker, kind, path.name)
                else:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup_path)
                    log.info("%s %s: backed up -> %s", ticker, kind, backup_path)
            elif not args.dry_run:
                shutil.copy2(backup_path, path)

            if not args.dry_run:
                total, adj = adjust_file(path, split_date, factor, price_cols, size_cols, dry_run=False)
                log.info("%s %s: adjusted %d/%d rows (factor %.6f)", ticker, kind, adj, total, factor)
            else:
                total, adj = adjust_file(path, split_date, factor, price_cols, size_cols, dry_run=True)
                log.info("[dry-run] %s %s: would adjust %d/%d rows (factor %.6f)", ticker, kind, adj, total, factor)

    if args.dry_run:
        log.info("Dry run complete (no changes written).")


if __name__ == "__main__":
    main()
