#!/usr/bin/env python3
"""
Verify that OHLCV, Trades and Quotes are on a consistent (split-adjusted)
price basis across every detected split.

For each record in data/corporate_actions/split_adjustments_<aggregate>.csv, it
computes the daily median ratio  trades_vwap / ohlcv_close  (and
avg_bid / ohlcv_close) on both sides of the split date. A consistent dataset
shows ratio ~ 1.0 on both sides; a raw (unadjusted) dataset shows a large step.

Exit code 0 = all consistent; 1 = inconsistencies remain.

Usage:
    python scripts/options/check_split_consistency.py --year 2024-2025
    python scripts/options/check_split_consistency.py --year 2014-2025 --aggregate 1min
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("check_splits")

DATA_OHLCV = Path("data/SPY")
DATA_TRADES = Path("data/trades")
DATA_QUOTES = Path("data/quotes")

TOL = 1.15  # a consistent dataset stays within +/-15% of 1.0


def _resolve(base, agg, year, name):
    p = base / agg / year / name
    if p.exists():
        return p
    gz = base / agg / year / (name + ".gz")
    return gz if gz.exists() else None


def _daily_ratio(price_file, close_file, price_col):
    p = pd.read_csv(price_file, usecols=["timestamp", price_col])
    p["t"] = pd.to_datetime(p["timestamp"], utc=True)
    p = p.set_index("t")
    c = pd.read_csv(close_file, usecols=["timestamp", "close"])
    c["t"] = pd.to_datetime(c["timestamp"], utc=True)
    c = c.set_index("t")
    return (p[price_col].resample("1D").mean() / c["close"].resample("1D").last()).dropna()


def check_dataset(ticker, year, agg, split_date, price_file, price_col):
    close_file = _resolve(DATA_OHLCV, agg, year, f"{ticker}_{year}_{agg}.csv")
    if not close_file:
        return "NO_OHLCV"
    if price_file is None:
        return "MISSING"
    r = _daily_ratio(price_file, close_file, price_col)
    r = r[r > 0]
    s = pd.Timestamp(split_date, tz="UTC")
    pre = r[r.index < s].median()
    post = r[r.index >= s].median()
    ok = (pre / post if post else 1.0)
    return ("OK" if (TOL**-1 <= ok <= TOL) else f"STEP x{ok:.3f} (pre={pre:.3f} post={post:.3f})")


def main():
    p = argparse.ArgumentParser(description="Verify split consistency of trades/quotes vs OHLCV")
    p.add_argument("--splits", default=None, help="path to the detected splits CSV")
    p.add_argument("--aggregate", default="1sec")
    p.add_argument("--year", default=None)
    args = p.parse_args()

    agg = args.aggregate
    splits_path = args.splits or f"data/corporate_actions/split_adjustments_{agg}.csv"
    with open(splits_path) as f:
        splits = list(csv.DictReader(f))

    years = None
    if args.year:
        parts = args.year.split("-")
        years = {parts[0]} if len(parts) == 1 else {str(y) for y in range(int(parts[0]), int(parts[1]) + 1)}

    bad = 0
    for rec in splits:
        if years and rec["year"] not in years:
            continue
        ticker, year, split_date = rec["ticker"], rec["year"], rec["split_date"]
        tr = check_dataset(ticker, year, agg, split_date,
                           _resolve(DATA_TRADES, agg, year, f"{ticker}_{year}_{agg}_trades.csv"), "vwap")
        qu = check_dataset(ticker, year, agg, split_date,
                           _resolve(DATA_QUOTES, agg, year, f"{ticker}_{year}_{agg}_quotes.csv"), "avg_bid")
        state = "PASS" if (tr == "OK" and qu == "OK") else "FAIL"
        if state == "FAIL":
            bad += 1
        print(f"  [{state}] {ticker} {year} split={split_date} factor={rec['factor']}: "
              f"trades={tr}  quotes={qu}")

    print(f"\n  {'ALL CONSISTENT' if bad == 0 else f'{bad} datasets still inconsistent'}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
