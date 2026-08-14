#!/usr/bin/env python3
"""
Detect split-related price-scale inconsistencies between Trades/Quotes and the
split-adjusted OHLCV.

Background
----------
* OHLCV (stocks_aggs_download.py) and option chains use `list_aggs(adjusted=True)`
  -> split-adjusted prices (and volume).
* Trades (list_trades) and Quotes (list_quotes) have NO `adjusted` parameter and
  are returned RAW by the API for some tickers/splits (empirically inconsistent:
  NVDA 2024 is raw, CMG 2025 is adjusted).

This script finds tickers whose Trades/Quotes are on a DIFFERENT price scale than
the adjusted OHLCV by detecting a step in the daily ratio  trades_vwap / ohlcv_close
(or avg_bid / ohlcv_close). For each such ticker it emits a split record with the
first post-split trading day and the price divisor (factor).

Output: data/corporate_actions/split_adjustments_<aggregate>.csv
        columns: ticker,year,split_date,factor,split_from,split_to,quotes_raw

Usage:
    python scripts/options/detect_split_adjustments.py --year 2024-2025
    python scripts/options/detect_split_adjustments.py --year 2014-2025 --aggregate 1min
"""

import argparse
import csv
import datetime
import logging
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("detect_splits")

DATA_OHLCV = Path("data/SPY")
DATA_TRADES = Path("data/trades")
DATA_QUOTES = Path("data/quotes")


def parse_years(year_arg):
    parts = year_arg.split("-")
    if len(parts) == 1:
        return [parts[0]]
    return [str(y) for y in range(int(parts[0]), int(parts[1]) + 1)]


DETECT_AGG = "1min"  # detection uses 1min (fast); split date/factor are aggregate-independent


def _resolve(base, year, agg, name):
    """Return the path to an existing .csv or .csv.gz file, else None."""
    p = base / agg / year / name
    if p.exists():
        return p
    gz = base / agg / year / (name + ".gz")
    return gz if gz.exists() else None


def _daily_ratio(ticker, year, agg=DETECT_AGG):
    oc = _resolve(DATA_OHLCV, year, agg, f"{ticker}_{year}_{agg}.csv")
    tc = _resolve(DATA_TRADES, year, agg, f"{ticker}_{year}_{agg}_trades.csv")
    if not oc or not tc:
        return None
    o = pd.read_csv(oc, usecols=["timestamp", "close"])
    o["t"] = pd.to_datetime(o["timestamp"], utc=True)
    o = o.set_index("t")
    t = pd.read_csv(tc, usecols=["timestamp", "vwap"])
    t["t"] = pd.to_datetime(t["timestamp"], utc=True)
    t = t.set_index("t")
    return (t["vwap"].resample("1D").mean() / o["close"].resample("1D").last()).dropna()


def _round_factor(f):
    if f > 1:
        for c in (1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 50, 100, 200):
            if abs(f - c) / c < 0.02:
                return float(c)
        return f
    if 0 < f < 1:
        return 1.0 / _round_factor(1.0 / f)
    return f


def detect_ticker(ticker, year, agg="1sec"):
    r = _daily_ratio(ticker, year)
    if r is None or len(r) < 40:
        return None
    r = r[r > 0]

    # A day is "divergent" when its price scale differs materially from adjusted OHLCV.
    divergent = (r > 1.2) | (r < 1.0 / 1.2)
    if divergent.sum() == 0 or divergent.all():
        return None

    # first consistent day that follows at least one divergent day = split boundary
    split_idx = None
    for i in range(1, len(r)):
        if (not divergent.iloc[i]) and divergent.iloc[:i].any():
            split_idx = i
            break
    if split_idx is None:
        return None

    split_date = r.index[split_idx].date()
    pre = r[r.index < pd.Timestamp(split_date, tz="UTC")]
    post = r[r.index >= pd.Timestamp(split_date, tz="UTC")]
    if pre.empty or post.empty or len(pre) < 10 or len(post) < 10:
        return None

    raw_factor = pre.median() / post.median()
    # real splits are material and span many days; drop noise/small ratios
    if not (raw_factor > 1.25 or raw_factor < 1.0 / 1.25):
        return None
    if divergent.iloc[:split_idx].mean() < 0.5:
        return None

    factor = _round_factor(raw_factor)
    fr = Fraction(factor).limit_denominator(1000)
    split_from, split_to = fr.denominator, fr.numerator

    qc = _resolve(DATA_QUOTES, year, agg, f"{ticker}_{year}_{agg}_quotes.csv")
    quotes_raw = False
    if qc:
        q = pd.read_csv(qc, usecols=["timestamp", "avg_bid"])
        q["t"] = pd.to_datetime(q["timestamp"], utc=True)
        q = q.set_index("t")
        oc = _resolve(DATA_OHLCV, year, agg, f"{ticker}_{year}_{agg}.csv")
        if oc:
            o = pd.read_csv(oc, usecols=["timestamp", "close"])
            o["t"] = pd.to_datetime(o["timestamp"], utc=True)
            o = o.set_index("t")
            qr = (q["avg_bid"].resample("1D").mean() / o["close"].resample("1D").last()).dropna()
            qr = qr[qr > 0]
            qpre = qr[qr.index < pd.Timestamp(split_date, tz="UTC")].median()
            quotes_raw = qpre > 1.2 if factor > 1 else qpre < 1.0 / 1.2

    return {
        "ticker": ticker,
        "year": year,
        "split_date": str(split_date),
        "factor": factor,
        "split_from": split_from,
        "split_to": split_to,
        "quotes_raw": quotes_raw,
    }


def _worker(args):
    ticker, year, agg = args
    try:
        return detect_ticker(ticker, year, agg)
    except Exception as e:  # noqa: BLE001
        return None


def main():
    p = argparse.ArgumentParser(description="Detect split adjustments needed for trades/quotes")
    p.add_argument("--year", default="2024-2025")
    p.add_argument("--aggregate", default="1sec")
    p.add_argument("--output", default=None)
    p.add_argument("--nprocs", type=int, default=os.cpu_count() or 1)
    args = p.parse_args()

    agg = args.aggregate
    years = parse_years(args.year)
    tickers_by_year = {}
    for y in years:
        tickers_by_year[y] = sorted({
            f.name.split("_")[0]
            for f in (DATA_TRADES / agg / y).glob(f"*_{y}_{agg}_trades.csv*")
        })
    work = [(t, y, agg) for y in years for t in tickers_by_year.get(y, [])]
    log.info("Scanning %d ticker-years across %s (%s) ...", len(work), years, agg)

    records = []
    nprocs = max(1, min(args.nprocs, len(work)))
    if nprocs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=nprocs) as pool:
            for rec in pool.map(_worker, work):
                if rec:
                    records.append(rec)
                    log.info("  %s %s: factor=%s (from %s:%s) quotes_raw=%s",
                             rec["ticker"], rec["year"], rec["factor"],
                             rec["split_from"], rec["split_to"], rec["quotes_raw"])
    else:
        for w in work:
            rec = _worker(w)
            if rec:
                records.append(rec)
                log.info("  %s %s: factor=%s (from %s:%s) quotes_raw=%s",
                         rec["ticker"], rec["year"], rec["factor"],
                         rec["split_from"], rec["split_to"], rec["quotes_raw"])

    out = Path(args.output) if args.output else Path("data/corporate_actions") / f"split_adjustments_{agg}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ticker", "year", "split_date", "factor", "split_from", "split_to", "quotes_raw"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rec in sorted(records, key=lambda r: (r["year"], r["ticker"])):
            w.writerow(rec)
    log.info("Wrote %d split records to %s", len(records), out)


if __name__ == "__main__":
    main()
