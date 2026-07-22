"""
Generate yearly universe ticker lists from OHLCV + flatfile + fundamentals data.

Universe files (data/universes/<year>/<name>.csv, format: ticker,market_cap,rank):

  top500_liquidity     Top 500 by annual dollar volume
  top1000_liquidity    Top 1000 by annual dollar volume
  top3000_liquidity    Top 3000 by annual dollar volume
  sp500                Market cap > $10B, NYSE/NASDAQ, not delisted
  large_cap_liquidity  Top 250 by dollar volume
  mid_cap_liquidity    Next 250 by dollar volume after large_cap_liquidity
  small_cap_liquidity  Next 250 by dollar volume after mid_cap_liquidity
  mini_cap_liquidity   Next 250 by dollar volume after small_cap_liquidity

  Note: all *_cap_liquidity files are ranked by dollar volume, not market cap.
  Market cap column is from fundamentals.csv (latest snapshot, not year-specific).

Data sources (priority order per year):
1. Full-year 1D OHLCV from API in data/SPY/1D/<year>/ (true annual dollar volume)
2. Year-end flatfile from data/flatfiles/stocks/1D/<date>.csv.gz (broad coverage,
   single-day dollar volume as liquidity proxy; unadjusted prices, which is fine
   for cross-sectional ranking)
3. Full-year 1min OHLCV from data/SPY/1min/<year>/ (legacy, ~600 tickers)

Filtered to common stocks (type=CS) using data/spy_tickers/ticker_types.csv.

Stats cached in data/.universe_cache/ for fast re-runs.

Usage:
    python scripts/generate_universes.py
    python scripts/generate_universes.py --min-year 2008 --max-year 2025
    python scripts/generate_universes.py --dry-run
"""

import argparse
import csv
import gzip
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

OHLCV_BASE = Path("data") / "SPY"
FLATFILE_DIR = Path("data") / "flatfiles" / "stocks" / "1D"
FUNDAMENTALS_PATH = Path("data") / "fundamentals" / "fundamentals.csv"
REFERENCE_PATH = Path("data") / "reference" / "reference.csv"
TICKER_TYPES_PATH = Path("data") / "spy_tickers" / "ticker_types.csv"
OUTPUT_BASE = Path("data") / "universes"
CACHE_DIR = Path("data") / ".universe_cache"

# Last trading day of each year (from flatfile filenames)
FLATFILE_YEARS: dict[int, str] = {}
if FLATFILE_DIR.exists():
    for f in sorted(FLATFILE_DIR.iterdir()):
        name = f.name
        if name.endswith(".csv.gz"):
            stem = name[:-7]
            try:
                d = date.fromisoformat(stem)
                FLATFILE_YEARS[d.year] = name
            except ValueError:
                pass


def load_ticker_types() -> dict[str, str]:
    types: dict[str, str] = {}
    if TICKER_TYPES_PATH.exists():
        with open(TICKER_TYPES_PATH) as f:
            for row in csv.DictReader(f):
                types[row["ticker"]] = row["type"]
    return types


def load_fundamentals() -> dict:
    fund: dict[str, dict[str, Any]] = {}
    if FUNDAMENTALS_PATH.exists():
        with open(FUNDAMENTALS_PATH) as f:
            for row in csv.DictReader(f):
                t = row.get("ticker", "").strip().upper()
                if t and t not in fund:
                    mc_str = row.get("market_cap", "").strip()
                    shares_str = row.get("shares_outstanding", "").strip()
                    fund[t] = {
                        "market_cap": float(mc_str) if mc_str else None,
                        "shares_outstanding": float(shares_str) if shares_str else None,
                    }
    return fund


def load_reference() -> dict:
    ref: dict[str, dict[str, str]] = {}
    if REFERENCE_PATH.exists():
        with open(REFERENCE_PATH) as f:
            for row in csv.DictReader(f):
                t = row.get("ticker", "").strip().upper()
                if t:
                    ref[t] = {
                        "exchange": row.get("exchange", "").strip(),
                        "sector": row.get("sector_industry", "").strip(),
                        "listing_date": row.get("listing_date", "").strip(),
                        "delisting_date": row.get("delisting_date", "").strip(),
                    }
    return ref


def get_available_years() -> list[int]:
    years: set[int] = set()
    for sub in (OHLCV_BASE / "1D").iterdir():
        if sub.is_dir() and sub.name.isdigit():
            years.add(int(sub.name))
    for sub in (OHLCV_BASE / "1min").iterdir():
        if sub.is_dir() and sub.name.isdigit():
            years.add(int(sub.name))
    years.update(FLATFILE_YEARS.keys())
    return sorted(years)


def cache_path(year: int) -> Path:
    return CACHE_DIR / f"year_{year}.json"


def compute_year_stats(
    year: int, fund: dict, ref: dict, ticker_types: dict, tickers: set[str] | None = None
) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path(year)
    cache: dict = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)

    stats: list[dict] = []
    loaded_from_cache = 0
    source = ""
    t0 = time.time()

    # Priority 1: full-year 1D API data (true annual dollar volume)
    # Only use if it has meaningful coverage (>500 tickers)
    full_year_dir = OHLCV_BASE / "1D" / str(year)
    use_1d_api = full_year_dir.exists() and not source
    if use_1d_api:
        csv_count = sum(1 for f in full_year_dir.iterdir() if f.suffix == ".csv")
        if csv_count < 500:
            use_1d_api = False
            sys.stderr.write(f"    1D dir has only {csv_count} files, falling through\n")
    if use_1d_api:
        source = "1D API"
        for fpath in sorted(full_year_dir.iterdir()):
            if fpath.suffix != ".csv":
                continue
            ticker = fpath.stem.split("_")[0]
            if ticker in cache and cache[ticker].get("dvol", 0) > 0:
                ref_info = ref.get(ticker, {})
                s = cache[ticker].copy()
                s["ticker"] = ticker
                s["sector"] = ref_info.get("sector", "")
                s["exchange"] = ref_info.get("exchange", "")
                s["listing_date"] = ref_info.get("listing_date", "")
                s["delisting_date"] = ref_info.get("delisting_date", "")
                stats.append(s)
                loaded_from_cache += 1
                continue
            try:
                with open(fpath) as f:
                    header = f.readline().strip().split(",")
                    ci = header.index("close")
                    vi = header.index("volume")
                dvol = 0.0
                close_sum = 0.0
                count = 0
                with open(fpath) as f:
                    next(f)
                    for line in f:
                        parts = line.strip().split(",")
                        try:
                            close = float(parts[ci])
                            vol = float(parts[vi])
                        except (ValueError, IndexError):
                            continue
                        dvol += close * vol
                        close_sum += close
                        count += 1
                if count == 0 or dvol == 0:
                    cache[ticker] = {"dvol": 0}
                    continue
            except Exception:
                cache[ticker] = {"dvol": 0}
                continue

            close_avg = close_sum / count
            fund_info = fund.get(ticker, {})
            ref_info = ref.get(ticker, {})
            mc_est = fund_info.get("market_cap")
            if mc_est is None and fund_info.get("shares_outstanding"):
                mc_est = close_avg * fund_info["shares_outstanding"]

            s = {
                "ticker": ticker,
                "dvol": dvol,
                "shares": fund_info.get("shares_outstanding"),
                "market_cap": mc_est,
                "sector": ref_info.get("sector", ""),
                "exchange": ref_info.get("exchange", ""),
                "listing_date": ref_info.get("listing_date", ""),
                "delisting_date": ref_info.get("delisting_date", ""),
            }
            stats.append(s)
            cache[ticker] = {
                "dvol": s["dvol"],
                "shares": s["shares"],
                "market_cap": s["market_cap"],
            }

    # Priority 2: flatfile (broad coverage, single-day liquidity proxy)
    if not source and year in FLATFILE_YEARS:
        source = "flatfile"
        ff_path = FLATFILE_DIR / FLATFILE_YEARS[year]
        try:
            with gzip.open(ff_path, "rt") as f:
                rows = list(csv.DictReader(f))

            for row in rows:
                ticker = row.get("ticker", "").strip()
                if not ticker or (tickers is not None and ticker not in tickers):
                    continue
                typ = ticker_types.get(ticker, "")
                if typ and typ != "CS":
                    continue

                if ticker in cache and cache[ticker].get("dvol", 0) > 0:
                    ref_info = ref.get(ticker, {})
                    s = cache[ticker].copy()
                    s["ticker"] = ticker
                    s["sector"] = ref_info.get("sector", "")
                    s["exchange"] = ref_info.get("exchange", "")
                    s["listing_date"] = ref_info.get("listing_date", "")
                    s["delisting_date"] = ref_info.get("delisting_date", "")
                    stats.append(s)
                    loaded_from_cache += 1
                    continue

                try:
                    close = float(row.get("close", 0))
                    vol = float(row.get("volume", 0))
                except (ValueError, TypeError):
                    cache[ticker] = {"dvol": 0}
                    continue

                dvol = close * vol
                if dvol == 0:
                    cache[ticker] = {"dvol": 0}
                    continue

                # Cross-sectional liquidity proxy using single-day data
                fund_info = fund.get(ticker, {})
                ref_info = ref.get(ticker, {})
                mc_est = fund_info.get("market_cap")
                if mc_est is None and fund_info.get("shares_outstanding"):
                    mc_est = close * fund_info["shares_outstanding"]

                s = {
                    "ticker": ticker,
                    "dvol": dvol,
                    "shares": fund_info.get("shares_outstanding"),
                    "market_cap": mc_est,
                    "sector": ref_info.get("sector", ""),
                    "exchange": ref_info.get("exchange", ""),
                    "listing_date": ref_info.get("listing_date", ""),
                    "delisting_date": ref_info.get("delisting_date", ""),
                }
                stats.append(s)
                cache[ticker] = {
                    "dvol": s["dvol"],
                    "shares": s["shares"],
                    "market_cap": s["market_cap"],
                }

            del rows
        except Exception as e:
            print(f"  flatfile error: {e}", file=sys.stderr)

    # Priority 3: 1min data (legacy, ~600 tickers)
    if not source:
        source = "1min"
        year_dir = OHLCV_BASE / "1min" / str(year)
        if year_dir.exists():
            csv_files = sorted(year_dir.iterdir())
            total = len(csv_files)
            processed = 0
            for fpath in csv_files:
                if fpath.suffix != ".csv":
                    continue
                ticker = fpath.stem.split("_")[0]
                if tickers is not None and ticker not in tickers:
                    continue

                if ticker in cache and cache[ticker].get("dvol", 0) > 0:
                    ref_info = ref.get(ticker, {})
                    s = cache[ticker].copy()
                    s["ticker"] = ticker
                    s["sector"] = ref_info.get("sector", "")
                    s["exchange"] = ref_info.get("exchange", "")
                    s["listing_date"] = ref_info.get("listing_date", "")
                    s["delisting_date"] = ref_info.get("delisting_date", "")
                    stats.append(s)
                    loaded_from_cache += 1
                    continue

                try:
                    with open(fpath) as f:
                        header = f.readline().strip().split(",")
                        ci = header.index("close")
                        vi = header.index("volume")
                    dvol = 0.0
                    close_sum = 0.0
                    count = 0
                    with open(fpath) as f:
                        next(f)
                        for line in f:
                            parts = line.strip().split(",")
                            try:
                                close = float(parts[ci])
                                vol = float(parts[vi])
                            except (ValueError, IndexError):
                                continue
                            dvol += close * vol
                            close_sum += close
                            count += 1
                    if count == 0 or dvol == 0:
                        cache[ticker] = {"dvol": 0}
                        continue
                except Exception:
                    cache[ticker] = {"dvol": 0}
                    continue

                close_avg = close_sum / count
                fund_info = fund.get(ticker, {})
                ref_info = ref.get(ticker, {})
                mc_est = fund_info.get("market_cap")
                if mc_est is None and fund_info.get("shares_outstanding"):
                    mc_est = close_avg * fund_info["shares_outstanding"]

                s = {
                    "ticker": ticker,
                    "dvol": dvol,
                    "shares": fund_info.get("shares_outstanding"),
                    "market_cap": mc_est,
                    "sector": ref_info.get("sector", ""),
                    "exchange": ref_info.get("exchange", ""),
                    "listing_date": ref_info.get("listing_date", ""),
                    "delisting_date": ref_info.get("delisting_date", ""),
                }
                stats.append(s)
                cache[ticker] = {
                    "dvol": s["dvol"],
                    "shares": s["shares"],
                    "market_cap": s["market_cap"],
                }
                processed += 1
                if processed % 50 == 0:
                    elapsed = time.time() - t0
                    sys.stderr.write(f"\r    {processed}/{total} files ({100*processed/total:.0f}%) in {elapsed:.0f}s")

    with open(cache_file, "w") as f:
        json.dump(cache, f)

    if source == "flatfile":
        sys.stderr.write(f"    {len(stats)} tickers from flatfile, {loaded_from_cache} cached; {time.time()-t0:.1f}s\n")
    elif source:
        sys.stderr.write(f"    {len(stats)} tickers from {source}, {loaded_from_cache} cached; {time.time()-t0:.1f}s\n")
    else:
        sys.stderr.write(f"    0 tickers (no data source found); {time.time()-t0:.1f}s\n")

    return stats


def write_universe(out_dir: Path, name: str, items: list[dict]):
    out_path = out_dir / f"{name}.csv"
    fieldnames = ["ticker", "market_cap", "rank"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(items, 1):
            ticker = item["ticker"] if isinstance(item, dict) else item
            mc = item.get("market_cap") if isinstance(item, dict) else None
            writer.writerow({
                "ticker": ticker,
                "market_cap": round(mc, 0) if mc is not None else "",
                "rank": rank,
            })
    print(f"  {name}: {len(items)} tickers -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate yearly universe ticker lists from OHLCV + flatfile + fundamentals data"
    )
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--max-year", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ticker_types = load_ticker_types()
    ref = load_reference()
    fund = load_fundamentals()
    all_years = get_available_years()

    if not all_years:
        print("No OHLCV or flatfile data found", file=sys.stderr)
        sys.exit(1)

    years = [
        y
        for y in all_years
        if (args.min_year or all_years[0]) <= y <= (args.max_year or all_years[-1])
    ]

    print(f"Ticker types loaded: {len(ticker_types)}")
    print(f"Reference tickers: {len(ref)}")
    print(f"Fundamentals tickers: {len(fund)}")
    print(f"Flatfile years: {len(FLATFILE_YEARS)}")
    print(f"Years: {years[0]}-{years[-1]} ({len(years)} years)")

    for year in years:
        print(f"\n{year}:")
        stats = compute_year_stats(year, fund, ref, ticker_types)

        if not stats:
            print("  no tickers with volume, skipping")
            continue

        stats.sort(key=lambda s: s["dvol"], reverse=True)

        out_dir = OUTPUT_BASE / str(year)
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            print(f"  top500_liquidity top 5: {', '.join(s['ticker'] for s in stats[:5])}")
            print(f"  top1000_liquidity: {len(stats[:1000])} tickers")
            print(f"  top3000_liquidity: {len(stats[:3000])} tickers")
            mc_count = sum(1 for s in stats if s.get("market_cap"))
            print(f"  large/mid/small cap: {mc_count} with market cap")
            sp500_count = sum(
                1
                for s in stats
                if s.get("market_cap") is not None
                and s["market_cap"] >= 10e9
                and (
                    not s.get("delisting_date")
                    or not s["delisting_date"][:4].isdigit()
                    or int(s["delisting_date"][:4]) > year
                )
                and (
                    not s.get("exchange")
                    or s["exchange"] in ("XNYS", "XNAS", "NYSE", "NASDAQ")
                )
            )
            print(f"  sp500: {sp500_count} candidates")
            continue

        write_universe(out_dir, "top500_liquidity", stats[:500])
        write_universe(out_dir, "top1000_liquidity", stats[:1000])
        write_universe(out_dir, "top3000_liquidity", stats[:3000])

        n = 250
        large_cap = stats[: n]
        mid_cap = stats[n : 2 * n]
        small_cap = stats[2 * n : 3 * n]
        mini_cap = stats[3 * n : 4 * n]

        write_universe(out_dir, "large_cap_liquidity", large_cap)
        write_universe(out_dir, "mid_cap_liquidity", mid_cap)
        write_universe(out_dir, "small_cap_liquidity", small_cap)
        write_universe(out_dir, "mini_cap_liquidity", mini_cap)

        sp500 = [
            s
            for s in stats
            if s.get("market_cap") is not None
            and s["market_cap"] >= 10e9
            and (
                not s.get("delisting_date")
                or not s["delisting_date"][:4].isdigit()
                or int(s["delisting_date"][:4]) > year
            )
            and (
                not s.get("exchange")
                or s["exchange"] in ("XNYS", "XNAS", "NYSE", "NASDAQ")
            )
        ]
        write_universe(out_dir, "sp500", sp500)

    print(f"\nDone. Universes written to {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
