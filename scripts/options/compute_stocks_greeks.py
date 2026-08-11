"""
Compute implied volatility and Greeks for options features CSV files and
append Greeks columns in-place (via processing/ staging).

Reads the per-ticker options features CSV, computes IV + Greeks for 4 contracts
per row (ATM call, ATM put, 30D call, 30D put), appends 24 new columns to the
output, writes to processing/, then atomically replaces the original.

Usage:
    python scripts/options/compute_stocks_greeks.py --tickers CLF --year 2025
    python scripts/options/compute_stocks_greeks.py --ohlcv_tickers --year 2025 --resume
    python scripts/options/compute_stocks_greeks.py --tickers AAPL --year 2025 --exclude_tickers /tmp/skip.txt
"""

import argparse
import csv
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

SCRIPT_NAME = Path(__file__).resolve().stem

AGGREGATE_MAP = {
    "1sec": (1, "second", "1sec"),
    "1min": (1, "minute", "1min"),
    "5min": (5, "minute", "5min"),
    "15min": (15, "minute", "15min"),
    "1H": (1, "hour", "1H"),
    "4H": (4, "hour", "4H"),
    "1D": (1, "day", "1D"),
}

_CONTRACT_SPECS = [
    ("atm_call", "atm_call_close", "atm_strike", "atm_days_to_expiry", "C"),
    ("atm_put", "atm_put_close", "atm_strike", "atm_days_to_expiry", "P"),
    ("iv30d_call", "iv30d_call_close", "iv30d_strike", "iv30d_days_to_expiry", "C"),
    ("iv30d_put", "iv30d_put_close", "iv30d_strike", "iv30d_days_to_expiry", "P"),
]

_GREEK_NAMES = ["iv", "delta", "gamma", "theta", "vega", "rho"]

APPEND_HEADERS = [
    f"{prefix}_{greek}" for prefix in ["atm_call", "atm_put", "iv30d_call", "iv30d_put"]
    for greek in _GREEK_NAMES
]


# ---- Black-Scholes ----

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_implied_vol(call_put: str, price: float, S: float, K: float, T: float,
                   r: float) -> float | None:
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return None
    sigma = 0.3
    for _ in range(20):
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        v = S * _norm_pdf(d1) * math.sqrt(T)
        if v < 1e-10:
            break
        d2 = d1 - sigma * math.sqrt(T)
        mp = (S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
              if call_put == "C"
              else K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1))
        diff = mp - price
        if abs(diff) < 0.0001:
            break
        sigma -= diff / v
        if sigma <= 0.001 or sigma >= 5.0:
            return None
    return sigma if 0.001 < sigma < 5.0 else None


def bs_greeks(call_put: str, S: float, K: float, T: float, r: float,
              sigma: float) -> dict[str, float] | None:
    if T <= 0 or sigma <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = _norm_pdf(d1)

    if call_put == "C":
        delta = _norm_cdf(d1)
        theta_raw = (-S * pdf_d1 * sigma / (2.0 * math.sqrt(T))
                     - r * K * math.exp(-r * T) * _norm_cdf(d2))
        rho_val = K * T * math.exp(-r * T) * _norm_cdf(d2) / 100.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta_raw = (-S * pdf_d1 * sigma / (2.0 * math.sqrt(T))
                     + r * K * math.exp(-r * T) * _norm_cdf(-d2))
        rho_val = -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100.0

    return {
        "iv": round(sigma, 6),
        "delta": round(delta, 6),
        "gamma": round(pdf_d1 / (S * sigma * math.sqrt(T)), 6),
        "theta": round(theta_raw / 365.0, 6),
        "vega": round(S * pdf_d1 * math.sqrt(T) / 100.0, 6),
        "rho": round(rho_val, 6),
    }


# ---- Yield curve ----

def load_yields(yields_path: str) -> list[dict]:
    yields: list[dict] = []
    if not os.path.exists(yields_path):
        return yields
    with open(yields_path) as f:
        for row in csv.DictReader(f):
            try:
                d = datetime.date.fromisoformat(row["date"])
            except (ValueError, KeyError):
                continue
            entry = {"date": d}
            for col, days in [("1m", 30), ("3m", 90), ("6m", 180), ("1y", 365),
                              ("2y", 730), ("3y", 1095), ("5y", 1825),
                              ("7y", 2555), ("10y", 3650), ("20y", 7300),
                              ("30y", 10950)]:
                try:
                    entry[days] = float(row.get(col, "") or 0)
                except (ValueError, KeyError):
                    entry[days] = 0.0
            yields.append(entry)
    yields.sort(key=lambda y: y["date"])
    return yields


def get_risk_free_rate(yields: list[dict], row_date: datetime.date, dte: int) -> float:
    if not yields:
        return 0.04
    best = None
    for entry in yields:
        if entry["date"] <= row_date:
            best = entry
        else:
            break
    if best is None:
        best = yields[0]
    maturities = sorted(k for k in best if isinstance(k, int))
    if not maturities:
        return 0.04
    if dte <= maturities[0]:
        return best[maturities[0]] / 100.0
    if dte >= maturities[-1]:
        return best[maturities[-1]] / 100.0
    for i in range(len(maturities) - 1):
        if maturities[i] <= dte <= maturities[i + 1]:
            frac = (dte - maturities[i]) / (maturities[i + 1] - maturities[i])
            y1, y2 = best[maturities[i]], best[maturities[i + 1]]
            if y1 <= 0 or y2 <= 0:
                return max(y1, y2) / 100.0
            return (y1 + frac * (y2 - y1)) / 100.0
    return 0.04


# ---- Underlying OHLCV lookup ----

import gzip as _gzip


def load_underlying_cache(underlying_dir: Path, ticker: str, year: str) -> dict[str, float]:
    cache: dict[str, float] = {}
    local_tz = datetime.timezone(datetime.timedelta(hours=8))
    for ext in (".csv", ".csv.gz"):
        path = underlying_dir / f"{ticker}_{year}_1min{ext}"
        if path.exists():
            opener = _gzip.open if ext == ".csv.gz" else open
            with opener(path, "rt", errors="replace") as f:
                for row in csv.DictReader(f):
                    ts = row.get("timestamp", "")
                    close_val = row.get("close", "")
                    if ts and close_val:
                        try:
                            dt_local = datetime.datetime.fromisoformat(ts)
                            dt_utc = (dt_local.replace(tzinfo=local_tz)
                                      .astimezone(datetime.timezone.utc))
                            ts_utc = dt_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")
                            cache[ts_utc] = float(close_val)
                        except (ValueError, KeyError):
                            pass
            break
    return cache


# ---- Args ----

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute IV and Greeks for options features CSV (in-place update)")
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--tickers_file", type=str, default=None)
    parser.add_argument("--ohlcv_tickers", action="store_true", default=False)
    parser.add_argument("--exclude_tickers", type=str, default=None)
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument("--aggregate", choices=list(AGGREGATE_MAP.keys()), default="1min")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip tickers whose atm_call_iv column is already populated")
    parser.add_argument("--yields_file", type=str,
                        default="data/treasury-yields/treasury_yields.csv")
    parser.add_argument("--inplace", type=lambda s: s.lower() == "true", default=False,
                        help="Overwrite original file (True) or write to greeks/ subfolder (False). Default: False.")
    parser.add_argument("--no_rename", action="store_true", default=False,
                        help="Used internally by parallel runner")
    return parser.parse_args()


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


def load_tickers(args) -> list[str]:
    tickers = []
    if args.tickers:
        tickers.extend(clean_ticker(t) for t in args.tickers.split(",") if t.strip())
    if args.tickers_file:
        with open(args.tickers_file) as f:
            for row in csv.DictReader(f):
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
            tickers.append(clean_ticker(name.split("_")[0]))
    if not tickers:
        raise SystemExit("Error: specify --tickers, --tickers_file, or --ohlcv_tickers")
    return list(dict.fromkeys(tickers))


def load_exclude(filepath: str) -> set[str]:
    if not filepath or not os.path.exists(filepath):
        return set()
    exclude = set()
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("ticker", "").strip()
            if t:
                exclude.add(clean_ticker(t))
        if not exclude:
            f.seek(0)
            for line in f:
                t = line.strip()
                if t and not t.startswith("ticker"):
                    exclude.add(clean_ticker(t))
    return exclude


def path_for(ticker: str, year: str, agg: str, output_dir: str | None = None,
             subdir: str | None = None) -> Path:
    folder = AGGREGATE_MAP[agg][2]
    base = Path(output_dir) if output_dir else Path("data")
    p = base / "options" / "stocks" / folder / year
    if subdir:
        p = p / subdir
    return p / f"{ticker}_{year}_{folder}_options.csv"


def is_complete(ticker: str, year: str, agg: str, output_dir: str | None = None,
                inplace: bool = False) -> bool:
    """Check if atm_call_iv column exists and has data."""
    check_subdir = None if inplace else "greeks"
    p = path_for(ticker, year, agg, output_dir, subdir=check_subdir)
    if not p.exists() or p.stat().st_size == 0:
        return False
    with open(p) as f:
        reader = csv.DictReader(f)
        if "atm_call_iv" not in (reader.fieldnames or []):
            return False
        for row in reader:
            iv = row.get("atm_call_iv", "")
            return bool(iv and iv.strip())
    return False


def compute_greek_for_contract(call_put: str, price_val: float, strike: float,
                               underlying: float, dte: int, yields: list[dict],
                               row_date: datetime.date) -> dict[str, float] | None:
    if price_val <= 0 or underlying <= 0 or strike <= 0 or dte <= 0:
        return None
    is_call = call_put == "C"
    intrinsic = (max(underlying - strike, 0.0) if is_call
                 else max(strike - underlying, 0.0))
    if price_val <= intrinsic:
        return None
    r = get_risk_free_rate(yields, row_date, dte)
    iv = bs_implied_vol(call_put, price_val, underlying, strike, dte / 365.0, r)
    if iv is None:
        return None
    return bs_greeks(call_put, underlying, strike, dte / 365.0, r, iv)


def process_ticker(ticker: str, year: str, agg: str, output_dir: str | None,
                   yields: list[dict], logger: logging.Logger) -> tuple[int, int]:
    in_path = path_for(ticker, year, agg, output_dir)
    if not in_path.exists():
        logger.warning("  [%s] file not found: %s", ticker, in_path)
        return 0, 0

    proc_path = path_for(ticker, year, agg, output_dir, subdir="processing")
    proc_path.parent.mkdir(parents=True, exist_ok=True)

    # Load OHLCV for backfilling missing underlying_price
    underlying_dir = (Path(output_dir) if output_dir else Path("data")) / "SPY" / "1min" / year
    underlying_cache = load_underlying_cache(underlying_dir, ticker, year)

    parsed = 0
    populated = 0
    backfilled = 0

    with open(in_path) as fin, open(proc_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        in_fields = list(reader.fieldnames or [])
        has_greeks = "atm_call_iv" in in_fields
        out_fields = list(in_fields)
        if not has_greeks:
            out_fields.extend(APPEND_HEADERS)

        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            parsed += 1
            ts = row.get("timestamp", "")
            try:
                row_date = datetime.datetime.fromisoformat(
                    ts.replace("+00:00", "")).date()
            except (ValueError, TypeError):
                writer.writerow({k: row.get(k, "") for k in out_fields})
                continue

            try:
                underlying = float(row.get("underlying_price", "") or 0)
            except (ValueError, TypeError):
                underlying = 0

            # Backfill missing underlying_price from OHLCV cache
            if underlying <= 0 and underlying_cache:
                und_lookup = underlying_cache.get(ts)
                if und_lookup and und_lookup > 0:
                    underlying = und_lookup
                    row["underlying_price"] = str(und_lookup)
                    backfilled += 1

            out_row = {k: row.get(k, "") for k in in_fields}
            row_had_greeks = False

            for prefix, price_col, strike_col, dte_col, cp in _CONTRACT_SPECS:
                try:
                    price_val = float(row.get(price_col, "") or 0)
                    strike = float(row.get(strike_col, "") or 0)
                    dte = int(float(row.get(dte_col, "") or 0))
                except (ValueError, TypeError):
                    for gk in _GREEK_NAMES:
                        out_row[f"{prefix}_{gk}"] = ""
                    continue

                g = compute_greek_for_contract(cp, price_val, strike, underlying,
                                               dte, yields, row_date)
                if g:
                    for gk in _GREEK_NAMES:
                        out_row[f"{prefix}_{gk}"] = str(g[gk])
                    row_had_greeks = True
                else:
                    for gk in _GREEK_NAMES:
                        out_row[f"{prefix}_{gk}"] = ""

            writer.writerow({k: out_row.get(k, "") for k in out_fields})
            if row_had_greeks:
                populated += 1

    logger.info("  [%s] %d rows, %d with greeks, %d backfilled", ticker, parsed, populated, backfilled)
    return parsed, populated


def main():
    args = parse_args()
    overall_start = time.time()
    tickers = load_tickers(args)
    exclude = load_exclude(args.exclude_tickers)
    if exclude:
        tickers = [t for t in tickers if t not in exclude]
        print(f"  Excluded {len(exclude)} tickers, {len(tickers)} remaining",
              flush=True)

    years = (
        [str(y) for y in range(int(args.year.split("-")[0]),
                               int(args.year.split("-")[1]) + 1)]
        if "-" in args.year else [args.year]
    )
    out_dir = args.output
    agg = args.aggregate

    log_base = Path(out_dir) if out_dir else Path("data")
    log_dir = log_base / "options" / "stocks" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / f"{SCRIPT_NAME}_{log_ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(SCRIPT_NAME)

    yields = load_yields(args.yields_file)
    if yields:
        logger.info("Loaded %d yield curve dates", len(yields))
    else:
        logger.warning("No yield data, using 4.0%% flat rate")

    logger.info("Stocks Greeks (in-place): %d tickers, years=%s, agg=%s",
                len(tickers), years, agg)

    all_ok = 0
    all_skip = 0

    for year in years:
        for i, ticker in enumerate(tickers, 1):
            if args.resume and is_complete(ticker, year, agg, out_dir, args.inplace):
                logger.info("[%d/%d] %s (%s) -> already complete, skipping",
                            i, len(tickers), ticker, year)
                all_skip += 1
                continue

            logger.info("[%d/%d] %s (%s) ...", i, len(tickers), ticker, year)
            t0 = time.time()
            try:
                parsed, populated = process_ticker(ticker, year, agg, out_dir, yields, logger)
            except Exception as e:
                elapsed = time.time() - t0
                logger.error("[%d/%d] %s (%s) -> FAILED after %.1fs: %s",
                             i, len(tickers), ticker, year, elapsed, e)
                continue
            elapsed = time.time() - t0

            if populated == 0:
                logger.info("[%d/%d] %s (%s) -> no greeks (%.1fs)",
                            i, len(tickers), ticker, year, elapsed)
                proc_path = path_for(ticker, year, agg, out_dir, subdir="processing")
                if proc_path.exists() and proc_path.stat().st_size > 0:
                    nd = path_for(ticker, year, agg, out_dir, subdir="no_data").parent
                    nd.mkdir(parents=True, exist_ok=True)
                    try:
                        proc_path.rename(nd / proc_path.name)
                    except OSError:
                        pass
            else:
                all_ok += 1
                logger.info("[%d/%d] %s (%s) -> %d/%d rows with greeks (%.1fs)",
                            i, len(tickers), ticker, year, populated, parsed, elapsed)
                logger.info('PARALLEL_RESULT:{"ticker":"%s","year":"%s","status":"ok","rows":%d}',
                            ticker, year, populated)
                sys.stderr.write(
                    'PARALLEL_RESULT:{"ticker":"%s","year":"%s","status":"ok","rows":%d}\n'
                    % (ticker, year, populated))
                sys.stderr.flush()

    if not args.no_rename:
        for year in years:
            for ticker in tickers:
                proc = path_for(ticker, year, agg, out_dir, subdir="processing")
                if args.inplace:
                    final = path_for(ticker, year, agg, out_dir)
                else:
                    final = path_for(ticker, year, agg, out_dir, subdir="greeks")
                if proc.exists() and proc.stat().st_size > 0:
                    final.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with open(proc) as pf:
                            if sum(1 for _ in pf) > 1:
                                proc.replace(final)
                    except OSError:
                        pass

    total_time = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("SUMMARY  Duration: %.1fs  OK: %d  Skipped: %d",
                total_time, all_ok, all_skip)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
