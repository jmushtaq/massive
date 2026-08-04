"""
Compute implied volatility and Greeks for options features CSV files.

Reads the per-ticker options features CSV (produced by stock_options_from_flatfiles)
and computes IV + Greeks for 4 contracts per row:
  - ATM Call
  - ATM Put
  - 30-Day Call
  - 30-Day Put

Uses US Treasury yield curve for risk-free rate interpolation.
Writes separate output files — does not modify input.

Output layout (default):
    data/options/stocks/<agg>/<year>/<ticker>_<year>_<agg>_options_greeks.csv

Usage:
    python scripts/options/compute_stocks_greeks.py --tickers CLF --year 2025
    python scripts/options/compute_stocks_greeks.py --ohlcv_tickers --year 2025 --resume
"""

import argparse
import csv
import datetime
import gzip
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

STOCKS_GREEKS_HEADERS = ["ticker", "timestamp"] + [
    f"{prefix}_{greek}" for prefix in ["atm_call", "atm_put", "iv30d_call", "iv30d_put"]
    for greek in _GREEK_NAMES
]


# ---- Black-Scholes ----

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(call_put: str, S: float, K: float, T: float, r: float,
             sigma: float) -> float | None:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if call_put == "C":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_implied_vol(call_put: str, price: float, S: float, K: float, T: float,
                   r: float) -> float | None:
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return None
    sigma = 0.3
    for _ in range(20):
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        vega_val = S * _norm_pdf(d1) * math.sqrt(T)
        if vega_val < 1e-10:
            break
        model_price = bs_price(call_put, S, K, T, r, sigma)
        if model_price is None:
            return None
        diff = model_price - price
        if abs(diff) < 0.0001:
            break
        sigma -= diff / vega_val
        if sigma <= 0.001 or sigma >= 5.0:
            return None
    return sigma if 0.001 < sigma < 5.0 else None


def bs_greeks(call_put: str, S: float, K: float, T: float, r: float,
              sigma: float) -> dict[str, float] | None:
    if T <= 0 or sigma <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if call_put == "C":
        delta = _norm_cdf(d1)
        theta_raw = (-S * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
                     - r * K * math.exp(-r * T) * _norm_cdf(d2))
        rho_val = K * T * math.exp(-r * T) * _norm_cdf(d2) / 100.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta_raw = (-S * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
                     + r * K * math.exp(-r * T) * _norm_cdf(-d2))
        rho_val = -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100.0

    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega_val = S * _norm_pdf(d1) * math.sqrt(T) / 100.0
    theta = theta_raw / 365.0

    return {
        "iv": round(sigma, 6),
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "theta": round(theta, 6),
        "vega": round(vega_val, 6),
        "rho": round(rho_val, 6),
    }


# ---- Yield curve ----

def load_yields(yields_path: str) -> list[dict]:
    yields: list[dict] = []
    if not os.path.exists(yields_path):
        return yields
    with open(yields_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
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


# ---- Args ----

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute IV and Greeks for options features CSV files")
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--tickers_file", type=str, default=None)
    parser.add_argument("--ohlcv_tickers", action="store_true", default=False)
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument("--aggregate", choices=list(AGGREGATE_MAP.keys()), default="1min")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--yields_file", type=str,
                        default="data/treasury-yields/treasury_yields.csv")
    parser.add_argument("--no_rename", action="store_true", default=False)
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
            tickers.append(clean_ticker(name.split("_")[0]))
    if not tickers:
        raise SystemExit("Error: specify --tickers, --tickers_file, or --ohlcv_tickers")
    return list(dict.fromkeys(tickers))


def output_path(ticker: str, year: str, agg: str, output_dir: str | None = None,
                subdir: str | None = None) -> Path:
    folder = AGGREGATE_MAP[agg][2]
    base = Path(output_dir) if output_dir else Path("data")
    p = base / "options" / "stocks" / folder / year
    if subdir:
        p = p / subdir
    return p / f"{ticker}_{year}_{folder}_options_greeks.csv"


def input_path(ticker: str, year: str, agg: str, output_dir: str | None = None) -> Path:
    folder = AGGREGATE_MAP[agg][2]
    base = Path(output_dir) if output_dir else Path("data")
    return base / "options" / "stocks" / folder / year / f"{ticker}_{year}_{folder}_options.csv"


def is_complete(ticker: str, year: str, agg: str, output_dir: str | None = None) -> bool:
    p = output_path(ticker, year, agg, output_dir)
    if not p.exists() or p.stat().st_size == 0:
        return False
    with open(p) as f:
        return sum(1 for _ in f) > 1


def compute_greeks_for_contract(call_put: str, price_val: float, strike: float,
                                underlying: float, dte: int, yields: list[dict],
                                row_date: datetime.date) -> dict[str, float] | None:
    if price_val <= 0 or underlying <= 0 or strike <= 0 or dte <= 0:
        return None
    is_call = call_put == "C"
    intrinsic = max(underlying - strike, 0.0) if is_call else max(strike - underlying, 0.0)
    if price_val <= intrinsic:
        return None
    r = get_risk_free_rate(yields, row_date, dte)
    iv = bs_implied_vol(call_put, price_val, underlying, strike, dte / 365.0, r)
    if iv is None:
        return None
    return bs_greeks(call_put, underlying, strike, dte / 365.0, r, iv)


def compute_greeks(ticker: str, year: str, agg: str, output_dir: str | None,
                   yields: list[dict], logger: logging.Logger) -> int:
    in_path = input_path(ticker, year, agg, output_dir)
    if not in_path.exists():
        logger.warning("  [%s] input file not found: %s", ticker, in_path)
        return 0

    out_path = output_path(ticker, year, agg, output_dir, subdir="processing")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    parsed = 0

    with open(in_path) as fin, open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=STOCKS_GREEKS_HEADERS)
        writer.writeheader()

        for row in reader:
            parsed += 1
            ts = row.get("timestamp", "")
            try:
                row_date = datetime.datetime.fromisoformat(
                    ts.replace("+00:00", "")).date()
            except (ValueError, TypeError):
                continue

            try:
                underlying = float(row.get("underlying_price", "") or 0)
            except (ValueError, TypeError):
                underlying = 0

            out_row = {"ticker": row.get("ticker", ticker), "timestamp": ts}

            for prefix, price_col, strike_col, dte_col, cp in _CONTRACT_SPECS:
                try:
                    price_val = float(row.get(price_col, "") or 0)
                    strike = float(row.get(strike_col, "") or 0)
                    dte = int(float(row.get(dte_col, "") or 0))
                except (ValueError, TypeError):
                    for gk in _GREEK_NAMES:
                        out_row[f"{prefix}_{gk}"] = ""
                    continue

                g = compute_greeks_for_contract(cp, price_val, strike, underlying,
                                                dte, yields, row_date)
                if g:
                    for gk in _GREEK_NAMES:
                        out_row[f"{prefix}_{gk}"] = str(g[gk])
                else:
                    for gk in _GREEK_NAMES:
                        out_row[f"{prefix}_{gk}"] = ""

            writer.writerow(out_row)
            written += 1

    logger.info("  [%s] %d rows parsed, %d written", ticker, parsed, written)
    return written


def main():
    args = parse_args()
    overall_start = time.time()

    tickers = load_tickers(args)
    years = (
        [str(y) for y in range(int(args.year.split("-")[0]), int(args.year.split("-")[1]) + 1)]
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
        logger.info("Loaded %d yield curve dates from %s", len(yields), args.yields_file)
    else:
        logger.warning("No yield data found, using 4.0%% flat rate")

    logger.info("Stocks Greeks compute: %d tickers, years=%s, aggregate=%s",
                len(tickers), years, agg)

    all_downloaded = 0
    all_skipped = 0

    for year in years:
        for i, ticker in enumerate(tickers, 1):
            if args.resume and is_complete(ticker, year, agg, out_dir):
                logger.info("[%d/%d] %s (%s) -> already complete, skipping",
                            i, len(tickers), ticker, year)
                all_skipped += 1
                continue

            logger.info("[%d/%d] %s (%s) ...", i, len(tickers), ticker, year)
            t0 = time.time()
            try:
                count = compute_greeks(ticker, year, agg, out_dir, yields, logger)
            except Exception as e:
                elapsed = time.time() - t0
                logger.error("[%d/%d] %s (%s) -> FAILED after %.1fs: %s",
                             i, len(tickers), ticker, year, elapsed, e)
                continue
            elapsed = time.time() - t0

            if count == 0:
                logger.info("[%d/%d] %s (%s) -> no rows (%.1fs)",
                            i, len(tickers), ticker, year, elapsed)
            else:
                all_downloaded += 1
                proc_path = output_path(ticker, year, agg, out_dir, subdir="processing")
                logger.info("[%d/%d] %s (%s) -> %d rows (%.1fs)",
                            i, len(tickers), ticker, year, count, elapsed)
                logger.info('PARALLEL_RESULT:{"ticker":"%s","year":"%s","status":"ok","rows":%d}',
                            ticker, year, count)
                sys.stderr.write('PARALLEL_RESULT:{"ticker":"%s","year":"%s","status":"ok","rows":%d}\n'
                                 % (ticker, year, count))
                sys.stderr.flush()

    if not args.no_rename:
        for year in years:
            for ticker in tickers:
                proc = output_path(ticker, year, agg, out_dir, subdir="processing")
                final = output_path(ticker, year, agg, out_dir)
                if proc.exists() and proc.stat().st_size > 0:
                    final.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with open(proc) as pf:
                            if sum(1 for _ in pf) > 1:
                                proc.rename(final)
                            else:
                                nd = output_path(ticker, year, agg, out_dir, subdir="no_data").parent
                                nd.mkdir(parents=True, exist_ok=True)
                                proc.rename(nd / proc.name)
                    except OSError:
                        pass

    total_time = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("SUMMARY  Duration: %.1fs  Downloaded: %d  Skipped: %d",
                total_time, all_downloaded, all_skipped)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
