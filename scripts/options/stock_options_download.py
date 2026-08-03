"""
Download daily options-chain features for a list of tickers from the Massive
REST API and save each ticker as a CSV or Parquet file with one row per
trading day.

Uses list_snapshot_options_chain (one API call per day per ticker) to get
the full options chain snapshot, then computes aggregate features:
ATM IV, term structure, skew, GEX/DEX, put/call volume/OI, spread, etc.

Output layout (default):
    data/options/stocks/<year>/<ticker>_<year>_options.csv

With --output:
    <output>/options/stocks/<year>/<ticker>_<year>_options.csv

Usage:
    python scripts/options/stock_options_download.py --tickers AAPL,NVDA --year 2025
    python scripts/options/stock_options_download.py --tickers_file data/universes/2025/combined_unique.csv --year 2025
    python scripts/options/stock_options_download.py --tickers AAPL --year 2022-2025 --resume
    python scripts/options/stock_options_download.py --tickers AAPL --year 2025 --parquet
    python scripts/options/stock_options_download.py --tickers AAPL --year 2025 --start_date 2025-06-01
    python scripts/options/stock_options_download.py --tickers AAPL --year 2025 --output data/combined

One of --tickers or --tickers_file must be specified.
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

import pyarrow as pa
import pyarrow.parquet as pq

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("APIKEY")
if not api_key:
    raise ValueError("APIKEY not found in .env")
os.environ["MASSIVE_API_KEY"] = api_key

from massive import RESTClient

SCRIPT_NAME = Path(__file__).resolve().stem

CSV_HEADERS = [
    "ticker",
    "date",
    "underlying_price",
    "atm_strike",
    "atm_iv",
    "atm_iv_30d",
    "expected_move",
    "gex",
    "dex",
    "net_gamma",
    "put_volume",
    "call_volume",
    "put_call_ratio",
    "open_interest",
    "call_oi",
    "put_oi",
    "atm_oi",
    "atm_spread",
    "atm_bid",
    "atm_ask",
    "atm_bid_size",
    "atm_ask_size",
    "iv_7d",
    "iv_30d",
    "iv_60d",
    "iv_90d",
    "iv_25d_put",
    "iv_25d_call",
    "skew_25d",
    "contract_count",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download daily options-chain features for tickers from Massive API"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of ticker symbols (e.g. AAPL,TSLA,NVDA)",
    )
    parser.add_argument(
        "--tickers_file",
        type=str,
        help="Path to CSV with ticker list (header 'ticker'; may also have market_cap,rank columns)",
    )
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help="Year or year range (e.g. 2025 or 2022-2025)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: <year>-01-01)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tickers that already have a non-empty output file",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Write output as Parquet instead of CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output directory (default: data/). Options/stocks/<year> subdirs are appended.",
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
    if not tickers:
        raise SystemExit("Error: specify at least one of --tickers or --tickers_file")
    return tickers


def parse_years(year_arg: str) -> list[str]:
    parts = year_arg.split("-")
    if len(parts) == 1:
        y = parts[0].strip()
        if not y.isdigit():
            raise SystemExit(f"Error: invalid year '{year_arg}'")
        return [y]
    elif len(parts) == 2:
        start, end = parts[0].strip(), parts[1].strip()
        if not start.isdigit() or not end.isdigit():
            raise SystemExit(f"Error: invalid year range '{year_arg}'")
        return [str(y) for y in range(int(start), int(end) + 1)]
    else:
        raise SystemExit(f"Error: invalid year format '{year_arg}' (use YYYY or YYYY-YYYY)")


def output_base(output_dir: str | None = None) -> Path:
    base = Path(output_dir) if output_dir else Path("data")
    return base / "options" / "stocks"


def output_path(ticker: str, year: str, parquet: bool = False, output_dir: str | None = None, subdir: str | None = None) -> Path:
    ext = "parquet" if parquet else "csv"
    base = output_base(output_dir) / year
    if subdir:
        base = base / subdir
    return base / f"{ticker}_{year}_options.{ext}"


def output_rows(ticker: str, year: str, parquet: bool, output_dir: str | None = None) -> int:
    path = output_path(ticker, year, parquet, output_dir)
    if not path.exists():
        return 0
    if parquet:
        return pq.read_table(path).num_rows
    with open(path) as f:
        return sum(1 for _ in f) - 1


def is_ticker_complete(ticker: str, year: str, parquet: bool = False, output_dir: str | None = None) -> bool:
    p = output_path(ticker, year, parquet, output_dir)
    if not p.exists() or p.stat().st_size == 0:
        return False
    if parquet:
        try:
            return pq.ParquetFile(p).metadata.num_rows > 0
        except Exception:
            return False
    with open(p) as f:
        return sum(1 for _ in f) > 1


def trading_days(year: str) -> list[datetime.date]:
    start = datetime.date(int(year), 1, 1)
    end = datetime.date(int(year), 12, 31)
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def compute_features(ticker: str, day: str, chain: list) -> dict:
    features: dict = {h: None for h in CSV_HEADERS}
    features["ticker"] = ticker
    features["date"] = day

    if not chain:
        features["error"] = "empty_chain"
        return features

    contracts = []
    for c in chain:
        det = c.details
        if det is None or det.strike_price is None or det.expiration_date is None or det.contract_type is None:
            continue
        contracts.append(c)

    if not contracts:
        features["error"] = "no_valid_contracts"
        return features

    features["contract_count"] = len(contracts)

    underlying_price = None
    for c in contracts:
        if c.underlying_asset and c.underlying_asset.price is not None:
            underlying_price = safe_float(c.underlying_asset.price)
            break
    features["underlying_price"] = underlying_price

    calls = [c for c in contracts if c.details.contract_type.lower() == "call"]
    puts = [c for c in contracts if c.details.contract_type.lower() == "put"]

    # --- ATM strike ---
    if underlying_price and underlying_price > 0:
        atm_strike = min(contracts, key=lambda c: abs(c.details.strike_price - underlying_price)).details.strike_price
    else:
        atm_strike = None
    features["atm_strike"] = atm_strike

    # --- ATM options ---
    atm_call = None
    atm_put = None
    if atm_strike is not None:
        atm_candidates_call = [c for c in calls if c.details.strike_price == atm_strike]
        atm_candidates_put = [c for c in puts if c.details.strike_price == atm_strike]
        atm_call = atm_candidates_call[0] if atm_candidates_call else None
        atm_put = atm_candidates_put[0] if atm_candidates_put else None

    # --- ATM IV ---
    atm_iv_vals = []
    for c in (atm_call, atm_put):
        if c and c.implied_volatility is not None:
            iv = safe_float(c.implied_volatility)
            if iv:
                atm_iv_vals.append(iv)
    features["atm_iv"] = sum(atm_iv_vals) / len(atm_iv_vals) if atm_iv_vals else None

    # --- ATM spread / bid / ask ---
    atm_quote = None
    if atm_call and atm_call.last_quote:
        atm_quote = atm_call.last_quote
    elif atm_put and atm_put.last_quote:
        atm_quote = atm_put.last_quote
    if atm_quote:
        features["atm_bid"] = safe_float(atm_quote.bid)
        features["atm_ask"] = safe_float(atm_quote.ask)
        features["atm_bid_size"] = safe_float(atm_quote.bid_size)
        features["atm_ask_size"] = safe_float(atm_quote.ask_size)
        if features["atm_bid"] is not None and features["atm_ask"] is not None:
            features["atm_spread"] = round(features["atm_ask"] - features["atm_bid"], 4)

    # --- ATM OI ---
    atm_oi_vals = []
    for c in (atm_call, atm_put):
        if c and c.open_interest is not None:
            oi = safe_float(c.open_interest)
            if oi:
                atm_oi_vals.append(oi)
    features["atm_oi"] = sum(atm_oi_vals) / len(atm_oi_vals) if atm_oi_vals else None

    # --- Expected Move (ATM straddle price) ---
    if atm_call and atm_put:
        call_price = safe_float(atm_call.last_trade.price) if atm_call.last_trade else None
        put_price = safe_float(atm_put.last_trade.price) if atm_put.last_trade else None
        if call_price is None and atm_call.day:
            call_price = safe_float(atm_call.day.close)
        if put_price is None and atm_put.day:
            put_price = safe_float(atm_put.day.close)
        if call_price is not None and put_price is not None:
            features["expected_move"] = round(call_price + put_price, 4)

    # --- 30-day ATM IV ---
    features["atm_iv_30d"] = _atm_iv_at_target_expiry(contracts, underlying_price, atm_strike, 30)

    # --- Term structure ---
    for d in [7, 30, 60, 90]:
        features[f"iv_{d}d"] = _atm_iv_at_target_expiry(contracts, underlying_price, atm_strike, d)

    # --- 25-delta skew ---
    if underlying_price:
        iv_25d_put_val = _iv_at_target_delta(puts, -0.25, underlying_price)
        iv_25d_call_val = _iv_at_target_delta(calls, 0.25, underlying_price)
        features["iv_25d_put"] = iv_25d_put_val
        features["iv_25d_call"] = iv_25d_call_val
        if iv_25d_put_val is not None and iv_25d_call_val is not None:
            features["skew_25d"] = round(iv_25d_put_val - iv_25d_call_val, 4)

    # --- GEX / DEX / Net Gamma ---
    if underlying_price and underlying_price > 0:
        gex_sum = 0.0
        dex_sum = 0.0
        for c in contracts:
            g = c.greeks
            oi = safe_float(c.open_interest) or 0
            if g and oi > 0:
                gamma = safe_float(g.gamma) or 0
                delta = safe_float(g.delta) or 0
                gex_sum += gamma * oi * underlying_price * 100  # notional
                dex_sum += delta * oi * underlying_price * 100
        features["gex"] = round(gex_sum, 2) if gex_sum != 0 else None
        features["dex"] = round(dex_sum, 2) if dex_sum != 0 else None
        features["net_gamma"] = features["gex"]

    # --- Put/Call volume ---
    call_vol = 0.0
    put_vol = 0.0
    for c in calls:
        if c.day and c.day.volume is not None:
            call_vol += safe_float(c.day.volume) or 0
    for c in puts:
        if c.day and c.day.volume is not None:
            put_vol += safe_float(c.day.volume) or 0
    features["call_volume"] = round(call_vol, 2)
    features["put_volume"] = round(put_vol, 2)
    if call_vol > 0:
        features["put_call_ratio"] = round(put_vol / call_vol, 4)

    # --- Open Interest ---
    call_oi = 0.0
    put_oi = 0.0
    total_oi = 0.0
    for c in calls:
        oi = safe_float(c.open_interest) or 0
        call_oi += oi
        total_oi += oi
    for c in puts:
        oi = safe_float(c.open_interest) or 0
        put_oi += oi
        total_oi += oi
    features["open_interest"] = round(total_oi, 2) if total_oi > 0 else None
    features["call_oi"] = round(call_oi, 2)
    features["put_oi"] = round(put_oi, 2)

    return features


def _atm_iv_at_target_expiry(contracts: list, underlying_price: float | None, atm_strike: float | None, target_days: int) -> float | None:
    today = datetime.date.today()
    target_date = today + datetime.timedelta(days=target_days)

    candidates = []
    for c in contracts:
        det = c.details
        if det is None or det.expiration_date is None or c.implied_volatility is None:
            continue
        try:
            exp = datetime.date.fromisoformat(det.expiration_date)
        except (ValueError, TypeError):
            continue
        days_diff = (exp - today).days
        if days_diff < 1:
            continue
        if atm_strike is not None and det.strike_price != atm_strike:
            continue
        candidates.append((abs(days_diff - target_days), c))

    if not candidates:
        if atm_strike is not None:
            candidates = []
            for c in contracts:
                det = c.details
                if det is None or det.expiration_date is None or c.implied_volatility is None:
                    continue
                try:
                    exp = datetime.date.fromisoformat(det.expiration_date)
                except (ValueError, TypeError):
                    continue
                days_diff = (exp - today).days
                if days_diff < 1:
                    continue
                if abs(det.strike_price - atm_strike) / atm_strike > 0.02:
                    continue
                candidates.append((abs(days_diff - target_days), c))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return round(safe_float(candidates[0][1].implied_volatility), 4)
    return None


def _iv_at_target_delta(options: list, target_delta: float, underlying_price: float) -> float | None:
    candidates = []
    for c in options:
        g = c.greeks
        if g is None or g.delta is None or c.implied_volatility is None:
            continue
        delta = safe_float(g.delta)
        iv = safe_float(c.implied_volatility)
        if delta is None or iv is None:
            continue
        candidates.append((abs(delta - target_delta), iv, delta))
    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])

    if len(candidates) >= 2:
        d1, iv1, delta1 = candidates[0][0], candidates[0][1], candidates[0][2]
        d2, iv2, delta2 = candidates[1][0], candidates[1][1], candidates[1][2]
        if abs(delta2 - delta1) > 1e-9:
            weight = (target_delta - delta1) / (delta2 - delta1)
            return round(iv1 + weight * (iv2 - iv1), 4)
    return round(candidates[0][1], 4)


def download_ticker(client, ticker: str, year: str, parquet: bool = False, start_date: str | None = None, output_dir: str | None = None) -> int:
    days = trading_days(year)
    if start_date:
        sd = datetime.date.fromisoformat(start_date)
        days = [d for d in days if d >= sd]

    if not days:
        return 0

    proc_path = output_path(ticker, year, parquet, output_dir, subdir="processing")
    proc_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    schema = pa.schema([(h, pa.float64() if h != "ticker" and h != "date" and h != "error" else pa.string()) for h in CSV_HEADERS])

    if parquet:
        writer = None
        batch = []
    else:
        f = open(proc_path, "w", newline="")
        writer_csv = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer_csv.writeheader()

    try:
        for day in days:
            day_str = day.strftime("%Y-%m-%d")

            chain = []
            try:
                for s in client.list_snapshot_options_chain(ticker, params={"as_of": day_str}):
                    chain.append(s)
            except Exception as e:
                logging.getLogger(SCRIPT_NAME).warning(
                    "  %s %s: snapshot API error: %s", ticker, day_str, e
                )
                continue

            features = compute_features(ticker, day_str, chain)

            if parquet:
                batch.append(features)
                if len(batch) >= 5000:
                    table = pa.Table.from_pylist(batch, schema=schema)
                    if writer is None:
                        writer = pq.ParquetWriter(proc_path, schema)
                    writer.write_table(table)
                    count += len(batch)
                    batch = []
            else:
                writer_csv.writerow(features)
                count += 1

            time.sleep(0.25)
    finally:
        if parquet:
            if batch:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(proc_path, schema)
                writer.write_table(table)
                count += len(batch)
            if writer is not None:
                writer.close()
        else:
            f.flush()
            f.close()

    if count == 0:
        return 0

    out = output_path(ticker, year, parquet, output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc_path.rename(out)
    return count


def fmt_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def main():
    args = parse_args()
    overall_start = time.time()
    tickers = load_tickers(args)
    years = parse_years(args.year)

    out_dir = args.output

    log_dir = output_base(out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / f"{SCRIPT_NAME}_{log_ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(SCRIPT_NAME)

    logger.info("Starting options download for %d tickers, years=%s", len(tickers), years)
    logger.info("Output format: %s", "parquet" if args.parquet else "csv")
    logger.info("Resume mode: %s", args.resume)
    if args.start_date:
        logger.info("Start date: %s", args.start_date)
    logger.info("Output base: %s", output_base(out_dir).resolve())

    client = RESTClient(trace=True)

    all_missing: list[str] = []
    all_results: list[dict] = []
    all_downloaded = 0
    all_skipped = 0

    for year in years:
        downloaded = 0
        skipped = 0

        for i, ticker in enumerate(tickers, 1):
            if args.resume and is_ticker_complete(ticker, year, args.parquet, out_dir):
                logger.info("[%d/%d] %s (%s) -> already complete, skipping", i, len(tickers), ticker, year)
                skipped += 1
                all_results.append({
                    "ticker": ticker, "year": year, "status": "skipped",
                    "rows": output_rows(ticker, year, args.parquet, out_dir),
                })
                continue

            logger.info("[%d/%d] %s (%s) -> downloading ...", i, len(tickers), ticker, year)

            t0 = time.time()
            try:
                count = download_ticker(client, ticker, year, args.parquet, args.start_date, out_dir)
            except Exception as e:
                elapsed = time.time() - t0
                logger.error("[%d/%d] %s (%s) -> FAILED after %.1fs: %s", i, len(tickers), ticker, year, elapsed, e)
                all_missing.append(ticker)
                all_results.append({"ticker": ticker, "year": year, "status": "failed", "error": str(e), "elapsed_s": round(elapsed, 1)})
                proc_path = output_path(ticker, year, args.parquet, out_dir, subdir="processing")
                if proc_path.exists():
                    err_dir = output_path(ticker, year, args.parquet, out_dir, subdir="errors").parent
                    err_dir.mkdir(parents=True, exist_ok=True)
                    proc_path.rename(err_dir / proc_path.name)
                continue

            elapsed = time.time() - t0

            if count == 0:
                logger.warning("[%d/%d] %s (%s) -> no data returned (%.1fs)", i, len(tickers), ticker, year, elapsed)
                all_missing.append(ticker)
                all_results.append({"ticker": ticker, "year": year, "status": "no_data", "elapsed_s": round(elapsed, 1)})
                proc_path = output_path(ticker, year, args.parquet, out_dir, subdir="processing")
                if proc_path.exists():
                    err_dir = output_path(ticker, year, args.parquet, out_dir, subdir="errors").parent
                    err_dir.mkdir(parents=True, exist_ok=True)
                    proc_path.rename(err_dir / proc_path.name)
            else:
                path = output_path(ticker, year, args.parquet, out_dir)
                size = path.stat().st_size
                logger.info(
                    "[%d/%d] %s (%s) -> %d days (%s, %.1fs) -> %s",
                    i, len(tickers), ticker, year, count, fmt_bytes(size), elapsed, path,
                )
                downloaded += 1
                all_results.append({
                    "ticker": ticker, "year": year, "status": "ok",
                    "rows": count, "size_bytes": size, "path": str(path),
                    "elapsed_s": round(elapsed, 1),
                })
                logger.info("PARALLEL_RESULT:{\"ticker\":\"%s\",\"year\":\"%s\",\"status\":\"ok\",\"rows\":%d}", ticker, year, count)
                sys.stderr.write("PARALLEL_RESULT:{\"ticker\":\"%s\",\"year\":\"%s\",\"status\":\"ok\",\"rows\":%d}\n" % (ticker, year, count))
                sys.stderr.flush()

            time.sleep(0.25)

        all_downloaded += downloaded
        all_skipped += skipped

    total_time = time.time() - overall_start

    summary = {
        "script": SCRIPT_NAME,
        "timestamp": log_ts,
        "years": years,
        "format": "parquet" if args.parquet else "csv",
        "total_tickers": len(tickers),
        "downloaded": all_downloaded,
        "skipped": all_skipped,
        "missing": sorted(set(all_missing)),
        "missing_count": len(set(all_missing)),
        "duration_s": round(total_time, 1),
        "results": all_results,
    }

    report_path = log_dir / f"{SCRIPT_NAME}_{log_ts}_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("SUMMARY REPORT")
    logger.info("  Duration:       %.1fs", total_time)
    logger.info("  Years:          %s", years)
    logger.info("  Downloaded:     %d", all_downloaded)
    logger.info("  Skipped:        %d", all_skipped)
    logger.info("  Missing/failed: %d", len(set(all_missing)))
    missing_unique = sorted(set(all_missing))
    if missing_unique:
        logger.info("  Missing tickers: %s", ", ".join(missing_unique))
    logger.info("  Report:         %s", report_path)


if __name__ == "__main__":
    main()
