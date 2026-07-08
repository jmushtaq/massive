"""
Download reference/ticker-detail data for a list of tickers from the Massive
(REST) API and save as a single CSV file.

Usage:
    python scripts/reference_download.py --tickers AAPL,NVDA,TSLA
    python scripts/reference_download.py --tickers_file data/spy_tickers/tickers_combined_unique.csv
    python scripts/reference_download.py --tickers AAPL,MSFT --parquet

Output:
    data/reference/reference.csv
    data/reference/reference.parquet  (with --parquet)

This script calls the /v3/reference/tickers/{ticker} endpoint for each ticker
and extracts the following fields: ticker, exchange, sic_description,
list_date, delisted_utc.

Missing or delisted tickers are logged at the end in a summary report.
"""

import argparse
import csv
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("APIKEY")
if not api_key:
    raise ValueError("APIKEY not found in .env")
os.environ["MASSIVE_API_KEY"] = api_key

from massive import RESTClient
from massive.rest.models import TickerDetails

SCRIPT_NAME = Path(__file__).resolve().stem

CSV_HEADERS = [
    "ticker",
    "exchange",
    "sector_industry",
    "listing_date",
    "delisting_date",
]

OUTPUT_BASE = Path("data") / "reference"
LOG_DIR = OUTPUT_BASE / "logs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download reference/ticker-detail data from Massive API"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of ticker symbols (e.g. AAPL,TSLA,NVDA)",
    )
    parser.add_argument(
        "--tickers_file",
        type=str,
        help="Path to CSV with ticker list (one ticker per row, header 'ticker')",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tickers already present in the output file",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Write output as Parquet instead of CSV",
    )
    return parser.parse_args()


def load_tickers(args) -> list[str]:
    tickers = []
    if args.tickers:
        tickers.extend(t.strip().upper() for t in args.tickers.split(",") if t.strip())
    if args.tickers_file:
        with open(args.tickers_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("ticker", "").strip()
                if t:
                    tickers.append(t)
    if not tickers:
        raise SystemExit("Error: specify at least one of --tickers or --tickers_file")
    return tickers


def output_path(parquet: bool = False) -> Path:
    ext = "parquet" if parquet else "csv"
    return OUTPUT_BASE / f"reference.{ext}"


def get_existing_tickers(parquet: bool) -> set:
    path = output_path(parquet)
    if not path.exists():
        return set()
    existing = set()
    if parquet:
        table = pq.read_table(path)
        for t in table.column("ticker").to_pylist():
            if t:
                existing.add(t)
    else:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("ticker", "").strip()
                if t:
                    existing.add(t)
    return existing


def extract_row(d: TickerDetails) -> dict:
    return {
        "ticker": d.ticker or "",
        "exchange": d.primary_exchange or "",
        "sector_industry": d.sic_description or "",
        "listing_date": d.list_date or "",
        "delisting_date": d.delisted_utc or "",
    }


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

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = LOG_DIR / f"{SCRIPT_NAME}_{log_ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(SCRIPT_NAME)

    logger.info("Fetching reference data for %d tickers", len(tickers))
    logger.info("Output format: %s", "parquet" if args.parquet else "csv")
    logger.info("Resume mode: %s", args.resume)

    existing = get_existing_tickers(args.parquet) if args.resume else set()
    logger.info("Already saved: %d tickers", len(existing))

    client = RESTClient(trace=True)

    rows: list[dict] = []
    missing: list[str] = []
    failed: list[str] = []

    for i, ticker in enumerate(tickers, 1):
        if ticker in existing:
            logger.info("[%d/%d] %s -> already saved, skipping", i, len(tickers), ticker)
            continue

        logger.info("[%d/%d] %s -> fetching ...", i, len(tickers), ticker)

        t0 = time.time()
        try:
            details = client.get_ticker_details(ticker)
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("[%d/%d] %s -> FAILED after %.1fs: %s", i, len(tickers), ticker, elapsed, e)
            failed.append(ticker)
            continue

        elapsed = time.time() - t0

        if details is None or details.ticker is None:
            logger.warning("[%d/%d] %s -> no data returned (%.1fs)", i, len(tickers), ticker, elapsed)
            missing.append(ticker)
        else:
            rows.append(extract_row(details))
            logger.info(
                "[%d/%d] %s -> %s (%.1fs)",
                i, len(tickers), ticker, details.name or "(no name)", elapsed,
            )

        time.sleep(0.25)

    # Append to output file
    out = output_path(args.parquet)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.parquet:
        new_table = pa.Table.from_pylist(rows)
        if out.exists():
            old_table = pq.read_table(out)
            combined = pa.concat_tables([old_table, new_table])
            pq.write_table(combined, out)
        else:
            pq.write_table(new_table, out)
    else:
        write_header = not out.exists()
        with open(out, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    total_time = time.time() - overall_start
    total_rows = len(rows)

    summary = {
        "script": SCRIPT_NAME,
        "timestamp": log_ts,
        "format": "parquet" if args.parquet else "csv",
        "total_tickers_requested": len(tickers),
        "new_tickers_saved": total_rows,
        "already_existing": len([t for t in tickers if t in existing]),
        "missing": sorted(set(missing)),
        "missing_count": len(missing),
        "failed": sorted(set(failed)),
        "failed_count": len(failed),
        "duration_s": round(total_time, 1),
    }

    report_path = LOG_DIR / f"{SCRIPT_NAME}_{log_ts}_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    size_str = fmt_bytes(out.stat().st_size) if out.exists() else "0B"

    logger.info("=" * 60)
    logger.info("SUMMARY REPORT")
    logger.info("  Duration:       %.1fs", total_time)
    logger.info("  Output file:    %s (%s)", out, size_str)
    logger.info("  New rows saved: %d", total_rows)
    if missing:
        logger.info("  Missing tickers (%d): %s", len(missing), ", ".join(missing))
    if failed:
        logger.info("  Failed tickers (%d): %s", len(failed), ", ".join(failed))
    logger.info("  Report:         %s", report_path)


if __name__ == "__main__":
    main()
