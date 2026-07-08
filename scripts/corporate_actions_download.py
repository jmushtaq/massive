"""
Download corporate actions data (splits, dividends, ticker changes) for a list
of tickers from the Massive REST API and save as per-category CSV files.

Usage:
    python scripts/corporate_actions_download.py --tickers AAPL,NVDA,TSLA
    python scripts/corporate_actions_download.py --tickers_file data/spy_tickers/tickers_combined_unique.csv
    python scripts/corporate_actions_download.py --tickers AAPL --parquet

Output layout:
    data/corporate_actions/splits.csv
    data/corporate_actions/dividends.csv
    data/corporate_actions/ticker_changes.csv
    data/corporate_actions/splits.parquet
    data/corporate_actions/dividends.parquet
    data/corporate_actions/ticker_changes.parquet   (with --parquet)

This script calls:
    - /v3/reference/splits for each ticker (stock splits + reverse splits)
    - /v3/reference/dividends for each ticker
    - /vX/reference/tickers/{ticker}/events for each ticker (ticker changes)

Resume mode skips tickers already present in the output files.
One of --tickers or --tickers_file must be specified.
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

SCRIPT_NAME = Path(__file__).resolve().stem

OUTPUT_BASE = Path("data") / "corporate_actions"
LOG_DIR = OUTPUT_BASE / "logs"

SPLIT_HEADERS = ["ticker", "execution_date", "split_from", "split_to", "id"]
DIVIDEND_HEADERS = [
    "ticker", "ex_dividend_date", "cash_amount", "currency",
    "declaration_date", "record_date", "pay_date", "frequency",
    "dividend_type", "id",
]
TICKER_CHANGE_HEADERS = ["ticker", "date", "old_ticker", "new_ticker"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download corporate actions data from Massive API"
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
        help="Skip tickers already present in the output files",
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


def file_path(name: str, parquet: bool) -> Path:
    ext = "parquet" if parquet else "csv"
    return OUTPUT_BASE / f"{name}.{ext}"


def get_existing_tickers(path: Path) -> set:
    if not path.exists():
        return set()
    existing = set()
    if path.suffix == ".parquet":
        table = pq.read_table(path)
        col = table.column("ticker").to_pylist() if "ticker" in table.column_names else []
        for t in col:
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


def fmt_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def append_rows(path: Path, headers: list[str], rows: list[dict], parquet: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if parquet:
        new_table = pa.Table.from_pylist(rows)
        if path.exists():
            old = pq.read_table(path)
            combined = pa.concat_tables([old, new_table])
            pq.write_table(combined, path)
        else:
            pq.write_table(new_table, path)
    else:
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


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

    logger.info("Downloading corporate actions for %d tickers", len(tickers))
    logger.info("Output format: %s", "parquet" if args.parquet else "csv")
    logger.info("Resume mode: %s", args.resume)

    client = RESTClient(trace=True)

    split_path = file_path("splits", args.parquet)
    div_path = file_path("dividends", args.parquet)
    change_path = file_path("ticker_changes", args.parquet)

    existing_splits = get_existing_tickers(split_path) if args.resume else set()
    existing_divs = get_existing_tickers(div_path) if args.resume else set()
    existing_changes = get_existing_tickers(change_path) if args.resume else set()

    split_rows: list[dict] = []
    div_rows: list[dict] = []
    change_rows: list[dict] = []
    failed: list[str] = []

    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] %s -> processing ...", i, len(tickers), ticker)

        # Splits
        if ticker not in existing_splits:
            t0 = time.time()
            try:
                for s in client.list_splits(ticker=ticker, limit=1000):
                    split_rows.append({
                        "ticker": s.ticker or ticker,
                        "execution_date": s.execution_date or "",
                        "split_from": s.split_from,
                        "split_to": s.split_to,
                        "id": s.id,
                    })
            except Exception as e:
                logger.error("[%d/%d] %s -> splits FAILED: %s", i, len(tickers), ticker, e)
            else:
                logger.info("[%d/%d] %s -> %d splits (%.1fs)", i, len(tickers), ticker, len(split_rows) - len(existing_splits), time.time() - t0)

        # Dividends
        if ticker not in existing_divs:
            t0 = time.time()
            try:
                for d in client.list_dividends(ticker=ticker, limit=1000):
                    div_rows.append({
                        "ticker": d.ticker or ticker,
                        "ex_dividend_date": d.ex_dividend_date or "",
                        "cash_amount": d.cash_amount,
                        "currency": d.currency or "",
                        "declaration_date": d.declaration_date or "",
                        "record_date": d.record_date or "",
                        "pay_date": d.pay_date or "",
                        "frequency": d.frequency,
                        "dividend_type": d.dividend_type or "",
                        "id": d.id,
                    })
            except Exception as e:
                logger.error("[%d/%d] %s -> dividends FAILED: %s", i, len(tickers), ticker, e)
            else:
                logger.info("[%d/%d] %s -> %d dividends (%.1fs)", i, len(tickers), ticker, len(div_rows) - len(existing_divs), time.time() - t0)

        # Ticker changes
        if ticker not in existing_changes:
            t0 = time.time()
            try:
                raw = client.get_ticker_events(ticker, raw=True)
                data = json.loads(raw.data.decode("utf-8"))
                results = data.get("results", {})
                for ev in results.get("events", []):
                    tc = ev.get("ticker_change", {})
                    change_rows.append({
                        "ticker": ticker,
                        "date": ev.get("date", ""),
                        "old_ticker": tc.get("ticker", ""),
                        "new_ticker": ticker,
                    })
            except Exception as e:
                logger.error("[%d/%d] %s -> ticker changes FAILED: %s", i, len(tickers), ticker, e)
            else:
                ct = len(change_rows) - len(existing_changes)
                if ct:
                    logger.info("[%d/%d] %s -> %d ticker changes (%.1fs)", i, len(tickers), ticker, ct, time.time() - t0)

        time.sleep(0.25)

    logger.info("Writing files ...")

    append_rows(split_path, SPLIT_HEADERS, split_rows, args.parquet)
    append_rows(div_path, DIVIDEND_HEADERS, div_rows, args.parquet)
    append_rows(change_path, TICKER_CHANGE_HEADERS, change_rows, args.parquet)

    total_time = time.time() - overall_start

    summary = {
        "script": SCRIPT_NAME,
        "timestamp": log_ts,
        "format": "parquet" if args.parquet else "csv",
        "total_tickers": len(tickers),
        "new_splits": len(split_rows),
        "new_dividends": len(div_rows),
        "new_ticker_changes": len(change_rows),
        "failed": failed,
        "duration_s": round(total_time, 1),
    }

    report_path = LOG_DIR / f"{SCRIPT_NAME}_{log_ts}_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    size_s = fmt_bytes(split_path.stat().st_size) if split_path.exists() else "0B"
    size_d = fmt_bytes(div_path.stat().st_size) if div_path.exists() else "0B"
    size_c = fmt_bytes(change_path.stat().st_size) if change_path.exists() else "0B"

    logger.info("=" * 60)
    logger.info("SUMMARY REPORT")
    logger.info("  Duration:      %.1fs", total_time)
    logger.info("  Splits:        %d (%s)", len(split_rows), size_s)
    logger.info("  Dividends:     %d (%s)", len(div_rows), size_d)
    logger.info("  Ticker changes:%d (%s)", len(change_rows), size_c)
    if failed:
        logger.info("  Failed:        %d tickers", len(failed))
    logger.info("  Report:        %s", report_path)


if __name__ == "__main__":
    main()
