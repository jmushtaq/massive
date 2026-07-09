"""
Download historical financial statement data for a list of tickers from the
Massive REST API and save per-ticker CSV files.

Usage:
    python scripts/financial_statements_download.py --tickers AAPL,NVDA,TSLA
    python scripts/financial_statements_download.py --tickers_file data/spy_tickers/tickers_combined_unique.csv
    python scripts/financial_statements_download.py --tickers AAPL --parquet

Output layout:
    data/financials/<ticker>_financials.csv
    data/financials/<ticker>_financials.parquet  (with --parquet)

This calls /vX/reference/financials for each ticker to fetch historical
SEC filing data (income statement, balance sheet, cash flow).

Fields: ticker, filing_date, period_end, fiscal_year, fiscal_period,
revenue, net_income, gross_profit, operating_income, ebitda,
assets, equity, liabilities, cash, debt, operating_cash_flow.

The limit param for this endpoint is capped at 100 by the API
(max rows per page, pagination handles the rest).

Resume mode skips tickers whose output file already exists and has data.
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
from massive.rest.models import StockFinancial

SCRIPT_NAME = Path(__file__).resolve().stem

CSV_HEADERS = [
    "ticker",
    "filing_date",
    "period_end",
    "fiscal_year",
    "fiscal_period",
    "revenue",
    "net_income",
    "gross_profit",
    "operating_income",
    "ebitda",
    "assets",
    "equity",
    "liabilities",
    "cash",
    "debt",
    "operating_cash_flow",
]

OUTPUT_BASE = Path("data") / "financials"
LOG_DIR = OUTPUT_BASE / "logs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download historical financial statements data from Massive API"
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
        help="Skip tickers that already have a non-empty output file",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Write output as Parquet instead of CSV",
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


def output_path(ticker: str, parquet: bool = False) -> Path:
    ext = "parquet" if parquet else "csv"
    return OUTPUT_BASE / f"{ticker}_financials.{ext}"


def is_ticker_complete(ticker: str, parquet: bool = False) -> bool:
    path = output_path(ticker, parquet)
    if not path.exists():
        return False
    if parquet:
        return pq.read_table(path).num_rows > 0
    with open(path) as f:
        return sum(1 for _ in f) > 1


def safe_get(obj, *attrs):
    for attr in attrs:
        v = getattr(obj, attr, None) if obj else None
        if v is not None:
            return v.value if hasattr(v, "value") else v
    return None


def extract_row(sf: StockFinancial, ticker: str) -> dict:
    inc = sf.financials.income_statement if sf.financials else None
    bal = sf.financials.balance_sheet if sf.financials else None
    cf = sf.financials.cash_flow_statement if sf.financials else None
    return {
        "ticker": ticker,
        "filing_date": sf.filing_date or "",
        "period_end": sf.end_date or "",
        "fiscal_year": sf.fiscal_year or "",
        "fiscal_period": sf.fiscal_period or "",
        "revenue": safe_get(inc, "revenues"),
        "net_income": safe_get(inc, "net_income_loss"),
        "gross_profit": safe_get(inc, "gross_profit"),
        "operating_income": safe_get(inc, "operating_income_loss"),
        "ebitda": safe_get(inc, "ebitda"),
        "assets": safe_get(bal, "assets"),
        "equity": safe_get(bal, "equity"),
        "liabilities": safe_get(bal, "liabilities"),
        "cash": safe_get(bal, "cash"),
        "debt": safe_get(bal, "long_term_debt"),
        "operating_cash_flow": safe_get(cf, "net_cash_flow_from_operating_activities"),
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

    logger.info("Fetching financial statements for %d tickers", len(tickers))
    logger.info("Output format: %s", "parquet" if args.parquet else "csv")
    logger.info("Resume mode: %s", args.resume)
    logger.info("Output base: %s", OUTPUT_BASE.resolve())

    client = RESTClient(trace=True)

    missing: list[str] = []
    results: list[dict] = []
    downloaded = 0

    for i, ticker in enumerate(tickers, 1):
        if args.resume and is_ticker_complete(ticker, args.parquet):
            logger.info("[%d/%d] %s -> already complete, skipping", i, len(tickers), ticker)
            results.append({"ticker": ticker, "status": "skipped"})
            continue

        logger.info("[%d/%d] %s -> downloading ...", i, len(tickers), ticker)

        t0 = time.time()
        try:
            rows = []
            for sf in client.vx.list_stock_financials(ticker=ticker, limit=100):
                if sf.end_date:
                    rows.append(extract_row(sf, ticker))
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("[%d/%d] %s -> FAILED after %.1fs: %s", i, len(tickers), ticker, elapsed, e)
            missing.append(ticker)
            results.append({"ticker": ticker, "status": "failed", "error": str(e), "elapsed_s": round(elapsed, 1)})
            continue

        elapsed = time.time() - t0

        if not rows:
            logger.warning("[%d/%d] %s -> no data returned (%.1fs)", i, len(tickers), ticker, elapsed)
            missing.append(ticker)
            results.append({"ticker": ticker, "status": "no_data", "elapsed_s": round(elapsed, 1)})
        else:
            out = output_path(ticker, args.parquet)
            out.parent.mkdir(parents=True, exist_ok=True)

            if args.parquet:
                table = pa.Table.from_pylist(rows)
                pq.write_table(table, out)
            else:
                with open(out, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    writer.writeheader()
                    writer.writerows(rows)

            size = out.stat().st_size
            logger.info(
                "[%d/%d] %s -> %d rows (%s, %.1fs) -> %s",
                i, len(tickers), ticker, len(rows), fmt_bytes(size), elapsed, out,
            )
            downloaded += 1
            results.append({
                "ticker": ticker, "status": "ok", "rows": len(rows),
                "size_bytes": size, "path": str(out), "elapsed_s": round(elapsed, 1),
            })

        time.sleep(0.25)

    total_time = time.time() - overall_start

    summary = {
        "script": SCRIPT_NAME,
        "timestamp": log_ts,
        "format": "parquet" if args.parquet else "csv",
        "total_tickers": len(tickers),
        "downloaded": downloaded,
        "missing": sorted(set(missing)),
        "missing_count": len(set(missing)),
        "duration_s": round(total_time, 1),
        "results": results,
    }

    report_path = LOG_DIR / f"{SCRIPT_NAME}_{log_ts}_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("SUMMARY REPORT")
    logger.info("  Duration:       %.1fs", total_time)
    logger.info("  Downloaded:     %d", downloaded)
    skips = len([r for r in results if r["status"] == "skipped"])
    if skips:
        logger.info("  Skipped:        %d", skips)
    logger.info("  Missing/failed: %d", len(set(missing)))
    missing_unique = sorted(set(missing))
    if missing_unique:
        logger.info("  Missing tickers: %s", ", ".join(missing_unique))
    logger.info("  Report:         %s", report_path)


if __name__ == "__main__":
    main()
