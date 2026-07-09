"""
Download aggregate bar data for a list of tickers from the Massive (REST) API
and save each ticker as a CSV or Parquet file.

Usage:
    python scripts/stocks_aggs_download.py --tickers AAPL,NVDA,TSLA --year 2025
    python scripts/stocks_aggs_download.py --tickers_file data/spy_tickers/tickers_combined_unique.csv --year 2025
    python scripts/stocks_aggs_download.py --tickers AAPL --year 2022-2025 --aggregate 1H --resume
    python scripts/stocks_aggs_download.py --tickers AAPL --year 2025 --parquet

Output layout:
    data/SPY/<aggregate>/<year>/<ticker>_<year>_<aggregate>.csv
    data/SPY/<aggregate>/<year>/<ticker>_<year>_<aggregate>.parquet  (with --parquet)

Resume behaviour:
    The script checks the output directory for already-saved tickers. If --resume
    is set, it skips tickers whose output file already exists AND has at least
    one row. If the file is empty (e.g. from a previous failure), the
    ticker is re-downloaded.

    Tickers that return zero aggregates are logged at the end in the summary
    report alongside output stats.

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
from massive.rest.models import Agg

SCRIPT_NAME = Path(__file__).resolve().stem

AGGREGATE_MAP = {
    "1min": (1, "minute", "1min"),
    "5min": (5, "minute", "5min"),
    "15min": (15, "minute", "15min"),
    "1H": (1, "hour", "1H"),
    "4H": (4, "hour", "4H"),
    "1D": (1, "day", "1D"),
}

CSV_HEADERS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
    "otc",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download aggregate bar data for tickers from Massive API"
    )
    parser.add_argument(
        "--aggregate",
        choices=list(AGGREGATE_MAP.keys()),
        default="1min",
        help="Aggregate window size (default: 1min)",
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
        "--year",
        type=str,
        required=True,
        help="Year or year range (e.g. 2025 or 2022-2025)",
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


def agg_params(agg: str):
    return AGGREGATE_MAP[agg]


def output_base(agg: str) -> Path:
    return Path("data") / "SPY" / AGGREGATE_MAP[agg][2]


def output_path(ticker: str, year: str, agg: str, parquet: bool = False) -> Path:
    folder = AGGREGATE_MAP[agg][2]
    ext = "parquet" if parquet else "csv"
    return output_base(agg) / year / f"{ticker}_{year}_{folder}.{ext}"


def output_rows(ticker: str, year: str, agg: str, parquet: bool) -> int:
    path = output_path(ticker, year, agg, parquet)
    if not path.exists():
        return 0
    if parquet:
        return pq.read_table(path).num_rows
    with open(path) as f:
        return sum(1 for _ in f) - 1


def is_ticker_complete(ticker: str, year: str, agg: str, parquet: bool = False) -> bool:
    return output_rows(ticker, year, agg, parquet) > 0


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


def build_rows(aggs):
    for a in aggs:
        if isinstance(a, Agg) and isinstance(a.timestamp, int):
            yield {
                "timestamp": datetime.datetime.fromtimestamp(
                    a.timestamp / 1000
                ).isoformat(),
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": a.volume,
                "vwap": a.vwap,
                "transactions": a.transactions,
                "otc": a.otc,
            }


def download_ticker(client, ticker: str, year: str, agg: str, parquet: bool = False) -> int:
    from_date = f"{year}-01-01"
    to_date = f"{year}-12-31"
    multiplier, timespan, _ = agg_params(agg)

    aggs = []
    for a in client.list_aggs(
        ticker,
        multiplier,
        timespan,
        from_date,
        to_date,
        adjusted=True,
        limit=50000,
    ):
        aggs.append(a)

    if not aggs:
        return 0

    rows = list(build_rows(aggs))
    out = output_path(ticker, year, agg, parquet)
    out.parent.mkdir(parents=True, exist_ok=True)

    if parquet:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out)
    else:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(rows)

    return len(rows)


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
    agg = args.aggregate
    _, _, agg_folder = agg_params(agg)

    log_dir = Path("data") / "SPY" / agg_folder / "logs"
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

    logger.info("Starting download for %d tickers, years=%s, aggregate=%s", len(tickers), years, agg)
    logger.info("Output format: %s", "parquet" if args.parquet else "csv")
    logger.info("Resume mode: %s", args.resume)
    logger.info("Output base: %s", output_base(agg).resolve())

    client = RESTClient(trace=True)

    all_missing: list[str] = []
    all_results: list[dict] = []
    all_downloaded = 0
    all_skipped = 0

    for year in years:
        downloaded = 0
        skipped = 0

        for i, ticker in enumerate(tickers, 1):
            if args.resume and is_ticker_complete(ticker, year, agg, args.parquet):
                logger.info("[%d/%d] %s (%s) -> already complete, skipping", i, len(tickers), ticker, year)
                skipped += 1
                all_results.append({
                    "ticker": ticker, "year": year, "status": "skipped",
                    "rows": output_rows(ticker, year, agg, args.parquet),
                })
                continue

            logger.info("[%d/%d] %s (%s) -> downloading ...", i, len(tickers), ticker, year)

            t0 = time.time()
            try:
                count = download_ticker(client, ticker, year, agg, args.parquet)
            except Exception as e:
                elapsed = time.time() - t0
                logger.error("[%d/%d] %s (%s) -> FAILED after %.1fs: %s", i, len(tickers), ticker, year, elapsed, e)
                all_missing.append(ticker)
                all_results.append({"ticker": ticker, "year": year, "status": "failed", "error": str(e), "elapsed_s": round(elapsed, 1)})
                continue

            elapsed = time.time() - t0

            if count == 0:
                logger.warning("[%d/%d] %s (%s) -> no data returned (%.1fs)", i, len(tickers), ticker, year, elapsed)
                all_missing.append(ticker)
                all_results.append({"ticker": ticker, "year": year, "status": "no_data", "elapsed_s": round(elapsed, 1)})
            else:
                path = output_path(ticker, year, agg, args.parquet)
                size = path.stat().st_size
                logger.info(
                    "[%d/%d] %s (%s) -> %d bars (%s, %.1fs) -> %s",
                    i, len(tickers), ticker, year, count, fmt_bytes(size), elapsed, path,
                )
                downloaded += 1
                all_results.append({
                    "ticker": ticker, "year": year, "status": "ok",
                    "rows": count, "size_bytes": size, "path": str(path),
                    "elapsed_s": round(elapsed, 1),
                })

            time.sleep(0.25)

        all_downloaded += downloaded
        all_skipped += skipped

    total_time = time.time() - overall_start

    summary = {
        "script": SCRIPT_NAME,
        "timestamp": log_ts,
        "years": years,
        "aggregate": agg,
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
