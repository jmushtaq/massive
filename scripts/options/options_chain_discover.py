"""
Discover tradable options contracts for a list of tickers by querying the
Massive REST API and filtering by moneyness (proximity to underlying price).

Outputs a contract manifest CSV that feeds options_chain_download.py.

Output layout:
    data/options/chains/contract_manifest_<year>.csv

Usage:
    python scripts/options/options_chain_discover.py --tickers AAPL,TSLA --year 2025
    python scripts/options/options_chain_discover.py --tickers_file data/universes/2025/combined_unique.csv --year 2025
    python scripts/options/options_chain_discover.py --ohlcv_tickers --year 2025
    python scripts/options/options_chain_discover.py --ohlcv_tickers --year 2025 --output data/combined
"""

import argparse
import csv
import datetime
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("APIKEY")
if not api_key:
    raise ValueError("APIKEY not found in .env")
os.environ["MASSIVE_API_KEY"] = api_key

from massive import RESTClient

SCRIPT_NAME = Path(__file__).resolve().stem

MANIFEST_HEADERS = [
    "ticker",
    "contract_symbol",
    "strike",
    "expiration_date",
    "contract_type",
    "shares_per_contract",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover tradable options contracts via the Massive REST API"
    )
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker symbols (e.g. AAPL,TSLA,NVDA)")
    parser.add_argument("--tickers_file", type=str, default=None,
                        help="Path to CSV with ticker list (header 'ticker')")
    parser.add_argument("--ohlcv_tickers", action="store_true", default=False,
                        help="Derive ticker list from saved OHLCV files in data/SPY/1min/<year>/")
    parser.add_argument("--year", type=str, required=True,
                        help="Year (e.g. 2025) — finds contracts expiring in this year range")
    parser.add_argument("--output", type=str, default=None,
                        help="Base output directory (default: data/)")
    parser.add_argument("--moneyness_pct", type=float, default=0.20,
                        help="Max absolute moneyness as fraction (default: 0.20 = ±20%% of underlying)")
    parser.add_argument("--aggregate", choices=["1sec", "1min"], default="1min",
                        help="Aggregate window size (default: 1min, for CLI consistency)")
    parser.add_argument("--delay", type=float, default=0.25,
                        help="Sleep seconds between API calls (default: 0.25)")
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
            ticker = name.split("_")[0]
            tickers.append(clean_ticker(ticker))
    if not tickers:
        raise SystemExit("Error: specify at least one of --tickers, --tickers_file, or --ohlcv_tickers")
    return list(dict.fromkeys(tickers))


def get_underlying_price_range(ticker: str, year: str) -> tuple[float, float]:
    src_dir = Path("data") / "SPY" / "1min" / year
    for ext in (".csv", ".csv.gz"):
        path = src_dir / f"{ticker}_{year}_1min{ext}"
        if path.exists():
            import gzip
            opener = gzip.open if ext == ".csv.gz" else open
            prices = []
            with opener(path, "rt", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        prices.append(float(row["close"]))
                    except (ValueError, KeyError):
                        continue
            if prices:
                return min(prices), max(prices)
    for ext in (".csv", ".csv.gz"):
        path = Path("data") / "SPY" / "1D" / year / f"{ticker}_{year}_1D{ext}"
        if path.exists():
            import gzip
            opener = gzip.open if ext == ".csv.gz" else open
            prices = []
            with opener(path, "rt", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        prices.append(float(row["close"]))
                    except (ValueError, KeyError):
                        continue
            if prices:
                return min(prices), max(prices)
    return 0.0, 0.0


def discover_contracts(client: RESTClient, ticker: str, year: str,
                       moneyness_pct: float, logger: logging.Logger) -> list[dict]:
    low_p, high_p = get_underlying_price_range(ticker, year)
    if low_p <= 0 or high_p <= 0:
        logger.warning("  [%s] no underlying OHLCV data, skipping", ticker)
        return []

    mid_price = (low_p + high_p) / 2.0
    strike_min = mid_price * (1.0 - moneyness_pct)
    strike_max = mid_price * (1.0 + moneyness_pct)

    year_int = int(year)
    exp_from = f"{year_int}-01-01"
    exp_to = f"{year_int + 1}-03-31"

    contracts = []
    try:
        for c in client.list_options_contracts(
            underlying_ticker=ticker,
            expiration_date_gte=exp_from,
            expiration_date_lte=exp_to,
            expired=True,
            limit=1000,
        ):
            if not c.ticker or not c.strike_price or not c.expiration_date:
                continue
            if c.strike_price < strike_min or c.strike_price > strike_max:
                continue
            contracts.append({
                "ticker": ticker,
                "contract_symbol": c.ticker,
                "strike": c.strike_price,
                "expiration_date": c.expiration_date,
                "contract_type": c.contract_type or "",
                "shares_per_contract": c.shares_per_contract or "",
            })
    except Exception as e:
        logger.error("  [%s] API error: %s", ticker, e)
        return []

    logger.info("  [%s] %d contracts (price range %.2f-%.2f, strikes %.2f-%.2f)",
                ticker, len(contracts), low_p, high_p, strike_min, strike_max)
    return contracts


def main():
    args = parse_args()
    overall_start = time.time()

    log_dir = Path(args.output or "data") / "options" / "chains" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / f"{SCRIPT_NAME}_{log_ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(SCRIPT_NAME)

    tickers = load_tickers(args)
    year = args.year.split("-")[0]

    output_base = Path(args.output) if args.output else Path("data")
    manifest_dir = output_base / "options" / "chains"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"contract_manifest_{year}.csv"

    logger.info("=" * 60)
    logger.info("OPTIONS CHAIN DISCOVERY  [year %s]", year)
    logger.info("  Tickers:      %d", len(tickers))
    logger.info("  Moneyness:    ±%.0f%%", args.moneyness_pct * 100)
    logger.info("  Manifest:     %s", manifest_path)
    logger.info("  Log:          %s", log_path)
    logger.info("=" * 60)

    client = RESTClient(trace=True)
    all_contracts = []
    tickers_with_data = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] %s ...", i, len(tickers), ticker)
        contracts = discover_contracts(client, ticker, year, args.moneyness_pct, logger)
        if contracts:
            all_contracts.extend(contracts)
            tickers_with_data += 1
        if args.delay > 0 and i < len(tickers):
            time.sleep(args.delay)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADERS)
        writer.writeheader()
        for c in all_contracts:
            writer.writerow(c)

    total_time = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Manifest:      %s", manifest_path)
    logger.info("  Tickers found: %d / %d", tickers_with_data, len(tickers))
    logger.info("  Contracts:     %d", len(all_contracts))
    logger.info("  Duration:      %.1fs", total_time)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
