"""
Download historical US Treasury yield curve data from the Massive REST API.

Saves daily yield data for maturities from 1 month to 30 years.
Used by Greeks computation scripts for risk-free rate interpolation.

Output: data/treasury-yields/treasury_yields.csv

Usage:
    python scripts/options/download_treasury_yields.py
"""

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

TENOR_MAP = [
    ("yield_1_month", "1m", 30),
    ("yield_3_month", "3m", 90),
    ("yield_6_month", "6m", 180),
    ("yield_1_year", "1y", 365),
    ("yield_2_year", "2y", 730),
    ("yield_3_year", "3y", 1095),
    ("yield_5_year", "5y", 1825),
    ("yield_7_year", "7y", 2555),
    ("yield_10_year", "10y", 3650),
    ("yield_20_year", "20y", 7300),
    ("yield_30_year", "30y", 10950),
]

HEADERS = ["date"] + [t[1] for t in TENOR_MAP]


def main():
    overall_start = time.time()

    out_dir = Path("data") / "treasury-yields"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "treasury_yields.csv"

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / f"{SCRIPT_NAME}_{log_ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(SCRIPT_NAME)

    logger.info("=" * 60)
    logger.info("TREASURY YIELDS DOWNLOAD")
    logger.info("  Output: %s", out_path)
    logger.info("=" * 60)

    client = RESTClient(trace=True)
    rows = []

    try:
        for y in client.list_treasury_yields(limit=50000, order="asc"):
            row = {"date": y.date}
            for attr, col_name, _ in TENOR_MAP:
                val = getattr(y, attr, None)
                row[col_name] = str(val) if val is not None else ""
            rows.append(row)
    except Exception as e:
        logger.error("API error: %s", e)
        sys.exit(1)

    if not rows:
        logger.error("No data returned.")
        sys.exit(1)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in HEADERS})

    total_time = time.time() - overall_start
    logger.info("  Downloaded %d dates covering %s to %s (%.1fs)",
                len(rows), rows[0]["date"], rows[-1]["date"], total_time)
    logger.info("  Written to %s", out_path)


if __name__ == "__main__":
    main()
