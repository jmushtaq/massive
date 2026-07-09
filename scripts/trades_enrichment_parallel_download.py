"""
Parallel dispatcher for trades_enrichment_download.py.

Spawns N worker subprocesses (each running trades_enrichment_download.py)
and distributes tickers from a CSV file or from saved OHLCV filenames
across them. One ticker per worker at a time — as a worker finishes,
the next ticker is assigned.

Usage:
    python scripts/trades_enrichment_parallel_download.py --tickers_file data/spy_tickers/tickers_combined_unique.csv --year 2010 --spawn 12

    python scripts/trades_enrichment_parallel_download.py \\
        --tickers_file data/spy_tickers/tickers_combined_unique.csv \\
        --year 2025 --spawn 8 --parquet --resume --logs

    python scripts/trades_enrichment_parallel_download.py \\
        --ohlcv_tickers --year 2022 --spawn 12 --parquet

State file: data/trades/.parallel_state_<year>_<aggregate>.json
  Records completed, in-progress, and timing metrics per ticker.
  Used by trades_enrichment_parallel_status.py for live monitoring.
"""

import argparse
import csv
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "trades_enrichment_download.py"

AGGREGATE_MAP = {
    "1min": (1, "minute", "1min"),
    "5min": (5, "minute", "5min"),
    "15min": (15, "minute", "15min"),
    "1H": (1, "hour", "1H"),
    "4H": (4, "hour", "4H"),
    "1D": (1, "day", "1D"),
}


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel download trade enrichment data using multiple worker processes"
    )
    parser.add_argument(
        "--aggregate",
        choices=list(AGGREGATE_MAP.keys()),
        default="1min",
        help="Aggregate window size (default: 1min)",
    )
    parser.add_argument(
        "--tickers_file",
        type=str,
        default=None,
        help="Path to CSV with ticker list (header 'ticker')",
    )
    parser.add_argument(
        "--ohlcv_tickers",
        action="store_true",
        default=False,
        help="Derive ticker list from saved OHLCV files in data/SPY/<aggregate>/<year>/",
    )
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help="Year to download (e.g. 2010)",
    )
    parser.add_argument(
        "--spawn",
        type=int,
        required=True,
        help="Number of parallel worker processes to spawn",
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
        "--logs",
        action="store_true",
        default=False,
        help="Save a dispatcher log file (default: False)",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        default=True,
        help="Skip tickers that already have a non-empty output file in the final dir (default: True)",
    )
    return parser.parse_args()


def load_tickers(tickers_file: str) -> list[str]:
    tickers = []
    with open(tickers_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("ticker", "").strip()
            if t:
                tickers.append(clean_ticker(t))
    if not tickers:
        raise SystemExit("Error: no tickers found in %s" % tickers_file)
    return tickers


def load_ohlcv_tickers(year: str, agg: str) -> list[str]:
    folder = AGGREGATE_MAP[agg][2]
    src_dir = Path("data") / "SPY" / folder / year
    if not src_dir.exists():
        raise SystemExit("Error: OHLCV directory not found: %s" % src_dir)
    pattern = f"*_{year}_{folder}.csv"
    tickers = []
    for f in sorted(src_dir.glob(pattern)):
        ticker = f.stem.split("_")[0]
        tickers.append(clean_ticker(ticker))
    if not tickers:
        raise SystemExit("Error: no OHLCV files matching '%s' in %s" % (pattern, src_dir))
    return tickers


def output_path(ticker: str, year: str, agg: str, parquet: bool = False) -> Path:
    folder = AGGREGATE_MAP[agg][2]
    ext = "parquet" if parquet else "csv"
    return Path("data") / "trades" / folder / year / f"{ticker}_{year}_{folder}_trades.{ext}"


def tx_key(ticker: str, year: str) -> str:
    return f"{ticker}_{year}"


def is_ticker_final(ticker: str, year: str, agg: str, parquet: bool) -> bool:
    p = output_path(ticker, year, agg, parquet)
    if not p.exists() or p.stat().st_size == 0:
        return False
    if parquet:
        import pyarrow.parquet as pq
        try:
            return pq.ParquetFile(p).metadata.num_rows > 0
        except Exception:
            return False
    with open(p) as f:
        return sum(1 for _ in f) > 1


def load_state(state_path: Path):
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"completed": {}, "in_progress": {}, "all_tickers": [], "stats": {}}


def save_state(state_path: Path, state: dict):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_path)


STDOUT_MARKER = "PARALLEL_RESULT:"
SCRIPT_NAME = Path(__file__).resolve().stem


def parse_worker_status(proc, ret: int) -> str:
    if ret != 0:
        return "failed"
    if proc.stdout:
        try:
            for line in proc.stdout:
                if STDOUT_MARKER in line:
                    result = json.loads(line.split(STDOUT_MARKER)[1].strip())
                    proc.stdout.close()
                    return result.get("status", "ok")
        except Exception:
            pass
        proc.stdout.close()
    return "ok"


def main():
    args = parse_args()
    overall_start = time.time()
    agg = args.aggregate
    year = args.year
    folder = AGGREGATE_MAP[agg][2]

    state_path = Path("data") / "trades" / f".parallel_state_{year}_{folder}.json"
    state = load_state(state_path)

    if args.ohlcv_tickers:
        all_tickers = load_ohlcv_tickers(year, agg)
    elif args.tickers_file:
        all_tickers = load_tickers(args.tickers_file)
    else:
        raise SystemExit("Error: specify one of --tickers_file or --ohlcv_tickers")

    if args.skip_completed:
        pre_filtered = []
        for t in all_tickers:
            if is_ticker_final(t, year, agg, args.parquet):
                continue
            pre_filtered.append(t)
        skipped_pre = len(all_tickers) - len(pre_filtered)
        all_tickers = pre_filtered
    else:
        skipped_pre = 0
    state["all_tickers"] = all_tickers

    ticker_queue = []
    for t in all_tickers:
        key = tx_key(t, year)
        if key in state.get("completed", {}):
            continue
        if args.resume:
            if is_ticker_final(t, year, agg, args.parquet):
                continue
        if key in state.get("in_progress", {}):
            continue
        ticker_queue.append(t)

    total = len(all_tickers)
    remaining = len(ticker_queue)
    completed_count = total - remaining + skipped_pre

    log_lines: list[str] = []

    def log(msg: str, end: str = "\n"):
        print(msg, end=end, flush=True)
        log_lines.append(msg)
        if log_fh:
            log_fh.write(msg + ("\n" if end == "\n" else end))
            log_fh.flush()

    log_fh = None
    if args.logs:
        log_dir = Path("data") / "trades" / folder / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        log_path = log_dir / f"parallel_{year}_{folder}_{log_ts}.log"
        log_fh = open(log_path, "w")

    ticker_source = "ohlcv_tickers" if args.ohlcv_tickers else args.tickers_file
    log("=" * 60)
    log("PARALLEL TRADE ENRICHMENT DOWNLOAD")
    log("  Workers:      %d" % args.spawn)
    log("  Year:         %s" % year)
    log("  Aggregate:    %s" % agg)
    log("  Format:       %s" % ("parquet" if args.parquet else "csv"))
    log("  Resume:       %s" % args.resume)
    log("  Logs:         %s" % ("enabled" if args.logs else "disabled"))
    if args.logs:
        log("  Log path:     %s" % log_path)
    log("  Ticker src:   %s" % ticker_source)
    log("  Skip compl:   %s" % args.skip_completed)
    log("  Total:        %d tickers" % total)
    log("  Already done: %d" % completed_count)
    log("  Remaining:    %d" % remaining)
    log("  State file:   %s" % state_path)
    log("=" * 60)

    if remaining == 0:
        log("All tickers already processed. Nothing to do.")
        return

    state["in_progress"] = {}
    save_state(state_path, state)

    queue = list(ticker_queue)
    queue.reverse()

    active_workers: list[dict] = []
    errors_occurred = False

    def spawn_worker(ticker: str) -> subprocess.Popen | None:
        nonlocal errors_occurred
        key = tx_key(ticker, year)
        cmd = [
            sys.executable,
            str(WORKER_SCRIPT),
            "--tickers", ticker,
            "--year", year,
            "--aggregate", agg,
        ]
        if args.parquet:
            cmd.append("--parquet")
        if args.resume:
            cmd.append("--resume")
        if args.logs:
            cmd.append("--logs")
        try:
            stderr_file = open(f"/tmp/worker_{ticker}_{year}.log", "w")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
            )
        except Exception as e:
            log("[ERROR] Failed to spawn worker for %s: %s" % (ticker, e))
            errors_occurred = True
            return None
        entry = {
            "ticker": ticker,
            "pid": proc.pid,
            "proc": proc,
            "start_time": time.time(),
            "stderr_file": stderr_file,
        }
        state["in_progress"][key] = {"pid": proc.pid, "start_time": entry["start_time"]}
        save_state(state_path, state)
        active_workers.append(entry)
        return proc

    for _ in range(min(args.spawn, len(queue))):
        ticker = queue.pop()
        spawn_worker(ticker)

    def reap_finished():
        nonlocal errors_occurred
        still_active = []
        for entry in active_workers:
            proc = entry["proc"]
            ret = proc.poll()
            if ret is not None:
                ticker = entry["ticker"]
                key = tx_key(ticker, year)
                duration = time.time() - entry["start_time"]

                worker_status = parse_worker_status(proc, ret)
                if worker_status == "failed":
                    errors_occurred = True

                entry["stderr_file"].close()
                state["completed"][key] = {
                    "ticker": ticker,
                    "duration_s": round(duration, 1),
                    "returncode": ret,
                    "status": worker_status,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                if key in state["in_progress"]:
                    del state["in_progress"][key]
                save_state(state_path, state)
                log("[%s] %s finished in %.1fs" % (worker_status.upper(), ticker, duration))
            else:
                still_active.append(entry)
        active_workers[:] = still_active

    try:
        while active_workers or queue:
            reap_finished()
            while len(active_workers) < args.spawn and queue:
                ticker = queue.pop()
                spawn_worker(ticker)
            if active_workers:
                time.sleep(1)
    except KeyboardInterrupt:
        log("\n[INTERRUPT] Terminating workers ...")
        for entry in active_workers:
            try:
                os.kill(entry["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for entry in active_workers:
            entry["proc"].wait(timeout=5)
        log("All workers terminated.")
        sys.exit(130)

    elapsed = time.time() - overall_start

    # Separate timing stats: exclude no_data tickers
    completed_ok = [k for k, v in state["completed"].items() if v.get("status") == "ok"]
    completed_no_data = [k for k, v in state["completed"].items() if v.get("status") == "no_data"]
    completed_fail = [k for k, v in state["completed"].items() if v.get("status") == "failed"]

    data_durations = [state["completed"][k]["duration_s"] for k in completed_ok]
    all_durations = [v["duration_s"] for v in state["completed"].values()]

    stats = {
        "total_tickers": total,
        "completed": len(state["completed"]),
        "successful": len(completed_ok),
        "no_data": len(completed_no_data),
        "failed": len(completed_fail),
        "elapsed_s": round(elapsed, 1),
    }
    if data_durations:
        stats["data_avg_time_s"] = round(sum(data_durations) / len(data_durations), 1)
        stats["data_min_time_s"] = round(min(data_durations), 1)
        stats["data_max_time_s"] = round(max(data_durations), 1)
    if all_durations:
        stats["avg_time_s"] = round(sum(all_durations) / len(all_durations), 1)
        stats["min_time_s"] = round(min(all_durations), 1)
        stats["max_time_s"] = round(max(all_durations), 1)
    state["stats"] = stats
    save_state(state_path, state)

    log("=" * 60)
    log("SUMMARY")
    log("  Total:        %d" % total)
    log("  Successful:   %d" % len(completed_ok))
    log("  No data:      %d" % len(completed_no_data))
    log("  Failed:       %d" % len(completed_fail))
    log("  Duration:     %.1fs" % elapsed)
    if data_durations:
        log("  Avg/ticker (with data):   %.1fs" % stats["data_avg_time_s"])
        log("  Min/ticker (with data):   %.1fs" % stats["data_min_time_s"])
        log("  Max/ticker (with data):   %.1fs" % stats["data_max_time_s"])
    if all_durations:
        log("  Avg/ticker (all):         %.1fs" % stats["avg_time_s"])
        log("  Min/ticker (all):         %.1fs" % stats["min_time_s"])
        log("  Max/ticker (all):         %.1fs" % stats["max_time_s"])
    log("  State file:   %s" % state_path)
    if completed_fail:
        log("  Failed keys:  %s" % ", ".join(completed_fail))
    log("=" * 60)

    if log_fh:
        log_fh.close()

    # Write report always
    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    report = {
        "script": "trades_enrichment_parallel_download",
        "timestamp": log_ts,
        "year": year,
        "aggregate": agg,
        "format": "parquet" if args.parquet else "csv",
        "workers": args.spawn,
        "resume": args.resume,
        "logs": args.logs,
        "total_tickers": total,
        "successful": len(completed_ok),
        "no_data": len(completed_no_data),
        "failed": len(completed_fail),
        "duration_s": round(elapsed, 1),
        "completed": state["completed"],
        "stats": stats,
    }
    report_dir = Path("data") / "trades" / folder
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"parallel_report_{year}_{folder}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("  Report:       %s" % report_path)


if __name__ == "__main__":
    main()
