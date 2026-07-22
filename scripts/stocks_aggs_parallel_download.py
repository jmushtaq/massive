"""
Parallel dispatcher for stocks_aggs_download.py.

Spawns N worker subprocesses (each running stocks_aggs_download.py)
and distributes tickers from a CSV file or from saved OHLCV filenames
across them. One ticker per worker at a time.

State file: data/SPY/.parallel_state_<year>_<aggregate>.json
  Records completed, in-progress, and timing metrics per ticker.
  Used by stocks_aggs_parallel_status.py for live monitoring.

Usage:
    python scripts/stocks_aggs_parallel_download.py --tickers_file data/universes/2025/combined_unique.csv --year 2025 --spawn 12

    python scripts/stocks_aggs_parallel_download.py --tickers_file data/spy_tickers/tickers_combined_unique.csv --year 2025 --spawn 8 --resume --parquet

    python scripts/stocks_aggs_parallel_download.py --ohlcv_tickers --year 2022 --spawn 12 --parquet

    python scripts/stocks_aggs_parallel_download.py --tickers_file foo.csv --year 2025 --spawn 10 --output data/combined
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
WORKER_SCRIPT = SCRIPT_DIR / "stocks_aggs_download.py"

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


def parse_years(year_arg: str) -> list[str]:
    parts = year_arg.split("-")
    if len(parts) == 1:
        y = parts[0].strip()
        if not y.isdigit():
            raise SystemExit("Error: invalid year '%s'" % year_arg)
        return [y]
    elif len(parts) == 2:
        start, end = parts[0].strip(), parts[1].strip()
        if not start.isdigit() or not end.isdigit():
            raise SystemExit("Error: invalid year range '%s'" % year_arg)
        return [str(y) for y in range(int(start), int(end) + 1)]
    else:
        raise SystemExit("Error: invalid year format '%s' (use YYYY or YYYY-YYYY)" % year_arg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel download stock aggregate bars using multiple worker processes"
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
        help="Path to CSV with ticker list (header 'ticker'; may also have market_cap,rank columns)",
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
        help="Skip tickers that already have a non-empty output file (default: True)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output directory passed to workers via --output (default: data/)",
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
    pattern = "*_%s_%s.csv" % (year, folder)
    tickers = []
    for f in sorted(src_dir.glob(pattern)):
        ticker = f.stem.split("_")[0]
        tickers.append(clean_ticker(ticker))
    if not tickers:
        raise SystemExit("Error: no OHLCV files matching '%s' in %s" % (pattern, src_dir))
    return tickers


def output_path(ticker: str, year: str, agg: str, parquet: bool = False, output_dir: str | None = None) -> Path:
    folder = AGGREGATE_MAP[agg][2]
    ext = "parquet" if parquet else "csv"
    base = Path(output_dir) if output_dir else Path("data")
    return base / "SPY" / folder / year / f"{ticker}_{year}_{folder}.{ext}"


def tx_key(ticker: str, year: str) -> str:
    return f"{ticker}_{year}"


def is_ticker_final(ticker: str, year: str, agg: str, parquet: bool, output_dir: str | None = None) -> bool:
    p = output_path(ticker, year, agg, parquet, output_dir)
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


def run_year(agg: str, year: str, args) -> dict:
    folder = AGGREGATE_MAP[agg][2]
    year_start = time.time()

    state_base = Path(args.output) if args.output else Path("data") / "SPY"
    state_path = state_base / f".parallel_state_{year}_{folder}.json"
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
            if is_ticker_final(t, year, agg, args.parquet, args.output):
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
            if is_ticker_final(t, year, agg, args.parquet, args.output):
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
        log_dir = state_base / folder / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        log_path = log_dir / f"parallel_{year}_{folder}_{log_ts}.log"
        log_fh = open(log_path, "w")

    state["config"] = {"workers": args.spawn, "aggregate": agg, "year": year}
    ticker_source = "ohlcv_tickers" if args.ohlcv_tickers else args.tickers_file
    log("=" * 60)
    log("PARALLEL STOCK AGGREGATE DOWNLOAD  [year %s]" % year)
    log("  Workers:      %d" % args.spawn)
    log("  Aggregate:    %s" % agg)
    log("  Format:       %s" % ("parquet" if args.parquet else "csv"))
    log("  Resume:       %s" % args.resume)
    log("  Logs:         %s" % ("enabled" if args.logs else "disabled"))
    if args.logs:
        log("  Log path:     %s" % log_path)
    log("  Ticker src:   %s" % ticker_source)
    log("  Skip compl:   %s" % args.skip_completed)
    if args.output:
        log("  Output base:  %s" % args.output)
    log("  Total:        %d tickers" % total)
    log("  Already done: %d" % completed_count)
    log("  Remaining:    %d" % remaining)
    log("  State file:   %s" % state_path)
    log("=" * 60)

    if remaining == 0:
        log("All tickers for %s already processed. Nothing to do." % year)
        if log_fh:
            log_fh.close()
        return {"year": year, "total": total, "completed": 0, "successful": 0, "no_data": 0, "failed": 0, "elapsed_s": 0, "skipped": True}

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
        if args.resume:
            cmd.append("--resume")
        if args.parquet:
            cmd.append("--parquet")
        if args.logs:
            cmd.append("--logs")
        if args.output:
            cmd.extend(["--output", args.output])
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

    year_elapsed = time.time() - year_start

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
        "elapsed_s": round(year_elapsed, 1),
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
    log("SUMMARY  [year %s]" % year)
    log("  Total:        %d" % total)
    log("  Successful:   %d" % len(completed_ok))
    log("  No data:      %d" % len(completed_no_data))
    log("  Failed:       %d" % len(completed_fail))
    log("  Duration:     %.1fs" % year_elapsed)
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

    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    report = {
        "script": "stocks_aggs_parallel_download",
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
        "duration_s": round(year_elapsed, 1),
        "completed": state["completed"],
        "stats": stats,
    }
    report_dir = state_base / folder
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"parallel_report_{year}_{folder}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("  Report:       %s" % report_path)

    return {
        "year": year,
        "total": total,
        "completed": len(completed_ok) + len(completed_no_data) + len(completed_fail),
        "successful": len(completed_ok),
        "no_data": len(completed_no_data),
        "failed": len(completed_fail),
        "elapsed_s": round(year_elapsed, 1),
        "skipped": False,
    }


def main():
    args = parse_args()
    overall_start = time.time()
    agg = args.aggregate
    years = parse_years(args.year)

    all_year_results = []
    for year in years:
        result = run_year(agg, year, args)
        all_year_results.append(result)

    total_elapsed = time.time() - overall_start
    total_successful = sum(r["successful"] for r in all_year_results)
    total_no_data = sum(r["no_data"] for r in all_year_results)
    total_failed = sum(r["failed"] for r in all_year_results)

    print()
    print("=" * 60)
    print("OVERALL SUMMARY (%d year(s): %s)" % (len(years), args.year))
    print("  Total elapsed:    %s" % datetime.timedelta(seconds=int(total_elapsed)))
    print("  Successful:       %d" % total_successful)
    print("  No data:          %d" % total_no_data)
    print("  Failed:           %d" % total_failed)
    print("=" * 60)


if __name__ == "__main__":
    main()
