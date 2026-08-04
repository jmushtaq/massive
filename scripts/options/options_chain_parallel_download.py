"""
Parallel dispatcher for options_chain_download.py.

Spawns N worker subprocesses and distributes tickers from a CSV file,
--tickers list, or saved OHLCV filenames across them. One ticker per worker.

Workers are killed on KeyboardInterrupt. State file tracks progress for
resume and status monitoring.

State file: data/options/chains/.parallel_state_<year>_chains.json

Usage:
    python scripts/options/options_chain_parallel_download.py --tickers AAPL,TSLA --year 2025 --spawn 16
    python scripts/options/options_chain_parallel_download.py --tickers_file data/universes/2025/combined_unique.csv --year 2025 --spawn 50
    python scripts/options/options_chain_parallel_download.py --ohlcv_tickers --year 2025 --spawn 50
    python scripts/options/options_chain_parallel_download.py --ohlcv_tickers --year 2025 --spawn 50 --resume --output data/combined
"""

import argparse
import csv
import datetime
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "options_chain_download.py"
STDOUT_MARKER = "PARALLEL_RESULT:"
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


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel download options chain bars")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker symbols (e.g. AAPL,TSLA,NVDA)")
    parser.add_argument("--tickers_file", type=str, default=None,
                        help="Path to CSV with ticker list (header 'ticker')")
    parser.add_argument("--ohlcv_tickers", action="store_true", default=False,
                        help="Derive ticker list from saved OHLCV files in data/SPY/1min/<year>/")
    parser.add_argument("--year", type=str, required=True,
                        help="Year to download (e.g. 2025)")
    parser.add_argument("--aggregate", choices=list(AGGREGATE_MAP.keys()),
                        default="1min",
                        help="Aggregate window size (default: 1min)")
    parser.add_argument("--spawn", type=int, required=True,
                        help="Number of parallel worker processes to spawn")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tickers that already have a non-empty output file")
    parser.add_argument("--logs", action="store_true", default=False,
                        help="Save a dispatcher log file")
    parser.add_argument("--output", type=str, default=None,
                        help="Base output directory passed to workers via --output (default: data/)")
    parser.add_argument("--check", type=str, default=None,
                        help="Directory to check for existing per-ticker files")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Sleep seconds between API calls in the worker (default: 0.1)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to contract manifest CSV. Default: data/options/chains/contract_manifest_<year>.csv")
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


def load_ohlcv_tickers(year: str) -> list[str]:
    src_dir = Path("data") / "SPY" / "1min" / year
    if not src_dir.exists():
        raise SystemExit("Error: OHLCV directory not found: %s" % src_dir)
    tickers = []
    for f in sorted(src_dir.glob(f"*_{year}_1min.csv*")):
        name = f.name.replace(".csv.gz", "").replace(".csv", "")
        ticker = name.split("_")[0]
        tickers.append(clean_ticker(ticker))
    if not tickers:
        raise SystemExit("Error: no OHLCV files found in %s" % src_dir)
    return tickers


def output_path(ticker: str, year: str, agg: str, output_dir: str | None = None,
                subdir: str | None = None) -> Path:
    base = Path(output_dir) if output_dir else Path("data")
    p = base / "options" / "chains" / AGGREGATE_MAP[agg][2] / year
    if subdir:
        p = p / subdir
    return p / f"{ticker}_{year}_{AGGREGATE_MAP[agg][2]}_chains.csv"


def tx_key(ticker: str, year: str) -> str:
    return f"{ticker}_{year}"


def is_ticker_final(ticker: str, year: str, agg: str, output_dir: str | None = None) -> bool:
    p = output_path(ticker, year, agg, output_dir)
    if not p.exists() or p.stat().st_size == 0:
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


def parse_worker_status(proc, ret: int, result_line: str | None = None) -> str:
    if ret != 0:
        return "failed"
    if result_line and STDOUT_MARKER in result_line:
        try:
            result = json.loads(result_line.split(STDOUT_MARKER)[1].strip())
            return result.get("status", "ok")
        except Exception:
            pass
    return "ok"


def _drain_worker_stderr(entry):
    proc = entry["proc"]
    if not proc or not proc.stderr:
        return
    while select.select([proc.stderr], [], [], 0)[0]:
        line = proc.stderr.readline()
        if not line:
            break
        entry["stderr_file"].write(line)
        entry["stderr_file"].flush()
        if STDOUT_MARKER in line:
            entry["_result_line"] = line.strip()


def main():
    args = parse_args()
    overall_start = time.time()
    year = args.year
    agg = args.aggregate
    folder = AGGREGATE_MAP[agg][2]

    state_base = Path(args.output) if args.output else Path("data")
    state_path = state_base / "options" / "chains" / f".parallel_state_{year}_{folder}_chains.json"
    state = load_state(state_path)

    if args.ohlcv_tickers:
        all_tickers = load_ohlcv_tickers(year)
    elif args.tickers_file:
        all_tickers = load_tickers(args.tickers_file)
    elif args.tickers:
        all_tickers = [clean_ticker(t) for t in args.tickers.split(",") if t.strip()]
    else:
        raise SystemExit("Error: specify one of --tickers, --tickers_file, or --ohlcv_tickers")

    pre_filtered = []
    for t in all_tickers:
        if is_ticker_final(t, year, agg, args.output):
            continue
        pre_filtered.append(t)
    skipped_pre = len(all_tickers) - len(pre_filtered)
    all_tickers = pre_filtered

    if args.check:
        check_filtered = []
        for t in all_tickers:
            check_file = Path(args.check) / "options" / "chains" / year / f"{t}_{year}_chains.csv"
            if check_file.exists() and check_file.stat().st_size > 0:
                continue
            check_filtered.append(t)
        checked_skipped = len(all_tickers) - len(check_filtered)
        if checked_skipped:
            print("  Check dir skipped: %d tickers (file exists in %s)" % (checked_skipped, args.check))
        all_tickers = check_filtered
        skipped_pre += checked_skipped

    state["all_tickers"] = all_tickers

    ticker_queue = []
    for t in all_tickers:
        key = tx_key(t, year)
        if key in state.get("completed", {}):
            continue
        if args.resume and is_ticker_final(t, year, args.output):
            continue
        if key in state.get("in_progress", {}):
            continue
        ticker_queue.append(t)

    total = len(all_tickers)
    remaining = len(ticker_queue)
    completed_count = total - remaining + skipped_pre

    log_lines = []
    log_fh = None
    if args.logs:
        log_dir = state_base / "options" / "chains" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        log_path = log_dir / f"parallel_{year}_chains_{log_ts}.log"
        log_fh = open(log_path, "w")

    def log(msg: str, end: str = "\n"):
        print(msg, end=end, flush=True)
        log_lines.append(msg)
        if log_fh:
            log_fh.write(msg + ("\n" if end == "\n" else end))
            log_fh.flush()

    ticker_source = "ohlcv_tickers" if args.ohlcv_tickers else (args.tickers_file or "--tickers")
    log("=" * 60)
    log("PARALLEL OPTIONS CHAIN DOWNLOAD  [year %s]" % year)
    log("  Workers:      %d" % args.spawn)
    log("  Resume:       %s" % args.resume)
    log("  Delay:        %.1fs" % args.delay)
    log("  Logs:         %s" % ("enabled" if args.logs else "disabled"))
    log("  Ticker src:   %s" % ticker_source)
    if args.output:
        log("  Output base:  %s" % args.output)
    if args.check:
        log("  Check dir:    %s" % args.check)
    log("  Total:        %d tickers" % total)
    log("  Already done: %d" % completed_count)
    log("  Remaining:    %d" % remaining)
    log("  State file:   %s" % state_path)
    log("=" * 60)

    if remaining == 0:
        log("All tickers for %s already processed. Nothing to do." % year)
        if log_fh:
            log_fh.close()
        return

    log("")
    errors_occurred = False

    pending = list(ticker_queue)
    pending.reverse()
    active = []
    last_status_time = time.time()

    def spawn_worker(ticker: str) -> dict | None:
        nonlocal errors_occurred
        key = tx_key(ticker, year)
        cmd = [
            sys.executable, str(WORKER_SCRIPT),
            "--tickers", ticker,
            "--year", year,
            "--aggregate", agg,
            "--delay", str(args.delay),
            "--no_rename",
        ]
        if args.resume:
            cmd.append("--resume")
        if args.output:
            cmd.extend(["--output", args.output])
        if args.manifest:
            cmd.extend(["--manifest", args.manifest])
        try:
            stderr_file = open("/tmp/worker_chain_%s_%s.log" % (ticker, year), "w")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
        except Exception as e:
            log("[ERROR] Failed to spawn worker for %s: %s" % (ticker, e))
            errors_occurred = True
            return None
        entry = {
            "ticker": ticker, "pid": proc.pid, "proc": proc,
            "start_time": time.time(), "stderr_file": stderr_file,
            "_result_line": None,
        }
        state["in_progress"][key] = {"pid": proc.pid, "start_time": entry["start_time"]}
        state["stats"] = {"elapsed_s": round(time.time() - overall_start, 1),
                          "total_tickers": total, "completed": len(state["completed"]),
                          "running": len(state["in_progress"])}
        save_state(state_path, state)
        return entry

    try:
        while pending or active:
            while len(active) < args.spawn and pending:
                ticker = pending.pop()
                key = tx_key(ticker, year)
                if key in state.get("completed", {}):
                    continue
                if key in state.get("in_progress", {}):
                    continue
                entry = spawn_worker(ticker)
                if entry:
                    active.append(entry)

            if not active and not pending:
                break

            still_active = []
            for entry in active:
                proc = entry["proc"]
                ret = proc.poll()
                if ret is not None:
                    ticker = entry["ticker"]
                    key = tx_key(ticker, year)
                    duration = time.time() - entry["start_time"]
                    _drain_worker_stderr(entry)
                    worker_status = parse_worker_status(proc, ret, entry.get("_result_line"))
                    if worker_status == "failed":
                        errors_occurred = True
                    entry["stderr_file"].close()
                    if proc.stderr:
                        proc.stderr.close()
                    state["completed"][key] = {
                        "ticker": ticker, "duration_s": round(duration, 1),
                        "returncode": ret, "status": worker_status,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                    if key in state["in_progress"]:
                        del state["in_progress"][key]
                    state["stats"] = {"elapsed_s": round(time.time() - overall_start, 1),
                                      "total_tickers": total,
                                      "completed": len(state["completed"]),
                                      "running": len(state["in_progress"])}
                    save_state(state_path, state)

                    # Finalize processing -> final immediately
                    p_proc = output_path(ticker, year, agg, args.output, subdir="processing")
                    final_path = output_path(ticker, year, agg, args.output)
                    if p_proc.exists():
                        with open(p_proc) as pf:
                            has_rows = sum(1 for _ in pf) > 1
                        if has_rows:
                            final_path.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                p_proc.rename(final_path)
                            except OSError:
                                pass
                        elif p_proc.stat().st_size > 0:
                            no_data_dir = final_path.parent / "no_data"
                            no_data_dir.mkdir(parents=True, exist_ok=True)
                            try:
                                p_proc.rename(no_data_dir / p_proc.name)
                            except OSError:
                                pass

                    completed = len(state["completed"])
                    log("  [%d/%d] %s done (%s, %.1fs)" % (completed, total, ticker, worker_status, duration))
                else:
                    still_active.append(entry)
            active = still_active

            if active:
                now = time.time()
                if now - last_status_time >= 5:
                    for entry in active:
                        _drain_worker_stderr(entry)
                    last_status_time = now
                time.sleep(1)

    except KeyboardInterrupt:
        log("\n[INTERRUPT] Terminating workers ...")
        for entry in active:
            try:
                os.kill(entry["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for entry in active:
            entry["proc"].wait(timeout=5)
        log("All workers terminated.")
        sys.exit(130)

    state["in_progress"] = {}
    save_state(state_path, state)

    year_elapsed = time.time() - overall_start

    completed_ok = [k for k, v in state["completed"].items() if v.get("status") == "ok"]
    completed_no_data = [k for k, v in state["completed"].items() if v.get("status") == "no_data"]
    completed_fail = [k for k, v in state["completed"].items() if v.get("status") == "failed"]

    stats = {"total_tickers": total, "completed": len(state["completed"]),
             "successful": len(completed_ok), "no_data": len(completed_no_data),
             "failed": len(completed_fail), "elapsed_s": round(year_elapsed, 1)}
    state["stats"] = stats
    save_state(state_path, state)

    log("=" * 60)
    log("SUMMARY  [year %s]" % year)
    log("  Total:        %d" % total)
    log("  Successful:   %d" % len(completed_ok))
    log("  No data:      %d" % len(completed_no_data))
    log("  Failed:       %d" % len(completed_fail))
    log("  Duration:     %.1fs" % year_elapsed)
    log("  State file:   %s" % state_path)
    if completed_fail:
        log("  Failed keys:  %s" % ", ".join(completed_fail))
    log("=" * 60)

    if log_fh:
        log_fh.close()

    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    report = {
        "script": "options_chain_parallel_download",
        "timestamp": log_ts, "year": year, "workers": args.spawn,
        "resume": args.resume, "logs": args.logs,
        "total_tickers": total, "successful": len(completed_ok),
        "no_data": len(completed_no_data), "failed": len(completed_fail),
        "duration_s": round(year_elapsed, 1),
        "completed": state["completed"], "stats": stats,
    }
    report_dir = state_base / "options" / "chains"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"parallel_report_{year}_chains.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log("  Report:       %s" % report_path)


if __name__ == "__main__":
    main()
