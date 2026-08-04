"""
Parallel dispatcher for compute_chain_greeks.py.

Spawns N workers to compute IV and Greeks for options chain CSV files.
One ticker per worker at a time.

State file: data/options/chains/.parallel_state_<year>_<agg>_chains_greeks.json

Usage:
    python scripts/options/compute_chain_greeks_parallel_download.py --ohlcv_tickers --year 2025 --spawn 16
    python scripts/options/compute_chain_greeks_parallel_download.py --tickers AAPL --year 2025 --spawn 4
    python scripts/options/compute_chain_greeks_parallel_download.py --tickers_file /tmp/tickers.txt --year 2025 --spawn 16 --resume
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
WORKER_SCRIPT = SCRIPT_DIR / "compute_chain_greeks.py"
STDOUT_MARKER = "PARALLEL_RESULT:"
SCRIPT_NAME = Path(__file__).resolve().stem

AGGREGATE_MAP = {
    "1sec": "1sec", "1min": "1min", "5min": "5min",
    "15min": "15min", "1H": "1H", "4H": "4H", "1D": "1D",
}


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel compute IV and Greeks for options chains")
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--tickers_file", type=str, default=None)
    parser.add_argument("--ohlcv_tickers", action="store_true", default=False)
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument("--aggregate", choices=list(AGGREGATE_MAP.keys()), default="1min")
    parser.add_argument("--spawn", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logs", action="store_true", default=False)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--check", type=str, default=None)
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


def output_path(ticker: str, year: str, agg: str, output_dir: str | None = None) -> Path:
    folder = AGGREGATE_MAP[agg]
    base = Path(output_dir) if output_dir else Path("data")
    return base / "options" / "chains" / folder / year / f"{ticker}_{year}_{folder}_chains_greeks.csv"


def tx_key(ticker: str, year: str) -> str:
    return f"{ticker}_{year}"


def is_final(ticker: str, year: str, agg: str, output_dir: str | None = None) -> bool:
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


def fmt_duration(s: float) -> str:
    if s < 60:
        return "%.1fs" % s
    m = int(s // 60)
    sec = s % 60
    if m < 60:
        return "%dm %02.0fs" % (m, sec)
    h = m // 60
    m = m % 60
    return "%dh %02dm" % (h, m)


def main():
    args = parse_args()
    overall_start = time.time()
    year = args.year
    agg = args.aggregate
    folder = AGGREGATE_MAP[agg]

    state_base = Path(args.output) if args.output else Path("data")
    state_path = state_base / "options" / "chains" / f".parallel_state_{year}_{folder}_chains_greeks.json"
    state = load_state(state_path)

    if args.ohlcv_tickers:
        all_tickers = load_ohlcv_tickers(year)
    elif args.tickers_file:
        all_tickers = load_tickers(args.tickers_file)
    elif args.tickers:
        all_tickers = [clean_ticker(t) for t in args.tickers.split(",") if t.strip()]
    else:
        raise SystemExit("Error: specify --tickers, --tickers_file, or --ohlcv_tickers")

    pre_filtered = [t for t in all_tickers if not is_final(t, year, agg, args.output)]
    skipped_pre = len(all_tickers) - len(pre_filtered)
    all_tickers = pre_filtered

    if args.check:
        chk = [t for t in all_tickers
               if not (Path(args.check) / "options" / "chains" / folder / year
                       / f"{t}_{year}_{folder}_chains_greeks.csv").exists()]
        skipped_pre += len(all_tickers) - len(chk)
        all_tickers = chk

    state["all_tickers"] = all_tickers

    ticker_queue = [t for t in all_tickers
                    if tx_key(t, year) not in state.get("completed", {})
                    and not (args.resume and is_final(t, year, agg, args.output))
                    and tx_key(t, year) not in state.get("in_progress", {})]

    total = len(all_tickers)
    remaining = len(ticker_queue)
    completed_count = total - remaining + skipped_pre

    log_fh = None
    if args.logs:
        log_dir = state_base / "options" / "chains" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        log_path = log_dir / f"parallel_{year}_{folder}_greeks_{log_ts}.log"
        log_fh = open(log_path, "w")

    def log(msg: str, end: str = "\n"):
        print(msg, end=end, flush=True)
        if log_fh:
            log_fh.write(msg + ("\n" if end == "\n" else end))
            log_fh.flush()

    log("=" * 60)
    log("PARALLEL CHAIN GREEKS  [year %s]" % year)
    log("  Workers:      %d" % args.spawn)
    log("  Aggregate:    %s" % agg)
    log("  Resume:       %s" % args.resume)
    log("  Total:        %d tickers" % total)
    log("  Already done: %d" % completed_count)
    log("  Remaining:    %d" % remaining)
    log("=" * 60)

    if remaining == 0:
        log("Nothing to do.")
        if log_fh:
            log_fh.close()
        return

    log("")
    errors_occurred = False
    pending = list(ticker_queue)
    pending.reverse()
    active = []

    def spawn_worker(ticker: str) -> dict | None:
        nonlocal errors_occurred
        key = tx_key(ticker, year)
        cmd = [sys.executable, str(WORKER_SCRIPT),
               "--tickers", ticker, "--year", year,
               "--aggregate", agg, "--no_rename"]
        if args.resume:
            cmd.append("--resume")
        if args.output:
            cmd.extend(["--output", args.output])
        try:
            stderr_file = open(f"/tmp/worker_chain_greeks_{ticker}_{year}.log", "w")
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, text=True)
        except Exception as e:
            log("[ERROR] Failed to spawn worker for %s: %s" % (ticker, e))
            errors_occurred = True
            return None
        entry = {"ticker": ticker, "pid": proc.pid, "proc": proc,
                 "start_time": time.time(), "stderr_file": stderr_file,
                 "_result_line": None}
        state["in_progress"][key] = {"pid": proc.pid, "start_time": entry["start_time"]}
        state["stats"] = {"elapsed_s": round(time.time() - overall_start, 1),
                          "total_tickers": total, "completed": len(state["completed"]),
                          "running": len(state["in_progress"]), "workers": args.spawn}
        save_state(state_path, state)
        return entry

    try:
        while pending or active:
            while len(active) < args.spawn and pending:
                ticker = pending.pop()
                key = tx_key(ticker, year)
                if key in state.get("completed", {}) or key in state.get("in_progress", {}):
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
                    ws = parse_worker_status(proc, ret, entry.get("_result_line"))
                    if ws == "failed":
                        errors_occurred = True
                    entry["stderr_file"].close()
                    if proc.stderr:
                        proc.stderr.close()
                    state["completed"][key] = {
                        "ticker": ticker, "duration_s": round(duration, 1),
                        "returncode": ret, "status": ws,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                    if key in state["in_progress"]:
                        del state["in_progress"][key]
                    state["stats"] = {"elapsed_s": round(time.time() - overall_start, 1),
                                      "total_tickers": total,
                                      "completed": len(state["completed"]),
                                      "running": len(state["in_progress"]),
                                      "workers": args.spawn}
                    save_state(state_path, state)
                    log("  [%d/%d] %s done (%s, %.1fs)"
                        % (len(state["completed"]), total, ticker, ws, duration))
                else:
                    still_active.append(entry)
            active = still_active
            if active:
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
    completed_fail = [k for k, v in state["completed"].items() if v.get("status") == "failed"]

    s = {"total_tickers": total, "completed": len(state["completed"]),
         "successful": len(completed_ok), "failed": len(completed_fail),
         "elapsed_s": round(year_elapsed, 1)}
    state["stats"] = s
    save_state(state_path, state)

    log("=" * 60)
    log("SUMMARY  [year %s]" % year)
    log("  Total:        %d" % total)
    log("  Successful:   %d" % len(completed_ok))
    log("  Failed:       %d" % len(completed_fail))
    log("  Duration:     %s" % fmt_duration(year_elapsed))
    log("=" * 60)

    if log_fh:
        log_fh.close()

    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    report = {"script": "compute_chain_greeks_parallel", "timestamp": log_ts,
              "year": year, "workers": args.spawn, "resume": args.resume,
              "total_tickers": total, "successful": len(completed_ok),
              "failed": len(completed_fail), "duration_s": round(year_elapsed, 1),
              "completed": state["completed"], "stats": s}
    report_dir = state_base / "options" / "chains"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / f"parallel_report_{year}_{folder}_greeks.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
