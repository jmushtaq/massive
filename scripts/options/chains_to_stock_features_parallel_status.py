"""
Status monitor for chains_to_stock_features_parallel.py.

Reads the state file written by chains_to_stock_features_parallel.py.

Usage:
    python scripts/options/chains_to_stock_features_parallel_status.py --year 2014 --aggregate 1sec
    python scripts/options/chains_to_stock_features_parallel_status.py --year 2014 --aggregate 1sec --watch
    python scripts/options/chains_to_stock_features_parallel_status.py --year 2014 --aggregate 1sec --kill
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

AGGREGATE_MAP = {
    "1sec": "1sec", "1min": "1min", "5min": "5min",
    "15min": "15min", "1H": "1H", "4H": "4H", "1D": "1D",
}


def parse_args():
    p = argparse.ArgumentParser(description="Monitor chain -> stock features progress")
    p.add_argument("--year", type=str, required=True)
    p.add_argument("--aggregate", choices=list(AGGREGATE_MAP.keys()), default="1sec")
    p.add_argument("--watch", action="store_true", default=False)
    p.add_argument("--kill", action="store_true", default=False)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


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


def find_state_files(year, agg, odir=None):
    folder = AGGREGATE_MAP[agg]
    dirs = [Path(odir) if odir else Path("data") / "options" / "stocks"]
    files = []
    for d in dirs:
        for p in d.glob(f".parallel_state_{year}_{folder}_chainfeatures.json"):
            files.append(p)
    if not files:
        for d in dirs:
            for p in d.glob(".parallel_state_*_chainfeatures.json"):
                files.append(p)
    return sorted(files)


def show_status(sp):
    if not sp.exists():
        print("  (no state file)")
        return False
    state = json.loads(sp.read_text())
    all_t = state.get("all_tickers", [])
    completed = state.get("completed", {})
    in_prog = state.get("in_progress", {})
    stats = state.get("stats", {})

    total = len(all_t)
    done = len(completed)
    running = len(in_prog)
    remaining = total - done - running
    ok = sum(1 for v in completed.values() if v.get("status") == "ok")
    fail = sum(1 for v in completed.values() if v.get("status") == "failed")
    elapsed = stats.get("elapsed_s", 0)
    workers = stats.get("workers", 0)
    pct = (done / total * 100) if total else 0
    thr = done / (elapsed / 3600) if elapsed else 0
    eta = (remaining / (done / elapsed)) if done and elapsed else 0

    print(f"  State file:   {sp.name}")
    print(f"  Total:        {total}")
    print(f"  Completed:    {done} ({pct:.1f}%)")
    print(f"    OK:         {ok}")
    print(f"    Failed:     {fail}")
    print(f"  Running:      {running}")
    print(f"  Remaining:    {remaining}")
    if workers:
        print(f"  Workers:      {workers}")
    print()
    print(f"  Elapsed:      {fmt_duration(elapsed)}")
    if done:
        print(f"  Throughput:   {thr:.0f} tickers/hr")
        print(f"  Est. finish:  {fmt_duration(eta)}")

    if in_prog:
        entries = sorted(in_prog.items(), key=lambda x: x[1].get("start_time", 0))
        max_d = 20
        disp = entries[:max_d]
        extra = len(entries) - max_d if len(entries) > max_d else 0
        print()
        print(f"  -- Running ({len(entries)} total{' showing ' + str(max_d) if extra else ''}) --")
        for k, v in disp:
            ticker = k.rsplit("_", 1)[0]
            er = time.time() - v.get("start_time", 0)
            print(f"    {ticker:<12} pid={v.get('pid',0):<6} {fmt_duration(er)}")
    return running > 0


def kill_workers(sp):
    if not sp.exists():
        print(f"No state file at {sp}")
        return
    state = json.loads(sp.read_text())
    ip = state.get("in_progress", {})
    if not ip:
        print("No running workers.")
        return
    pids = [(k.rsplit("_", 1)[0], v.get("pid")) for k, v in ip.items() if v.get("pid")]
    print(f"Killing {len(pids)} worker(s) ...")
    for tkr, pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"  Killed {tkr} (pid {pid})")
        except ProcessLookupError:
            print(f"  {tkr} (pid {pid}) gone")
        except PermissionError as e:
            print(f"  Cannot kill {tkr}: {e}")


def main():
    args = parse_args()
    files = find_state_files(args.year, args.aggregate, args.output)
    if not files:
        print("No state files found.")
        sys.exit(1)
    if args.kill:
        for f in files:
            print(f"\nKilling for {f.name}\n{'='*50}")
            kill_workers(f)
        return
    for f in files:
        print(f"\n{f.stem.replace('.parallel_state_', 'state: ')}\n{'-'*50}")
        show_status(f)
        print()
    if args.watch:
        try:
            while True:
                time.sleep(5)
                print("\033[2J\033[H", end="")
                for f in files:
                    print(f"{f.stem.replace('.parallel_state_', 'state: ')}\n{'-'*50}")
                    show_status(f)
                    print()
        except KeyboardInterrupt:
            print("\nWatch stopped.")


if __name__ == "__main__":
    main()
