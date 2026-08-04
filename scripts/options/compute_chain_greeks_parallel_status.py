"""
Status monitor for compute_chain_greeks_parallel_download.py.

Usage:
    python scripts/options/compute_chain_greeks_parallel_status.py --year 2025
    python scripts/options/compute_chain_greeks_parallel_status.py --year 2025 --watch
    python scripts/options/compute_chain_greeks_parallel_status.py --year 2025 --kill
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
    parser = argparse.ArgumentParser(
        description="Monitor chain greeks compute progress")
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument("--aggregate", choices=list(AGGREGATE_MAP.keys()), default="1min")
    parser.add_argument("--watch", action="store_true", default=False)
    parser.add_argument("--kill", action="store_true", default=False)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


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


def find_state_files(year: str, agg: str, output_dir: str | None = None) -> list[Path]:
    folder = AGGREGATE_MAP[agg]
    search_dirs = [Path(output_dir) if output_dir else Path("data") / "options" / "chains"]
    state_files = []
    for d in search_dirs:
        pattern = f".parallel_state_{year}_{folder}_chains_greeks.json"
        if d.exists():
            for p in d.glob(pattern):
                state_files.append(p)
    if not state_files:
        for d in search_dirs:
            for p in d.glob(".parallel_state_*_chains_greeks.json"):
                state_files.append(p)
    return sorted(state_files)


def show_status(state_path: Path) -> bool:
    if not state_path.exists():
        print("  (no state file)")
        return False
    with open(state_path) as f:
        state = json.load(f)

    all_tickers = state.get("all_tickers", [])
    completed = state.get("completed", {})
    in_progress = state.get("in_progress", {})
    stats = state.get("stats", {})

    total = len(all_tickers)
    done = len(completed)
    running = len(in_progress)
    remaining = total - done - running

    ok = sum(1 for v in completed.values() if v.get("status") == "ok")
    fail = sum(1 for v in completed.values() if v.get("status") == "failed")

    elapsed = stats.get("elapsed_s", 0)
    elapsed_hours = elapsed / 3600 if elapsed > 0 else 0.001
    pct = (done / total * 100) if total > 0 else 0
    throughput = done / elapsed_hours if elapsed_hours > 0 else 0
    eta = (remaining / (done / elapsed)) if done > 0 and elapsed > 0 else 0
    workers = stats.get("workers", 0)

    print("  State file:   %s" % state_path.name)
    print("  Total:        %d" % total)
    print("  Completed:    %d (%2.1f%%)" % (done, pct))
    print("    OK:         %d" % ok)
    print("    Failed:     %d" % fail)
    print("  Running:      %d" % running)
    print("  Remaining:    %d" % remaining)
    if workers:
        print("  Workers:      %d" % workers)
    print()
    print("  Elapsed:      %s" % fmt_duration(elapsed))
    if done > 0:
        print("  Throughput:   %.0f tickers/hr" % throughput)
        print("  Est. finish:  %s" % fmt_duration(eta))

    if in_progress:
        entries = sorted(in_progress.items(), key=lambda x: x[1].get("start_time", 0))
        max_display = 20
        display = entries[:max_display]
        extra = len(entries) - max_display if len(entries) > max_display else 0
        print()
        print("  -- Currently running (%d total%s) --"
              % (len(entries), "; showing first %d" % max_display if extra else ""))
        for key, info in display:
            start = info.get("start_time", 0)
            elapsed_running = time.time() - start
            ticker = key.rsplit("_", 1)[0]
            print("    %-12s pid=%-6d %s" % (ticker, info.get("pid", 0),
                                              fmt_duration(elapsed_running)))
    return running > 0


def kill_workers(state_path: Path):
    if not state_path.exists():
        print("No state file at %s" % state_path)
        return
    with open(state_path) as f:
        state = json.load(f)
    in_progress = state.get("in_progress", {})
    if not in_progress:
        print("No running workers found.")
        return
    pids = [(key.rsplit("_", 1)[0], info.get("pid"))
            for key, info in in_progress.items() if info.get("pid")]
    if not pids:
        print("No PIDs.")
        return
    print("Killing %d worker(s) ..." % len(pids))
    for ticker, pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            print("  Killed %s (pid %d)" % (ticker, pid))
        except ProcessLookupError:
            print("  %s (pid %d) already gone" % (ticker, pid))
        except PermissionError as e:
            print("  Cannot kill %s: %s" % (ticker, e))


def main():
    args = parse_args()
    state_files = find_state_files(args.year, args.aggregate, args.output)
    if not state_files:
        print("No matching state files found.")
        sys.exit(1)

    if args.kill:
        for sf in state_files:
            print()
            print("Killing for %s" % sf.name)
            print("-" * 50)
            kill_workers(sf)
        return

    for sf in state_files:
        print()
        label = sf.stem.replace(".parallel_state_", "state: ")
        print(label)
        print("-" * 50)
        show_status(sf)
        print()

    if args.watch:
        try:
            while True:
                time.sleep(5)
                print("\033[2J\033[H", end="")
                for sf in state_files:
                    label = sf.stem.replace(".parallel_state_", "state: ")
                    print(label)
                    print("-" * 50)
                    show_status(sf)
                    print()
        except KeyboardInterrupt:
            print()
            print("Watch stopped.")


if __name__ == "__main__":
    main()
