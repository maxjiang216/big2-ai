#!/usr/bin/env python3
"""
Measure wall-clock time for self-play (same player vs itself) with pure greedy
vs tree_greedy at d10 and d15 (default). Uses bin/eval_match.

eval_match runs 2 * --deals games (paired deals). Reported "games/s" uses that
total game count.

Example:
    python analysis/selfplay_speed.py --deals 2000 --seed 0 --threads 1
    make eval_match && python analysis/selfplay_speed.py --compile --deals 500
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class RunResult:
    label: str
    elapsed_s: float
    total_games: int
    games_per_s: float
    stdout_tail: str


_ELAPSED_RE = re.compile(
    r"Elapsed:\s+(.+?)\s+\(\s*(\d+)\s+games/s\s*\)",
    re.MULTILINE,
)


def _run_eval_match(
    root: str,
    exe: str,
    p0: str,
    p1: str,
    p0_param: Optional[float],
    p1_param: Optional[float],
    deals: int,
    seed: int,
    threads: int,
) -> Tuple[RunResult, subprocess.CompletedProcess]:
    cmd = [
        exe,
        "--p0",
        p0,
        "--p1",
        p1,
        "--deals",
        str(deals),
        "--seed",
        str(seed),
        "--threads",
        str(threads),
    ]
    if p0_param is not None:
        cmd.extend(["--p0-param", str(p0_param)])
    if p1_param is not None:
        cmd.extend(["--p1-param", str(p1_param)])

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
    )
    t1 = time.perf_counter()
    elapsed_s = t1 - t0
    out = proc.stdout or ""
    total_games = 2 * deals
    gps = total_games / elapsed_s if elapsed_s > 0 else 0.0

    m = _ELAPSED_RE.search(out)
    if m:
        # Cross-check: eval_match prints integer games/s; wall clock is authoritative.
        _ = m.group(2)

    return (
        RunResult(
            label="",
            elapsed_s=elapsed_s,
            total_games=total_games,
            games_per_s=gps,
            stdout_tail="\n".join(out.strip().splitlines()[-6:]),
        ),
        proc,
    )


def _compile_eval_match(root: str) -> None:
    r = subprocess.run(["cargo", "build", "--release", "-p", "big2-datagen"], cwd=root)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> int:
    root = _repo_root()
    exe = os.path.join(root, "target", "release", "eval_match")

    ap = argparse.ArgumentParser(
        description="Compare self-play speed: greedy vs tree_greedy (d10, d15 by default).",
    )
    ap.add_argument(
        "--deals",
        type=int,
        required=True,
        help="Unique deals (total games = 2 * deals).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed (default: 0).")
    ap.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Worker threads for eval_match (default: 1 for stable relative timings).",
    )
    ap.add_argument(
        "--depths",
        type=int,
        nargs="*",
        default=[10, 15],
        help="Tree max_depth values to test (loads data/tree_model_d{N}.txt). Default: 10 15.",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="Run cargo build --release before timing.",
    )
    ap.add_argument(
        "--show-eval-output",
        action="store_true",
        help="Print last lines of eval_match stdout for each run.",
    )
    args = ap.parse_args()

    if args.compile:
        _compile_eval_match(root)

    if not os.path.isfile(exe):
        print(
            "error: target/release/eval_match not found; run: cargo build --release -p big2-datagen",
            file=sys.stderr,
        )
        return 1

    if args.deals <= 0:
        print("error: --deals must be positive", file=sys.stderr)
        return 1

    configs: List[Tuple[str, str, Optional[float], Optional[float]]] = [
        ("greedy", "greedy", None, None),
    ]
    for d in args.depths:
        configs.append(("tree_greedy", "tree_greedy", float(d), float(d)))

    results: List[RunResult] = []
    for p0, p1, p0p, p1p in configs:
        label = "greedy" if p0 == "greedy" else f"tree_d{p0p:.0f}"
        rr, proc = _run_eval_match(
            root, exe, p0, p1, p0p, p1p, args.deals, args.seed, args.threads
        )
        rr.label = label
        results.append(rr)
        if proc.returncode != 0:
            print(
                f"eval_match failed ({label}), exit {proc.returncode}", file=sys.stderr
            )
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            return proc.returncode or 1
        if args.show_eval_output:
            print(f"--- {label} ---\n{rr.stdout_tail}\n")

    baseline_s = results[0].elapsed_s
    print("=== self-play speed (eval_match) ===")
    print(
        f"deals={args.deals}  total_games={results[0].total_games}  seed={args.seed}  threads={args.threads}"
    )
    print()
    print(f"{'model':<12} {'seconds':>10} {'games/s':>12} {'vs greedy':>12}")
    print("-" * 48)
    for rr in results:
        vs = "1.00x" if rr.label == "greedy" else f"{rr.elapsed_s / baseline_s:.2f}x"
        print(f"{rr.label:<12} {rr.elapsed_s:10.3f} {rr.games_per_s:12.1f} {vs:>12}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
