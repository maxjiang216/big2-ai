#!/usr/bin/env python3
"""
Compare fixed-hand vs per-step resampled PIMC rollouts (``pimc`` vs ``pimc_redet``).

Runs ``bin/eval_match`` for a fixed matrix of matchups with the same ``--deals``,
``--seed``, and ``--threads`` so win rates and wall time are comparable.

Example:
    uv run python scripts/compare_pimc_rollout_strategies.py --deals 2000 --seed 42
    uv run python scripts/compare_pimc_rollout_strategies.py --deals 500 --compile
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_eval_output(text: str) -> dict[str, str | float | None]:
    out: dict[str, str | float | None] = {
        "p0_wins": None,
        "total_games": None,
        "p0_pct": None,
        "wilson_lo": None,
        "wilson_hi": None,
        "deals_p0_sweep": None,
        "deals_split": None,
        "deals_p1_sweep": None,
        "deals_total": None,
        "decisive_p0_sweep": None,
        "decisive_n": None,
        "decisive_pct": None,
        "decisive_wilson_lo": None,
        "decisive_wilson_hi": None,
        "elapsed_line": None,
        "wall_sec": None,
        "games_per_s": None,
    }
    m = re.search(
        r"P0\s+(.+?)\s+wins:\s*(\d+)\s*/\s*(\d+)\s*\(([\d.]+)%\)",
        text,
    )
    if m:
        out["p0_label"] = m.group(1).strip()
        out["p0_wins"] = int(m.group(2))
        out["total_games"] = int(m.group(3))
        out["p0_pct"] = float(m.group(4))

    # First Wilson line is per-game CI; decisive CI is parsed separately below.
    m = re.search(
        r"95% Wilson CI:\s*\[([\d.]+)%\s*,\s*([\d.]+)%\]",
        text,
    )
    if m:
        out["wilson_lo"] = float(m.group(1))
        out["wilson_hi"] = float(m.group(2))

    m = re.search(
        r"Deals: P0_sweep=(\d+) split=(\d+) P1_sweep=(\d+) total=(\d+)",
        text,
    )
    if m:
        out["deals_p0_sweep"] = int(m.group(1))
        out["deals_split"] = int(m.group(2))
        out["deals_p1_sweep"] = int(m.group(3))
        out["deals_total"] = int(m.group(4))

    m = re.search(
        r"Among decisive deals \(non-split\): P0 sweep (\d+) / (\d+) "
        r"\(([\d.]+)%\)\s+95% Wilson CI: \[([\d.]+)%,\s*([\d.]+)%\]",
        text,
    )
    if m:
        out["decisive_p0_sweep"] = int(m.group(1))
        out["decisive_n"] = int(m.group(2))
        out["decisive_pct"] = float(m.group(3))
        out["decisive_wilson_lo"] = float(m.group(4))
        out["decisive_wilson_hi"] = float(m.group(5))

    m = re.search(r"Elapsed:\s*(.+?)(?:\n|$)", text)
    if m:
        line = m.group(1).strip()
        out["elapsed_line"] = line
        g = re.search(r"\(([\d.]+)\s*games/s\)", line)
        if g:
            out["games_per_s"] = float(g.group(1))
        if "ms" in line:
            mm = re.search(r"([\d.]+)\s*ms", line)
            if mm:
                out["wall_sec"] = float(mm.group(1)) / 1000.0
        elif "m" in line and "s" in line:
            mm = re.search(r"(\d+)m\s*(\d+)s", line)
            if mm:
                out["wall_sec"] = int(mm.group(1)) * 60.0 + int(mm.group(2))
        else:
            mm = re.search(r"([\d.]+)\s*s\b", line)
            if mm:
                out["wall_sec"] = float(mm.group(1))

    return out


def run_one(
    exe: Path,
    p0: str,
    p1: str,
    p0_param: float,
    p1_param: float,
    deals: int,
    seed: int,
    threads: int,
) -> tuple[str, float]:
    cmd = [
        str(exe),
        "--p0",
        p0,
        "--p1",
        p1,
        "--deals",
        str(deals),
        "--p0-param",
        str(p0_param),
        "--p1-param",
        str(p1_param),
        "--seed",
        str(seed),
        "--threads",
        str(threads),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wall = time.perf_counter() - t0
    out = proc.stdout
    if proc.returncode != 0:
        print(f"FAILED ({proc.returncode}): {' '.join(cmd)}\n{out}", file=sys.stderr)
        sys.exit(proc.returncode)
    return out, wall


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pimc vs pimc_redet vs random/greedy (eval_match matrix)"
    )
    parser.add_argument("--deals", type=int, default=2000, help="Unique deals (games = 2*deals)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--pimc-n",
        type=int,
        default=20,
        help="Determinizations for pimc / pimc_redet (param)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Run `make eval_match` before benchmarking",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write CSV summary (repo-relative or absolute)",
    )
    args = parser.parse_args()

    n = float(args.pimc_n)

    if args.compile:
        subprocess.run(["make", "eval_match"], cwd=str(REPO_ROOT), check=True)

    exe = REPO_ROOT / "bin" / "eval_match"
    if not exe.is_file() or not os.access(exe, os.X_OK):
        print(f"Missing or not executable: {exe}; run: make eval_match", file=sys.stderr)
        sys.exit(1)

    # (label, p0, p1, p0_param, p1_param)
    matrix: list[tuple[str, str, str, float, float]] = [
        ("greedy vs random", "greedy", "random", 0.0, 0.0),
        (f"pimc({args.pimc_n}) vs random", "pimc", "random", n, 0.0),
        (f"pimc_redet({args.pimc_n}) vs random", "pimc_redet", "random", n, 0.0),
        (f"pimc({args.pimc_n}) vs greedy", "pimc", "greedy", n, 0.0),
        (f"pimc_redet({args.pimc_n}) vs greedy", "pimc_redet", "greedy", n, 0.0),
        (
            f"pimc({args.pimc_n}) vs pimc_redet({args.pimc_n})",
            "pimc",
            "pimc_redet",
            n,
            n,
        ),
    ]

    rows: list[dict[str, object]] = []

    print(
        f"deals={args.deals} (total games={2 * args.deals})  "
        f"seed={args.seed}  threads={args.threads}\n"
    )

    for label, p0, p1, p0p, p1p in matrix:
        text, wall_proc = run_one(
            exe, p0, p1, p0p, p1p, args.deals, args.seed, args.threads
        )
        parsed = parse_eval_output(text)
        parsed["matchup"] = label
        parsed["wall_sec_process"] = round(wall_proc, 3)
        rows.append(parsed)

        gp = parsed.get("games_per_s")
        pct = parsed.get("p0_pct")
        ci = ""
        if parsed.get("wilson_lo") is not None:
            ci = f" [{parsed['wilson_lo']:.1f}–{parsed['wilson_hi']:.1f}%]"
        deal_bits = ""
        if parsed.get("deals_total") is not None:
            deal_bits = (
                f"  deals: P0_sweep={parsed['deals_p0_sweep']} "
                f"split={parsed['deals_split']} P1_sweep={parsed['deals_p1_sweep']}"
            )
            if parsed.get("decisive_n") is not None:
                dci = ""
                if parsed.get("decisive_wilson_lo") is not None:
                    dci = (
                        f" [{parsed['decisive_wilson_lo']:.1f}–"
                        f"{parsed['decisive_wilson_hi']:.1f}%]"
                    )
                deal_bits += (
                    f"  decisive: {parsed['decisive_p0_sweep']}/{parsed['decisive_n']}"
                    f" ({parsed['decisive_pct']:.2f}%){dci}"
                )
        print(
            f"{label}\n"
            f"  P0 win: {pct}%{ci}{deal_bits}   "
            f"eval_match: {parsed.get('elapsed_line')}   "
            f"process wall: {wall_proc:.2f}s\n"
        )

    if args.csv:
        p = args.csv
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "matchup",
            "p0_pct",
            "wilson_lo",
            "wilson_hi",
            "deals_p0_sweep",
            "deals_split",
            "deals_p1_sweep",
            "deals_total",
            "decisive_p0_sweep",
            "decisive_n",
            "decisive_pct",
            "decisive_wilson_lo",
            "decisive_wilson_hi",
            "games_per_s",
            "wall_sec",
            "wall_sec_process",
            "elapsed_line",
        ]
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
