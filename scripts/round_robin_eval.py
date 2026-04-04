#!/usr/bin/env python3
"""
Round-robin tournament: every pair of players runs bin/eval_match once with the
same paired-deal protocol as a single matchup (K deals → 2*K total games).

Config JSON:
  players            — required, list of player specs (see below)
  deals_per_matchup  — or "k": number of deals per pairwise matchup (required)
  seed               — optional base seed (default: random)
  threads            — optional, passed to each eval_match
  compile            — optional: set true to run make eval_match first (or use --compile)

Each player spec is either a string (player type name) or an object:
  { "type": "greedy_no_bomb", "param": 0.3, "label": "nb0.3" }
  type   — required (random, greedy, greedy_random, …)
  param  — optional float, default 0
  label  — optional short name for tables (default: type, or type(param))

Example:
  python scripts/round_robin_eval.py scripts/configs/round_robin_example.json
  python scripts/round_robin_eval.py scripts/configs/round_robin_example.json --compile
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Labels may contain parentheses (e.g. greedy_random(0.1)); do not use [^)]+ for the name.
WINS_RE = re.compile(
    r"P0\s+.+\s+wins:\s*(\d+)\s*/\s*(\d+)\s*\(([\d.]+)%\)",
    re.MULTILINE,
)


def compile_binary(verbose: bool = True) -> None:
    if verbose:
        print("Compiling eval_match binary...", flush=True)
    result = subprocess.run(
        ["make", "eval_match"],
        cwd=str(REPO_ROOT),
        capture_output=not verbose,
        text=True,
    )
    if result.returncode != 0:
        print("Compilation failed!", file=sys.stderr)
        if not verbose:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)
    if verbose:
        print("Compilation successful.\n", flush=True)


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        print(f"Config not found: {p}", file=sys.stderr)
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def normalize_players(raw: list) -> list[dict]:
    out: list[dict] = []
    for i, p in enumerate(raw):
        if isinstance(p, str):
            out.append({"type": p, "param": 0.0, "label": p, "_idx": i})
            continue
        if not isinstance(p, dict) or "type" not in p:
            print(f"Invalid player entry at index {i}: {p!r}", file=sys.stderr)
            sys.exit(1)
        t = str(p["type"])
        param = float(p.get("param", 0.0))
        label = p.get("label")
        if not label:
            label = t if param == 0.0 else f"{t}({param:g})"
        out.append({"type": t, "param": param, "label": str(label), "_idx": i})
    return out


def deals_from_config(cfg: dict) -> int:
    if "deals_per_matchup" in cfg:
        return int(cfg["deals_per_matchup"])
    if "k" in cfg:
        return int(cfg["k"])
    print(
        'Config must include "deals_per_matchup" or "k" (deals per pairwise matchup).',
        file=sys.stderr,
    )
    sys.exit(1)


def parse_eval_wins(text: str) -> tuple[int, int] | None:
    m = WINS_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def build_eval_cmd(
    p0: dict,
    p1: dict,
    deals: int,
    seed: int,
    threads: int | None,
) -> list[str]:
    cmd = [
        str(REPO_ROOT / "bin" / "eval_match"),
        "--p0",
        p0["type"],
        "--p1",
        p1["type"],
        "--deals",
        str(deals),
        "--seed",
        str(seed),
    ]
    cmd.extend(["--p0-param", str(p0["param"])])
    cmd.extend(["--p1-param", str(p1["param"])])
    if threads is not None:
        cmd.extend(["--threads", str(threads)])
    return cmd


def run_one_matchup(cmd: list[str], quiet: bool) -> tuple[int, str]:
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text = result.stdout
    if not quiet:
        print(text, end="", flush=True)
    return result.returncode, text


def format_matrix(
    labels: list[str],
    win_rate: list[list[float | None]],
) -> str:
    n = len(labels)
    w = max(len("vs") + 1, max(len(x) for x in labels) + 1, 8)
    lines = []
    header = " " * w + "".join(f"{labels[j][: w - 1]:>{w}}" for j in range(n))
    lines.append(header)
    for i in range(n):
        row = f"{labels[i][: w - 1]:<{w}}"
        for j in range(n):
            if i == j:
                cell = "—"
            else:
                r = win_rate[i][j]
                cell = f"{100.0 * r:.1f}%" if r is not None else "?"
            row += f"{cell:>{w}}"
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round-robin eval_match tournament from JSON config",
    )
    parser.add_argument("config", help="Path to JSON configuration")
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="Compile eval_match before running",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip auto-compile if binary missing",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print one line per matchup + final summary (hide eval_match spam)",
    )
    parser.add_argument(
        "--deals",
        type=int,
        default=None,
        metavar="N",
        help="Override deals_per_matchup / k from the config",
    )
    args = parser.parse_args()

    binary = REPO_ROOT / "bin" / "eval_match"
    if args.compile or (not binary.exists() and not args.no_compile):
        compile_binary()
    elif not binary.exists():
        print(f"Binary '{binary}' not found. Use --compile.", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    if "players" not in cfg or not isinstance(cfg["players"], list):
        print('Config must contain a non-empty "players" list.', file=sys.stderr)
        sys.exit(1)
    players = normalize_players(cfg["players"])
    n = len(players)
    if n < 2:
        print("Need at least 2 players for a round robin.", file=sys.stderr)
        sys.exit(1)

    deals = args.deals if args.deals is not None else deals_from_config(cfg)
    if deals <= 0:
        print("deals_per_matchup / k / --deals must be positive.", file=sys.stderr)
        sys.exit(1)
    base_seed = int(cfg["seed"]) if "seed" in cfg else None
    if base_seed is None:
        import random

        base_seed = random.randint(0, 2**31 - 1)
    threads = cfg.get("threads")
    if threads is not None:
        threads = int(threads)

    labels = [p["label"] for p in players]
    print(
        f"Round robin: {n} players, {deals} deals per matchup "
        f"({2 * deals} games per pair), base_seed={base_seed}\n",
        flush=True,
    )

    # wins_first[(i,j)] = (wins for player i, total games), i < j
    wins_first: dict[tuple[int, int], tuple[int, int]] = {}

    pair_index = 0
    total_pairs = n * (n - 1) // 2
    t0 = time.perf_counter()

    for i in range(n):
        for j in range(i + 1, n):
            pair_index += 1
            # Distinct seed per pair so matchups are independent but reproducible.
            seed = (base_seed + i * 100003 + j * 1000003) & 0xFFFFFFFF
            cmd = build_eval_cmd(players[i], players[j], deals, seed, threads)
            print(
                f"[{pair_index}/{total_pairs}] {labels[i]}  vs  {labels[j]}  "
                f"(seed={seed})",
                flush=True,
            )
            code, text = run_one_matchup(cmd, quiet=args.quiet)
            if code != 0:
                print(f"eval_match failed with code {code}", file=sys.stderr)
                sys.exit(code)
            parsed = parse_eval_wins(text)
            if parsed is None:
                print("Could not parse wins from eval_match output.", file=sys.stderr)
                sys.exit(1)
            w, tot = parsed
            wins_first[(i, j)] = (w, tot)
            pct = 100.0 * w / tot if tot else 0.0
            print(
                f"    → P0 ({labels[i]}) wins {w}/{tot} ({pct:.2f}%)\n",
                flush=True,
            )

    elapsed = time.perf_counter() - t0

    # Win rate matrix: rate[i][j] = fraction of games won by i vs j
    rate: list[list[float | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if i < j:
                w, t = wins_first[(i, j)]
                rate[i][j] = w / t if t else 0.0
                rate[j][i] = (t - w) / t if t else 0.0
            # j < i filled when processing (j, i)

    # Total wins across all games for each player
    total_wins = [0] * n
    total_games = [0] * n
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            w, t = wins_first[(i, j)]
            total_wins[i] += w
            total_wins[j] += t - w
            total_games[i] += t
            total_games[j] += t

    print("=" * 72)
    print("Win rate matrix (row vs column: row win % in that matchup)")
    print("=" * 72)
    print(format_matrix(labels, rate))
    print()
    print("Standings (by total wins, then win rate)")
    print("-" * 72)
    rows = []
    for i in range(n):
        tg = total_games[i]
        wr = total_wins[i] / tg if tg else 0.0
        rows.append((-total_wins[i], -wr, labels[i], total_wins[i], tg, wr))
    rows.sort()
    for rank, (_, _, lab, tw, tg, wr) in enumerate(rows, start=1):
        print(
            f"  {rank}. {lab:24}  wins {tw:6} / {tg:6}  ({100.0 * wr:.2f}%)",
        )
    print("-" * 72)
    print(f"Elapsed: {elapsed:.1f}s  ({total_pairs} matchups)\n")


if __name__ == "__main__":
    main()
