#!/usr/bin/env python3
"""
Run bin/generate_data and summarize built-in game-level features from Parquet.

This replaces the old standalone research programs `avg_game_length.cpp` and
`avg_moves.cpp`: game length is `length`, and the analogue of “average legal
moves in a random starting hand” is `start_legal_moves`.

Requires: `make generate_data` from the repository root (produces bin/generate_data).
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_GAME_FEATURES = "length,outcome,tb_hits,start_legal_moves"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_generate_data(
    *,
    bin_gd: Path,
    games: int,
    threads: int,
    seed: int | None,
    player: str,
    game_features: str,
    output_prefix: Path,
) -> None:
    cmd = [
        str(bin_gd),
        "--games",
        str(games),
        "--threads",
        str(threads),
        "--player",
        player,
        "--game-features",
        game_features,
        "--output",
        str(output_prefix),
        "--turn-features",
        "player_hand_size",
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    p = subprocess.run(
        cmd,
        cwd=str(repo_root()),
        text=True,
    )
    sys.stdout.flush()
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def summarize_parquet(path: Path) -> None:
    df = pd.read_parquet(path)
    if df.empty:
        print("No rows in Parquet.")
        return

    keys = sorted(df.columns)
    print("\n--- From game Parquet (per-game) ---")
    for k in keys:
        if k == "game_index":
            continue
        vals = df[k].tolist()
        if k == "outcome":
            c = Counter(vals)
            print(f"  {k}: " + ", ".join(f"P{w}:{c[w]}" for w in sorted(c)))
            continue
        arr = df[k].astype(float)
        print(
            f"  {k}: mean={arr.mean():.4f}  "
            f"stdev={arr.std(ddof=1) if len(arr) > 1 else 0:.4f}  "
            f"min={arr.min():.0f}  max={arr.max():.0f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=10_000, help="Self-play games")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default: random)")
    ap.add_argument("--player", default="random", choices=("random", "greedy"))
    ap.add_argument(
        "--game-features",
        default=DEFAULT_GAME_FEATURES,
        help=f"Comma-separated names (default: {DEFAULT_GAME_FEATURES})",
    )
    args = ap.parse_args()

    bin_gd = repo_root() / "bin" / "generate_data"
    if not bin_gd.is_file():
        print(
            f"Missing {bin_gd}. Build with:  make generate_data",
            file=sys.stderr,
        )
        raise SystemExit(1)

    with tempfile.TemporaryDirectory(dir=str(repo_root() / "research")) as tmp:
        prefix = Path(tmp) / "stats_run"
        run_generate_data(
            bin_gd=bin_gd,
            games=args.games,
            threads=args.threads,
            seed=args.seed,
            player=args.player,
            game_features=args.game_features,
            output_prefix=prefix,
        )
        game_parquet = Path(str(prefix) + "_game.parquet")
        summarize_parquet(game_parquet)


if __name__ == "__main__":
    main()
