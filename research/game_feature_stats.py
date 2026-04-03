#!/usr/bin/env python3
"""
Run the engine self-play binary and summarize built-in game-level features.

This replaces the old standalone research programs `avg_game_length.cpp` and
`avg_moves.cpp`: game length is `length`, and the analogue of “average legal
moves in a random starting hand” is `start_legal_moves` (count of legal moves
for the opening player on turn 0, using the same rules as the core `Game`).

Requires: `make selfplay` from the repository root (produces `bin/selfplay`).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


DEFAULT_GAME_FEATURES = "length,outcome,tb_hits,start_legal_moves"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_selfplay(
    *,
    bin_selfplay: Path,
    games: int,
    threads: int,
    seed: int | None,
    player: str,
    game_features: str,
    game_jsonl: Path,
) -> None:
    cmd = [
        str(bin_selfplay),
        "--games",
        str(games),
        "--threads",
        str(threads),
        "--player",
        player,
        "--game-features",
        game_features,
        "--game-jsonl",
        str(game_jsonl),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    env = os.environ.copy()
    p = subprocess.run(
        cmd,
        cwd=repo_root(),
        env=env,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(p.stdout)
    if p.stderr:
        sys.stderr.write(p.stderr)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def summarize_jsonl(path: Path) -> None:
    rows: list[dict[str, int]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print("No rows in JSONL.")
        return

    keys = sorted(rows[0].keys())
    print("\n--- From --game-jsonl (per-game) ---")
    for k in keys:
        vals = [r[k] for r in rows]
        if k == "outcome":
            c = Counter(vals)
            print(f"  {k}: " + ", ".join(f"{w}:{c[w]}" for w in sorted(c)))
            continue
        print(
            f"  {k}: mean={statistics.mean(vals):.4f}  "
            f"stdev={statistics.stdev(vals) if len(vals) > 1 else 0:.4f}  "
            f"min={min(vals)}  max={max(vals)}"
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
    ap.add_argument(
        "--jsonl-out",
        type=Path,
        default=None,
        help="Write per-game JSONL here (default: temp file, deleted after)",
    )
    ap.add_argument(
        "--keep-jsonl",
        action="store_true",
        help="Keep the JSONL when using a temp path",
    )
    args = ap.parse_args()

    bin_selfplay = repo_root() / "bin" / "selfplay"
    if not bin_selfplay.is_file():
        print(
            f"Missing {bin_selfplay}. Build with:  make selfplay",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.jsonl_out is not None:
        game_jsonl = args.jsonl_out
        game_jsonl.parent.mkdir(parents=True, exist_ok=True)
        use_temp = False
    else:
        research_dir = repo_root() / "research"
        research_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", dir=research_dir)
        os.close(fd)
        game_jsonl = Path(tmp_path)
        use_temp = True

    try:
        run_selfplay(
            bin_selfplay=bin_selfplay,
            games=args.games,
            threads=args.threads,
            seed=args.seed,
            player=args.player,
            game_features=args.game_features,
            game_jsonl=game_jsonl,
        )
        summarize_jsonl(game_jsonl)
    finally:
        if use_temp and not args.keep_jsonl and game_jsonl.is_file():
            game_jsonl.unlink()


if __name__ == "__main__":
    main()
