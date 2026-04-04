#!/usr/bin/env python3
"""
Compare pure greedy vs depth-15 tree greedy on the chosen "best" move at each
non–tablebase turn. Requires `make move_agreement` (binary: bin/move_agreement).

Default tree model is data/tree_model_d15.txt. The underlying game line is
advanced with --advance greedy (so positions match a greedy rollout) unless
you pass --advance tree.

Example:
    python analysis/move_agreement.py --games 500 --seed 0 --max-samples 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    root = _repo_root()
    exe = os.path.join(root, "bin", "move_agreement")
    if not os.path.isfile(exe):
        print("error: bin/move_agreement not found; run: make move_agreement", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description="Greedy vs tree move agreement (wrapper around bin/move_agreement).")
    p.add_argument("--games", type=int, required=True, help="Number of random deals to simulate.")
    p.add_argument("--seed", type=int, help="RNG seed (default: C++ random_device).")
    p.add_argument(
        "--tree-model",
        default="data/tree_model_d15.txt",
        help="Exported tree text file (default: data/tree_model_d15.txt).",
    )
    p.add_argument(
        "--advance",
        choices=("greedy", "tree"),
        default="greedy",
        help="Which policy steps the game after each comparison (default: greedy).",
    )
    p.add_argument("--max-samples", type=int, default=8, help="How many disagreement examples to print.")

    args = p.parse_args()
    tm = args.tree_model
    if not os.path.isabs(tm):
        tm = os.path.join(root, tm)

    cmd = [
        exe,
        "--games",
        str(args.games),
        "--tree-model",
        tm,
        "--advance",
        args.advance,
        "--max-samples",
        str(args.max_samples),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    return subprocess.call(cmd, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
