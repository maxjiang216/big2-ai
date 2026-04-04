#!/usr/bin/env python3
"""
Parse stdout/stderr from bin/eval_match and print a short summary.

Designed to be piped after a match run, or given a log file:

  ./bin/eval_match ... 2>&1 | python3 scripts/analyze_eval_match.py
  python3 scripts/analyze_eval_match.py --file analysis/eval_match_last.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def analyze_text(text: str) -> str:
    lines = []
    title = None
    m = re.search(r"=== Eval Match:\s*(.+?)\s*===", text)
    if m:
        title = m.group(1).strip()
        lines.append(f"**Match:** {title}")

    # Name may contain ")" (e.g. greedy_random(0.1)); match from P0 through " wins:".
    m = re.search(
        r"P0\s+(.+?)\s+wins:\s*(\d+)\s*/\s*(\d+)\s*\(([\d.]+)%\)",
        text,
    )
    if m:
        name, wins, total, pct = m.group(1).strip(), m.group(2), m.group(3), m.group(4)
        lines.append(f"**P0 wins:** {wins} / {total} ({pct}%)  (`{name}`)")

    m = re.search(
        r"95% Wilson CI:\s*\[([\d.]+)%\s*,\s*([\d.]+)%\]",
        text,
    )
    if m:
        lines.append(f"**95% Wilson CI:** [{m.group(1)}%, {m.group(2)}%]")

    m = re.search(r"Result:\s*(.+?)(?:\n|$)", text)
    if m:
        lines.append(f"**Result:** {m.group(1).strip()}")

    m = re.search(r"Elapsed:\s*(.+?)(?:\n|$)", text)
    if m:
        lines.append(f"**Elapsed:** {m.group(1).strip()}")

    if not lines:
        return "(Could not parse eval_match output — paste full log.)\n"

    out = "\n".join(lines) + "\n"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval_match output")
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Read from file instead of stdin",
    )
    args = parser.parse_args()

    if args.file:
        text = args.file.read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    print("--- eval_match analysis ---")
    sys.stdout.write(analyze_text(text))


if __name__ == "__main__":
    main()
