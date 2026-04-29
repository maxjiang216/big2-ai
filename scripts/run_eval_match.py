#!/usr/bin/env python3
"""
Run head-to-head evaluation (target/release/eval_match), save a log, print raw
output, then run analyze_eval_match on the same text.

Usage:
  python scripts/run_eval_match.py scripts/configs/eval_match.json
  python scripts/run_eval_match.py scripts/configs/eval_match.json --compile
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def compile_binary(verbose: bool = True) -> None:
    if verbose:
        print("Building eval_match binary (cargo build --release)...")
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "big2-datagen"],
        cwd=str(REPO_ROOT),
        capture_output=not verbose,
        text=True,
    )
    if result.returncode != 0:
        print("Build failed!", file=sys.stderr)
        if not verbose:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)
    if verbose:
        print("Build successful.\n")


def load_config(config_path: str) -> dict:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (REPO_ROOT / config_path).resolve()
    if not config_file.exists():
        print(f"Config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)
    required = ["p0", "p1", "deals"]
    missing = [k for k in required if k not in config]
    if missing:
        print(f"Config missing required fields: {missing}", file=sys.stderr)
        sys.exit(1)
    return config


def build_cmd(config: dict) -> list[str]:
    binary = REPO_ROOT / "target" / "release" / "eval_match"
    cmd = [
        str(binary),
        "--p0",
        config["p0"],
        "--p1",
        config["p1"],
        "--deals",
        str(config["deals"]),
    ]
    if config.get("p0_param") is not None:
        cmd.extend(["--p0-param", str(config["p0_param"])])
    if config.get("p1_param") is not None:
        cmd.extend(["--p1-param", str(config["p1_param"])])
    if "seed" in config:
        cmd.extend(["--seed", str(config["seed"])])
    if "threads" in config:
        cmd.extend(["--threads", str(config["threads"])])
    return cmd


def run_eval(config: dict) -> tuple[int, str]:
    cmd = build_cmd(config)
    print("Running eval_match...")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Merged stdout+stderr for one log (progress + summary)
    full_output = result.stdout
    return result.returncode, full_output


def resolve_output_log(config: dict) -> Path:
    raw = config.get("output_log", "analysis/eval_match_last.log")
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run eval_match from JSON config, log, and analyze",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/run_eval_match.py scripts/configs/eval_match.json
  python scripts/run_eval_match.py scripts/configs/eval_match.json --compile
        """,
    )
    parser.add_argument("config", help="Path to JSON configuration file")
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="Compile eval_match before running",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip compilation even if binary is missing",
    )
    args = parser.parse_args()

    binary = REPO_ROOT / "target" / "release" / "eval_match"
    if args.compile or (not binary.exists() and not args.no_compile):
        compile_binary()
    elif not binary.exists():
        print(f"Binary '{binary}' not found!", file=sys.stderr)
        print("  Run with --compile to build it first", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    log_path = resolve_output_log(config)

    code, full_output = run_eval(config)

    log_path.write_text(full_output, encoding="utf-8")
    print(full_output, end="", flush=True)
    print(f"\nWrote full log to: {log_path}\n", flush=True)

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from analyze_eval_match import analyze_text  # noqa: E402

    print("--- eval_match analysis ---", flush=True)
    sys.stdout.write(analyze_text(full_output))
    sys.stdout.flush()

    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
