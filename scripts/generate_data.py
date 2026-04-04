#!/usr/bin/env python3
"""
Generate training data from self-play games (bin/generate_data → Parquet).

Run from anywhere; paths are resolved relative to the repository root (parent of
this scripts/ directory).

Usage:
    python scripts/generate_data.py scripts/configs/greedy.json
    python scripts/generate_data.py scripts/configs/greedy.json --compile
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def compile_binary(verbose=True):
    """Compile the generate_data binary (make target: generate_data)."""
    if verbose:
        print("Compiling generate_data binary...")

    result = subprocess.run(
        ["make", "generate_data"],
        cwd=str(REPO_ROOT),
        capture_output=not verbose,
        text=True,
    )

    if result.returncode != 0:
        print("Compilation failed!")
        if not verbose:
            print(result.stderr)
        sys.exit(1)

    if verbose:
        print("Compilation successful.\n")


def load_config(config_path):
    """Load and validate configuration file."""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (REPO_ROOT / config_path).resolve()

    if not config_file.exists():
        print(f"Config file not found: {config_file}")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    required = ["player", "num_games"]
    missing = [field for field in required if field not in config]

    if missing:
        print(f"Config missing required fields: {missing}")
        sys.exit(1)

    return config


def run_datagen(config, binary_path=None):
    """Run the data generation binary with the given config."""
    if binary_path is None:
        binary_path = REPO_ROOT / "bin" / "generate_data"
    else:
        binary_path = Path(binary_path)

    cmd = [
        str(binary_path),
        "--player",
        config["player"],
        "--games",
        str(config["num_games"]),
    ]

    out_raw = config.get("output_path")
    if out_raw:
        outp = Path(out_raw)
        try:
            if outp.is_absolute():
                outp.parent.mkdir(parents=True, exist_ok=True)
                print(f"Ensured output directory: {outp.parent}")
            else:
                (REPO_ROOT / outp.parent).mkdir(parents=True, exist_ok=True)
                print(f"Ensured output directory: {REPO_ROOT / outp.parent}")
        except PermissionError:
            print("Permission denied: Cannot create output directory")
            sys.exit(1)
        except Exception as e:
            print(f"Error creating output directory: {e}")
            sys.exit(1)
        cmd.extend(["--output", out_raw])

    if "game_features" in config:
        cmd.extend(["--game-features", ",".join(config["game_features"])])

    if "turn_features" in config:
        cmd.extend(["--turn-features", ",".join(config["turn_features"])])

    if "seed" in config:
        cmd.extend(["--seed", str(config["seed"])])

    if "threads" in config:
        cmd.extend(["--threads", str(config["threads"])])

    if "samples_md_path" in config and config["samples_md_path"]:
        p = Path(config["samples_md_path"])
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--samples-md", str(p)])

    if out_raw and "game_features" not in config and "turn_features" not in config:
        print(
            "Error: output_path requires game_features and/or turn_features in config.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Running data generation...")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))

    if result.returncode != 0:
        print("\nData generation failed!")
        sys.exit(1)

    print("\nData generation complete.")

    if not out_raw:
        return

    raw = out_raw
    outp = Path(raw)

    def parquet_file(suffix: str) -> Path:
        p = f"{raw}_{suffix}.parquet"
        return Path(p) if outp.is_absolute() else REPO_ROOT / p

    game_file = parquet_file("game")
    turn_file = parquet_file("turn")

    if game_file.exists():
        size_mb = game_file.stat().st_size / (1024 * 1024)
        print(f"  Game data: {game_file} ({size_mb:.1f} MB)")

    if turn_file.exists():
        size_mb = turn_file.stat().st_size / (1024 * 1024)
        print(f"  Turn data: {turn_file} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from self-play games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_data.py scripts/configs/greedy.json
  python scripts/generate_data.py scripts/configs/greedy.json --compile
  cd <repo-root> && python scripts/generate_data.py scripts/configs/test.json
        """,
    )

    parser.add_argument("config", help="Path to JSON configuration file")
    parser.add_argument(
        "--compile",
        "-c",
        action="store_true",
        help="Compile the binary before running",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip compilation even if binary doesn't exist",
    )

    args = parser.parse_args()

    binary = REPO_ROOT / "bin" / "generate_data"

    if args.compile or (not binary.exists() and not args.no_compile):
        compile_binary()
    elif not binary.exists():
        print(f"Binary '{binary}' not found!")
        print("   Run with --compile to build it first")
        sys.exit(1)

    config = load_config(args.config)
    run_datagen(config)


if __name__ == "__main__":
    main()
