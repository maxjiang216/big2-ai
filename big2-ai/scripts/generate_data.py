#!/usr/bin/env python3
"""
Generate training data from self-play games.

Usage:
    python scripts/generate_data.py configs/greedy_selfplay.json
    python scripts/generate_data.py configs/greedy_selfplay.json --compile
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

def compile_binary(verbose=True):
    """Compile the generate_data binary."""
    if verbose:
        print("🔨 Compiling generate_data binary...")
    
    result = subprocess.run(
        ["make", "datagen"],  # Specific target
        capture_output=not verbose,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ Compilation failed!")
        if not verbose:
            print(result.stderr)
        sys.exit(1)
    
    if verbose:
        print("✓ Compilation successful\n")

def load_config(config_path):
    """Load and validate configuration file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Validate required fields
    required = ["player", "num_games", "output_path"]
    missing = [field for field in required if field not in config]
    
    if missing:
        print(f"❌ Config missing required fields: {missing}")
        sys.exit(1)
    
    return config

def run_datagen(config, binary_path="./bin/generate_data"):
    """Run the data generation binary with the given config."""
    
    # Create parent directories for output path
    output_path = Path(config["output_path"])
    
    # Check if the parent directory exists or can be created
    try:
        if output_path.parent != Path('.'):  # Only create if not current directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created output directory: {output_path.parent}")
    except PermissionError:
        print(f"❌ Permission denied: Cannot create directory {output_path.parent}")
        print(f"   Please check your permissions or choose a different output path")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error creating directory {output_path.parent}: {e}")
        sys.exit(1)
    
    # Build command-line arguments
    cmd = [
        binary_path,
        "--player", config["player"],
        "--games", str(config["num_games"]),
        "--output", config["output_path"]
    ]
    
    # Add optional arguments
    if "game_features" in config:
        cmd.extend(["--game-features", ",".join(config["game_features"])])
    
    if "turn_features" in config:
        cmd.extend(["--turn-features", ",".join(config["turn_features"])])
    
    if "seed" in config:
        cmd.extend(["--seed", str(config["seed"])])
    
    if "threads" in config:
        cmd.extend(["--threads", str(config["threads"])])
    
    print("🎮 Running data generation...")
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run the binary
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("\n❌ Data generation failed!")
        sys.exit(1)
    
    print("\n✓ Data generation complete!")
    
    # Print output files
    game_file = Path(config["output_path"] + "_game.parquet")
    turn_file = Path(config["output_path"] + "_turn.parquet")
    
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
  # Generate data using config file
  python scripts/generate_data.py configs/greedy_selfplay.json
  
  # Compile first, then generate
  python scripts/generate_data.py configs/greedy_selfplay.json --compile
  
  # Quick test with small dataset
  python scripts/generate_data.py configs/quick_test.json --compile
        """
    )
    
    parser.add_argument(
        "config",
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--compile", "-c",
        action="store_true",
        help="Compile the binary before running"
    )
    
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip compilation even if binary doesn't exist"
    )
    
    args = parser.parse_args()
    
    # Check if binary exists
    binary = Path("./bin/generate_data")
    
    if args.compile or (not binary.exists() and not args.no_compile):
        compile_binary()
    elif not binary.exists():
        print("❌ Binary './bin/generate_data' not found!")
        print("   Run with --compile to build it first")
        sys.exit(1)
    
    # Load config and run
    config = load_config(args.config)
    run_datagen(config)

if __name__ == "__main__":
    main()