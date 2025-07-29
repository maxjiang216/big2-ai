# Big 2 AI

A C++ implementation of the Big 2 card game (Shanghainese variant) with AI players and data collection capabilities for machine learning research.

## 🎯 Overview

This project implements a complete Big 2 game engine with:
- Full rule implementation of the game
- Multi-threaded game simulation
- Different AI agents (in progress)
- Feature extraction for ML training (used to craft agents)

## 🃏 Game Rules

Big 2 is a shedding-type card game where players try to be the first to play all their cards. This implementation follows the **Shanghainese variant** with these key features:

- **Modified deck**: 49 cards (standard 52-card deck minus three 2's and one ace)
- **2-player format**: Each player gets 16 cards, 17 cards remain unused
- **Card ranking**: 3 (lowest) → 4 → 5 → ... → K → A → 2 (highest)
- **Combinations**: Singles, doubles, triples, full houses, straights, sisters (consecutive doubles), triple straights, and bombs
- **Special rules**: Triple aces count as a bomb, bombs can "burn" other combinations

## 🏗️ Architecture

### Core Components

- **`Game`**: Internal game state management (hands, discards, legal moves)
- **`PartialGame`**: Player's limited view of the game state
- **`Move`**: Represents all possible card combinations with encoding/decoding
- **`Player`**: Abstract base class for AI implementations
- **`GameSimulator`**: Runs individual games between two players
- **`GameCoordinator`**: Orchestrates multi-threaded game simulations

### Feature Extraction

The system includes a pluggable feature extraction framework:
- **Game-level features**: One value per completed game (e.g., winner)
- **Turn-level features**: One value per turn (e.g., hand size)
- Exports to Apache Parquet format for ML workflows

## 🚀 Getting Started

### Prerequisites

- C++17 compatible compiler (GCC/Clang)
- Apache Arrow C++ libraries
- OpenMP (for parallel processing)

On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install build-essential libarrow-dev libparquet-dev
```

### Building

```bash
git clone <repository-url>
cd big2-ai/cpp
make
```

### Running Simulations

```bash
# Run N games with default settings
./big2-trainer

# The program will output:
# - game_features.parquet (game-level data)
# - turn_features.parquet (turn-level data)
```

### Analyzing Results

Use the included Python analysis script:

```bash
cd big2-ai/cpp
python3 analysis.py
```

This will generate:
- Statistical summaries
- Visualizations (game_analysis.png)
- CSV exports for further analysis
