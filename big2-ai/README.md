# Big 2 AI - C++ Implementation

High-performance C++ engine for simulating Big 2 card games, generating training data, and evaluating player strategies.

## Project Structure

```
cpp/
├── core/               # Game fundamentals (stable, rarely modified)
├── datagen/           # Self-play data generation pipeline
├── players/           # Player strategy implementations
├── features/          # Feature extraction for ML training
├── scripts/           # Python workflow automation
├── analysis/          # Python analysis and visualization
├── bin/               # Compiled executables (gitignored)
├── build/             # Build artifacts (gitignored)
├── data/              # Generated datasets (gitignored)
└── Makefile           # Build system
```

## Directory Details

### `core/` - Game Engine

The foundational game logic. **Rarely needs modification** unless fixing bugs or adding game mechanics.

**Key files:**
- `game.h/cpp` - Full game state (used by simulator)
- `partial_game.h/cpp` - Player's view with hidden information
- `move.h/cpp` - Move representation and encoding (472 possible moves)
- `game_record.h/cpp` - Recording game history for training data
- `util.h/cpp` - Helper functions (rank conversion, move-to-cards mapping)

### `datagen/` - Data Generation

Self-play simulation and Parquet export pipeline.

**Key files:**
- `generate_data.cpp` - Main executable for data generation
- `game_coordinator.h/cpp` - Parallel batch simulation (handles 200k+ games)
- `game_simulator.h/cpp` - Single game execution
- `configs/` - JSON configuration files for different experiments

**Adding a new config:**
```json
{
  "player": "greedy",
  "num_games": 100000,
  "output_path": "data/greedy_100k",
  "game_features": ["outcome", "length"],
  "turn_features": ["turn_outcome", "player_hand_size"],
  "threads": 18,
  "seed": 42
}
```

### `players/` - Strategy Implementations

Each player type gets its own subdirectory.

**Structure:**
```
players/
├── player.h                    # Abstract base class
├── player_factory.h            # Factory pattern interface
├── random/
│   ├── random_player.h/cpp
│   └── random_player_factory.h/cpp
└── greedy/
    ├── greedy_player.h/cpp
    └── greedy_player_factory.h/cpp
```

**Adding a new player:**
1. Create `players/myplayer/myplayer_player.h/cpp`
2. Implement `Player` interface (see `player.h`)
3. Create corresponding `myplayer_player_factory.h/cpp`
4. Register in `datagen/generate_data.cpp` player registry:
   ```cpp
   if (name == "myplayer") {
     return std::make_shared<MyPlayerFactory>();
   }
   ```

### `features/` - Feature Extraction

Modular feature system for ML training data.

**Structure:**
```
features/
├── feature_extractor.h         # Base class
├── game_level/                 # One value per game
│   ├── outcome_feature.h       # Winner (0 or 1)
│   └── length_feature.h        # Number of turns
└── turn_level/                 # One value per turn
    ├── turn_outcome_feature.h  # Who wins from this state
    ├── player_hand_size_feature.h
    └── opponent_hand_size_feature.h
```

**Adding a new feature:**
1. Inherit from `FeatureExtractor`
2. Implement `name()` and either `gameExtract()` or `turnExtract()`
3. Register in `datagen/generate_data.cpp` feature registry:
   ```cpp
   if (name == "my_feature") return std::make_shared<MyFeature>();
   ```

### `scripts/` - Python Automation

Workflow helpers that call C++ binaries.

**Usage:**
```bash
# Generate training data (auto-compiles if needed)
python scripts/generate_data.py datagen/configs/greedy_selfplay.json

# Force recompilation
python scripts/generate_data.py datagen/configs/quick_test.json --compile
```

### `analysis/` - Data Analysis

Python scripts for analyzing generated data.

**Planned scripts:**
- `analyze_games.py` - Game-level statistics (win rates, game lengths)
- `analyze_turns.py` - Turn-level distributions
- `visualize.py` - Plotting utilities

## Quick Start

### 1. Build the binaries

```bash
# Build data generation binary
make datagen

# Or let Python handle it automatically
python scripts/generate_data.py datagen/configs/quick_test.json --compile
```

### 2. Generate training data

```bash
# Quick test (1k games)
python scripts/generate_data.py datagen/configs/quick_test.json

# Full run (100k games)
python scripts/generate_data.py datagen/configs/greedy_selfplay.json
```

### 3. Analyze results

```bash
python analysis/analyze_games.py data/greedy_100k_game.parquet
```

## Development Workflow

### Weekend Session Pattern

**Saturday Morning (2-3 hours):**
1. Review `DEVLOG.md` from last session
2. Run quick test to verify setup
3. Implement ONE feature/player/improvement
4. Update `DEVLOG.md` before stopping

**Example session:**
```bash
# 1. Quick sanity check
python scripts/generate_data.py datagen/configs/quick_test.json --compile

# 2. Make changes (e.g., add new feature)
# Edit features/turn_level/bomb_count_feature.h

# 3. Register feature in generate_data.cpp

# 4. Test
python scripts/generate_data.py datagen/configs/quick_test.json --compile

# 5. Generate full dataset
python scripts/generate_data.py datagen/configs/greedy_selfplay.json

# 6. Update DEVLOG.md with what you did and what's next
```

### Adding a New Player (Complete Example)

**1. Create player files:**

`players/mcts/mcts_player.h`:
```cpp
#ifndef MCTS_PLAYER_H
#define MCTS_PLAYER_H

#include "player.h"
#include "partial_game.h"

class MCTSPlayer : public Player {
public:
  MCTSPlayer(int iterations = 100);
  void accept_deal(std::array<int, 13> hand, int turn) override;
  void accept_opponent_move(const Move &move) override;
  Move select_move() override;

private:
  int iterations_;
  PartialGame game_;
  // MCTS tree state...
};

#endif
```

`players/mcts/mcts_player_factory.h`:
```cpp
#ifndef MCTS_PLAYER_FACTORY_H
#define MCTS_PLAYER_FACTORY_H

#include "player_factory.h"
#include "mcts_player.h"

class MCTSPlayerFactory : public PlayerFactory {
public:
  std::unique_ptr<Player> create_player() override {
    return std::make_unique<MCTSPlayer>(100);
  }
};

#endif
```

**2. Register in `datagen/generate_data.cpp`:**

```cpp
#include "players/mcts/mcts_player_factory.h"

std::shared_ptr<PlayerFactory> create_player_factory(const std::string& name) {
  if (name == "random") return std::make_shared<RandomPlayerFactory>();
  if (name == "greedy") return std::make_shared<GreedyPlayerFactory>();
  if (name == "mcts") return std::make_shared<MCTSPlayerFactory>();  // ADD THIS
  
  std::cerr << "Available players: random, greedy, mcts\n";  // UPDATE THIS
  return nullptr;
}
```

**3. Create config and test:**

```bash
# Create config
cat > datagen/configs/mcts_test.json << 'EOF'
{
  "player": "mcts",
  "num_games": 1000,
  "output_path": "data/mcts_test",
  "game_features": ["outcome"],
  "turn_features": ["player_hand_size"]
}
EOF

# Test it
python scripts/generate_data.py datagen/configs/mcts_test.json --compile
```

## Build System

The Makefile uses specific targets for different executables:

```bash
make datagen      # Build data generation binary
make eval         # Build evaluation binary (future)
make clean        # Remove all build artifacts
make help         # Show available targets
```

**Auto-discovery:** The Makefile automatically finds all `.cpp` files in `core/`, `datagen/`, `players/`, and `features/`. No manual updates needed when adding new files.

## Performance Notes

- **Batch size:** Data generation uses 200k games per batch to avoid OOM
- **Threading:** Defaults to `hardware_concurrency - 2` to leave CPU for other tasks
- **Memory:** ~16GB RAM recommended for 100k+ game runs
- **Speed:** ~10k games/second on modern hardware (greedy vs greedy)

## Troubleshooting

### Compilation fails
```bash
make clean
make datagen
```

### Binary not found
```bash
ls -lh bin/
# If empty, compile:
python scripts/generate_data.py <config> --compile
```

### Output files not created
Check that parent directories exist or let Python create them automatically (now handled by `scripts/generate_data.py`).

### Out of memory during large runs
Reduce batch size in `datagen/game_coordinator.cpp`:
```cpp
constexpr int BATCH_SIZE = 100'000;  // Down from 200k
```

## Next Steps

See `ROADMAP.md` for planned features and milestones.

**Immediate priorities:**
1. Add more turn-level features (bomb counts, legal move counts)
2. Implement `eval_players` binary for head-to-head testing
3. Create MCTS player baseline
4. Generate 1M+ game dataset for supervised learning