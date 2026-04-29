# Build from repository root (directory containing this Makefile).
CXX         = g++
CXXFLAGS    = -O2 -std=c++17 -Wall -Wextra -march=native -fopenmp
DEPFLAGS    = -MMD -MP
INCLUDES    = -Isrc/core -Isrc/simulation -Isrc/players -Isrc/datagen -Isrc/features -Itest
LDFLAGS     = -pthread
LDFLAGS_PARQUET = -lparquet -larrow -pthread

BUILD_DIR   = build
BIN_DIR     = bin

CORE_CPP    := $(wildcard src/core/*.cpp)
SIM_CPP     := src/simulation/game_simulator.cpp
COORD_CPP   := src/simulation/game_coordinator.cpp
DATAGEN_CPP := src/datagen/parquet_export.cpp src/datagen/samples_md.cpp

TEST_CPP    := $(wildcard test/*.cpp)

RESEARCH_BIN_NAMES := best_hand multi_comb play_probs count_turn_states
RESEARCH_BINS      := $(addprefix $(BUILD_DIR)/research/,$(RESEARCH_BIN_NAMES))

.PHONY: all clean dirs test_core coordinator generate_data eval_match pass_greedy_datagen move_agreement benchmark tablebase_opp1_gen research standards_init standards_configs help

all: help

help:
	@echo "Targets:"
	@echo "  make test_core            - unit tests (no Arrow)"
	@echo "  make benchmark            - perf benchmark binary (no Arrow)"
	@echo "  make coordinator          - coordinator + Parquet objects (needs libarrow)"
	@echo "  make generate_data        - self-play + Parquet + stats (needs libarrow)"
	@echo "  make eval_match           - head-to-head evaluation binary (no Arrow)"
	@echo "  make pass_greedy_datagen  - CSV dataset for pass-vs-greedy logistic (no Arrow)"
	@echo "  make move_agreement       - greedy vs tree move agreement stats (no Arrow)"
	@echo "  make tablebase_opp1_gen   - build opp-1-card tablebase binary generator (no Arrow)"
	@echo "  make research             - standalone research/*.cpp -> build/research/"
	@echo "  make standards_init       - git submodule: fetch projects/standard-linter"
	@echo "  make standards_configs    - copy configs from projects/standard-linter → .code-standards/ (for pre-commit)"
	@echo "  make clean"

# ------------------------------------------------------------------------------
# Shared code standards (vendored git submodule: github.com/maxjiang216/standard-linter)
# ------------------------------------------------------------------------------

standards_init:
	git submodule update --init projects/standard-linter

standards_configs: standards_init
	DEST="$(CURDIR)/.code-standards" bash projects/standard-linter/scripts/install-local-standards.sh

dirs:
	@mkdir -p $(BUILD_DIR)/src/core $(BUILD_DIR)/src/simulation $(BUILD_DIR)/src/datagen \
	          $(BUILD_DIR)/test $(BUILD_DIR)/research $(BIN_DIR)

# ============================================================================
# Object file rules
# ============================================================================

$(BUILD_DIR)/src/core/%.o: src/core/%.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/simulation/game_simulator.o: src/simulation/game_simulator.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/simulation/game_coordinator.o: src/simulation/game_coordinator.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/datagen/parquet_export.o: src/datagen/parquet_export.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/datagen/samples_md.o: src/datagen/samples_md.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/datagen/generate_data.o: src/datagen/generate_data.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test/%.o: test/%.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test/test_perf_bench.o: test/test_perf.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -DBENCHMARK_MAIN -c $< -o $@

# ============================================================================
# Common object lists
# ============================================================================

CORE_OBJS := $(patsubst src/core/%.cpp,$(BUILD_DIR)/src/core/%.o,$(CORE_CPP))

# test_core excludes test_perf.cpp (it has its own main under BENCHMARK_MAIN)
TEST_OBJS := $(CORE_OBJS) \
             $(BUILD_DIR)/src/simulation/game_simulator.o \
             $(patsubst test/%.cpp,$(BUILD_DIR)/test/%.o,$(TEST_CPP))

# ============================================================================
# bin/test_core — unit tests (no Arrow)
# ============================================================================

$(BIN_DIR)/test_core: $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/test_core"

test_core: dirs $(BIN_DIR)/test_core

# ============================================================================
# bin/benchmark — perf timing binary (no Arrow)
# ============================================================================

BENCH_OBJS := $(CORE_OBJS) \
              $(BUILD_DIR)/src/simulation/game_simulator.o \
              $(BUILD_DIR)/test/test_perf_bench.o

$(BIN_DIR)/benchmark: $(BENCH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/benchmark"

benchmark: dirs $(BIN_DIR)/benchmark

# ============================================================================
# coordinator — Parquet objects (compile check; needs libarrow-dev)
# ============================================================================

COORD_OBJS := $(CORE_OBJS) \
              $(BUILD_DIR)/src/simulation/game_simulator.o \
              $(BUILD_DIR)/src/simulation/game_coordinator.o \
              $(BUILD_DIR)/src/datagen/parquet_export.o

coordinator: dirs $(COORD_OBJS)
	@echo "✓ Coordinator + Parquet objects built (no main binary)."

# ============================================================================
# bin/generate_data — full Parquet pipeline (needs libarrow-dev)
# ============================================================================

GENDATA_OBJS := $(CORE_OBJS) \
                $(BUILD_DIR)/src/simulation/game_simulator.o \
                $(BUILD_DIR)/src/simulation/game_coordinator.o \
                $(BUILD_DIR)/src/datagen/parquet_export.o \
                $(BUILD_DIR)/src/datagen/samples_md.o \
                $(BUILD_DIR)/src/datagen/generate_data.o

$(BIN_DIR)/generate_data: $(GENDATA_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_PARQUET)
	@echo "✓ $(BIN_DIR)/generate_data"

generate_data: dirs $(BIN_DIR)/generate_data

# ============================================================================
# bin/eval_match — head-to-head evaluation (no Arrow)
# ============================================================================

# PIMC stats (dets saved, etc.) printed at end of eval_match when enabled.
$(BUILD_DIR)/src/datagen/eval_match.o: src/datagen/eval_match.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -DBIG2_PIMC_STATS=1 -c $< -o $@

EVALMATCH_OBJS := $(CORE_OBJS) \
                  $(BUILD_DIR)/src/simulation/game_simulator.o \
                  $(BUILD_DIR)/src/datagen/eval_match.o

$(BIN_DIR)/eval_match: $(EVALMATCH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/eval_match"

eval_match: dirs $(BIN_DIR)/eval_match

# ============================================================================
# bin/pass_greedy_datagen — pass vs greedy labels from PIMC self-play (no Arrow)
# ============================================================================

$(BUILD_DIR)/src/datagen/pass_greedy_datagen.o: src/datagen/pass_greedy_datagen.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

PASSGREEDY_OBJS := $(CORE_OBJS) \
                  $(BUILD_DIR)/src/simulation/game_simulator.o \
                  $(BUILD_DIR)/src/datagen/pass_greedy_datagen.o

$(BIN_DIR)/pass_greedy_datagen: $(PASSGREEDY_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/pass_greedy_datagen"

pass_greedy_datagen: dirs $(BIN_DIR)/pass_greedy_datagen

# ============================================================================
# bin/move_agreement — greedy vs tree move choice comparison (no Arrow)
# ============================================================================

$(BUILD_DIR)/src/datagen/move_agreement.o: src/datagen/move_agreement.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

MOVE_AGREEMENT_OBJS := $(CORE_OBJS) \
                       $(BUILD_DIR)/src/datagen/move_agreement.o

$(BIN_DIR)/move_agreement: $(MOVE_AGREEMENT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/move_agreement"

move_agreement: dirs $(BIN_DIR)/move_agreement

# ============================================================================
# bin/game_stats — game-length + legal-move distributions for any player pair
# ============================================================================

$(BUILD_DIR)/research/game_stats.o: research/game_stats.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

GAME_STATS_OBJS := $(CORE_OBJS) \
                   $(BUILD_DIR)/src/simulation/game_simulator.o \
                   $(BUILD_DIR)/src/core/game_record.o \
                   $(BUILD_DIR)/research/game_stats.o

$(BIN_DIR)/game_stats: $(GAME_STATS_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/game_stats"

game_stats: dirs $(BIN_DIR)/game_stats

# ============================================================================
# bin/legal_move_dist — empirical legal-move-count distribution (no Arrow)
# ============================================================================

$(BUILD_DIR)/research/legal_move_dist.o: research/legal_move_dist.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

LEGAL_MOVE_DIST_OBJS := $(CORE_OBJS) \
                        $(BUILD_DIR)/src/simulation/game_simulator.o \
                        $(BUILD_DIR)/research/legal_move_dist.o

$(BIN_DIR)/legal_move_dist: $(LEGAL_MOVE_DIST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/legal_move_dist"

legal_move_dist: dirs $(BIN_DIR)/legal_move_dist

# ============================================================================
# bin/tablebase_opp1_gen — precompute tablebase binary (no Arrow)
# ============================================================================

$(BIN_DIR)/tablebase_opp1_gen: scripts/tablebase_opp1_gen.cpp | dirs
	$(CXX) $(CXXFLAGS) -o $@ $< -Isrc/core
	@echo "✓ $(BIN_DIR)/tablebase_opp1_gen"

tablebase_opp1_gen: dirs $(BIN_DIR)/tablebase_opp1_gen

# ============================================================================
# build/research/* — standalone research programs (no Arrow)
# ============================================================================

$(BUILD_DIR)/research/%: research/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -o $@ $^

research: $(RESEARCH_BINS)
	@echo "✓ research binaries -> $(BUILD_DIR)/research/"

# ============================================================================

# Auto-generated header dependency files (-MMD -MP writes .d alongside .o).
-include $(shell find $(BUILD_DIR) -name '*.d' 2>/dev/null)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
