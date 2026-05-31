# Build from repository root (directory containing this Makefile).
comma := ,
CXX         = g++
CXXFLAGS    = -O2 -std=c++17 -Wall -Wextra -march=native -fopenmp
DEPFLAGS    = -MMD -MP
INCLUDES    = -Isrc/core -Isrc/simulation -Isrc/players -Isrc/datagen -Isrc/features -Itest
LDFLAGS     = -pthread
LDFLAGS_PARQUET = -lparquet -larrow -pthread

# LibTorch (from project venv) — used only for generate_nn_selfplay
TORCH_BASE      := $(shell .venv/bin/python -c "import torch,os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
TORCH_INCLUDES  := -I$(TORCH_BASE)/include -I$(TORCH_BASE)/include/torch/csrc/api/include
CUDA_LIB_BASE   := $(shell .venv/bin/python -c "import os; p=os.path.abspath('.venv/lib/python3.13/site-packages/nvidia/cu13/lib'); print(p) if os.path.isdir(p) else print('')" 2>/dev/null)
TORCH_LDFLAGS   := -L$(TORCH_BASE)/lib -ltorch -ltorch_cpu -lc10 \
                   -Wl,-rpath,$(TORCH_BASE)/lib \
                   $(if $(CUDA_LIB_BASE),-Wl$(comma)-rpath$(comma)$(CUDA_LIB_BASE))
LDFLAGS_TORCH   := $(LDFLAGS_PARQUET) $(TORCH_LDFLAGS)

BUILD_DIR   = build
BIN_DIR     = bin

CORE_CPP    := $(wildcard src/core/*.cpp)
SIM_CPP     := src/simulation/game_simulator.cpp
COORD_CPP   := src/simulation/game_coordinator.cpp
DATAGEN_CPP := src/datagen/parquet_export.cpp src/datagen/samples_md.cpp

TEST_CPP    := $(wildcard test/*.cpp)

RESEARCH_BIN_NAMES := best_hand multi_comb play_probs count_turn_states
RESEARCH_BINS      := $(addprefix $(BUILD_DIR)/research/,$(RESEARCH_BIN_NAMES))

.PHONY: all clean dirs test_core coordinator generate_data generate_nn_data eval_match eval_nn_match eval_nn_vs_classic play_games pass_greedy_datagen move_agreement benchmark tablebase_opp1_gen move_audit az_compose_gen az_vs_teacher_agree az_nn_check az_search_smoke az_play_check az_selfplay eval_az_match research help

all: help

help:
	@echo "  make generate_nn_selfplay - NN self-play data gen (needs libarrow + LibTorch)"
	@echo "  make eval_nn_match        - head-to-head NN model eval (needs LibTorch)"
	@echo "  make eval_nn_vs_classic   - NN vs classic player eval (needs LibTorch)"
	@echo "  make play_games           - play games + print per-turn decisions (needs LibTorch)"
	@echo "Targets:"
	@echo "  make test_core            - unit tests (no Arrow)"
	@echo "  make benchmark            - perf benchmark binary (no Arrow)"
	@echo "  make coordinator          - coordinator + Parquet objects (needs libarrow)"
	@echo "  make generate_data        - self-play + Parquet + stats (needs libarrow)"
	@echo "  make generate_nn_data     - NN training data (random self-play, needs libarrow)"
	@echo "  make eval_match           - head-to-head evaluation binary (no Arrow)"
	@echo "  scripts/profile_pimc_selfplay.sh - PIMC(20) symmetric eval_match: perf/callgrind/plain"
	@echo "  make pass_greedy_datagen  - CSV dataset for pass-vs-greedy logistic (no Arrow)"
	@echo "  make move_agreement       - greedy vs tree move agreement stats (no Arrow)"
	@echo "  make tablebase_opp1_gen   - build opp-1-card tablebase binary generator (no Arrow)"
	@echo "  make exact_hint           - exact P(beat) for each single move via exhaustive enum"
	@echo "  make research             - standalone research/*.cpp -> build/research/"
	@echo "  make clean"

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

$(BUILD_DIR)/src/simulation/nn_game_runner.o: src/simulation/nn_game_runner.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/datagen/parquet_export.o: src/datagen/parquet_export.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/datagen/samples_md.o: src/datagen/samples_md.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/datagen/generate_data.o: src/datagen/generate_data.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/datagen/generate_nn_data.o: src/datagen/generate_nn_data.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/datagen/generate_nn_selfplay.o: src/datagen/generate_nn_selfplay.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -c $< -o $@

# typed_search module: compile every .cpp under src/players/typed_search/.
TYPED_SEARCH_CPP := $(wildcard src/players/typed_search/*.cpp)
TYPED_SEARCH_OBJS := $(patsubst src/players/typed_search/%.cpp,$(BUILD_DIR)/src/players/typed_search/%.o,$(TYPED_SEARCH_CPP))

$(BUILD_DIR)/src/players/typed_search/%.o: src/players/typed_search/%.cpp | dirs
	@mkdir -p $(BUILD_DIR)/src/players/typed_search
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

# az_search module: torch-free core (search/grouping/forced search) links into
# test_core; nn_eval.cpp is torch-dependent (explicit override below). Defined
# here (before TEST_OBJS) because TEST_OBJS expands AZ_SEARCH_CORE_OBJS immediately.
# Torch-dependent az_search sources (LibTorch / NN). Everything else under
# az_search/ is the torch-free core that links into test_core.
AZ_SEARCH_NN_CPP    := src/players/az_search/nn_eval.cpp src/players/az_search/az_search_player.cpp
AZ_SEARCH_CORE_CPP  := $(filter-out $(AZ_SEARCH_NN_CPP),$(wildcard src/players/az_search/*.cpp))
AZ_SEARCH_CORE_OBJS := $(patsubst src/players/az_search/%.cpp,$(BUILD_DIR)/src/players/az_search/%.o,$(AZ_SEARCH_CORE_CPP))
AZ_SEARCH_NN_OBJS   := $(patsubst src/players/az_search/%.cpp,$(BUILD_DIR)/src/players/az_search/%.o,$(AZ_SEARCH_NN_CPP))
AZ_SEARCH_OBJS      := $(AZ_SEARCH_CORE_OBJS) $(AZ_SEARCH_NN_OBJS)

# Torch-free pattern rule for az_search core sources.
$(BUILD_DIR)/src/players/az_search/%.o: src/players/az_search/%.cpp | dirs
	@mkdir -p $(BUILD_DIR)/src/players/az_search
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

# Static pattern rule (overrides the pattern above for the listed targets):
# torch-dependent az_search objects get TORCH_INCLUDES + -DBIG2_WITH_TORCH.
$(AZ_SEARCH_NN_OBJS): $(BUILD_DIR)/src/players/az_search/%.o: src/players/az_search/%.cpp | dirs
	@mkdir -p $(BUILD_DIR)/src/players/az_search
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -DBIG2_WITH_TORCH -c $< -o $@

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
             $(TYPED_SEARCH_OBJS) \
             $(AZ_SEARCH_CORE_OBJS) \
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
                $(TYPED_SEARCH_OBJS) \
                $(BUILD_DIR)/src/datagen/parquet_export.o \
                $(BUILD_DIR)/src/datagen/samples_md.o \
                $(BUILD_DIR)/src/datagen/generate_data.o

$(BIN_DIR)/generate_data: $(GENDATA_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_PARQUET)
	@echo "✓ $(BIN_DIR)/generate_data"

generate_data: dirs $(BIN_DIR)/generate_data

# ============================================================================
# bin/generate_nn_data — NN training data (random self-play, needs libarrow)
# ============================================================================

GENNNDATA_OBJS := $(CORE_OBJS) \
                  $(BUILD_DIR)/src/simulation/nn_game_runner.o \
                  $(BUILD_DIR)/src/datagen/parquet_export.o \
                  $(BUILD_DIR)/src/datagen/generate_nn_data.o

$(BIN_DIR)/generate_nn_data: $(GENNNDATA_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_PARQUET)
	@echo "✓ $(BIN_DIR)/generate_nn_data"

generate_nn_data: dirs $(BIN_DIR)/generate_nn_data

# ============================================================================
# bin/generate_nn_selfplay — NN self-play data (needs libarrow + LibTorch)
# ============================================================================

GENNN_SELFPLAY_OBJS := $(CORE_OBJS) \
                       $(BUILD_DIR)/src/simulation/nn_game_runner.o \
                       $(BUILD_DIR)/src/datagen/parquet_export.o \
                       $(BUILD_DIR)/src/datagen/generate_nn_selfplay.o

$(BIN_DIR)/generate_nn_selfplay: $(GENNN_SELFPLAY_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_TORCH)
	@echo "✓ $(BIN_DIR)/generate_nn_selfplay"

generate_nn_selfplay: dirs $(BIN_DIR)/generate_nn_selfplay

# ============================================================================
# bin/eval_nn_match — head-to-head NN model evaluation (needs LibTorch)
# ============================================================================

$(BUILD_DIR)/src/datagen/eval_nn_match.o: src/datagen/eval_nn_match.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -c $< -o $@

EVALNNMATCH_OBJS := $(CORE_OBJS) \
                    $(BUILD_DIR)/src/simulation/nn_game_runner.o \
                    $(BUILD_DIR)/src/datagen/eval_nn_match.o

$(BIN_DIR)/eval_nn_match: $(EVALNNMATCH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_TORCH)
	@echo "✓ $(BIN_DIR)/eval_nn_match"

eval_nn_match: dirs $(BIN_DIR)/eval_nn_match

# ============================================================================
# ============================================================================
# bin/play_games — play games and print per-turn NN decisions (needs LibTorch)
# ============================================================================

$(BUILD_DIR)/src/datagen/play_games.o: src/datagen/play_games.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -c $< -o $@

PLAYGAMES_OBJS := $(CORE_OBJS) \
                  $(BUILD_DIR)/src/datagen/play_games.o

$(BIN_DIR)/play_games: $(PLAYGAMES_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_TORCH)
	@echo "✓ $(BIN_DIR)/play_games"

play_games: dirs $(BIN_DIR)/play_games

# ============================================================================
# bin/eval_nn_vs_classic — NN model vs classic player evaluation (needs LibTorch)
# ============================================================================

$(BUILD_DIR)/src/datagen/eval_nn_vs_classic.o: src/datagen/eval_nn_vs_classic.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -c $< -o $@

EVALNNVSCLA_OBJS := $(CORE_OBJS) \
                    $(BUILD_DIR)/src/simulation/game_simulator.o \
                    $(TYPED_SEARCH_OBJS) \
                    $(BUILD_DIR)/src/datagen/eval_nn_vs_classic.o

$(BIN_DIR)/eval_nn_vs_classic: $(EVALNNVSCLA_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_TORCH)
	@echo "✓ $(BIN_DIR)/eval_nn_vs_classic"

eval_nn_vs_classic: dirs $(BIN_DIR)/eval_nn_vs_classic

# ============================================================================
# bin/typed_search_train — generation training driver (no Arrow)
# ============================================================================

$(BUILD_DIR)/src/datagen/typed_search_train.o: src/datagen/typed_search_train.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

TYPED_TRAIN_OBJS := $(CORE_OBJS) \
                    $(BUILD_DIR)/src/simulation/game_simulator.o \
                    $(TYPED_SEARCH_OBJS) \
                    $(BUILD_DIR)/src/datagen/typed_search_train.o

$(BIN_DIR)/typed_search_train: $(TYPED_TRAIN_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/typed_search_train"

typed_search_train: dirs $(BIN_DIR)/typed_search_train

# ============================================================================
# bin/eval_match — head-to-head evaluation (no Arrow)
# ============================================================================

# PIMC stats (dets saved, etc.) printed at end of eval_match when enabled.
$(BUILD_DIR)/src/datagen/eval_match.o: src/datagen/eval_match.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -DBIG2_PIMC_STATS=1 -c $< -o $@

EVALMATCH_OBJS := $(CORE_OBJS) \
                  $(BUILD_DIR)/src/simulation/game_simulator.o \
                  $(TYPED_SEARCH_OBJS) \
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
                  $(TYPED_SEARCH_OBJS) \
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
# bin/exact_hint — exact P(opponent beats move) via exhaustive enumeration
# ============================================================================

$(BUILD_DIR)/research/exact_hint.o: research/exact_hint.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

EXACT_HINT_OBJS := $(CORE_OBJS) \
                   $(BUILD_DIR)/research/exact_hint.o

$(BIN_DIR)/exact_hint: $(EXACT_HINT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/exact_hint"

exact_hint: dirs $(BIN_DIR)/exact_hint

# ============================================================================
# bin/az_selfplay — self-play data generation for az_search (needs LibTorch+Arrow)
# ============================================================================

$(BUILD_DIR)/src/datagen/az_selfplay.o: src/datagen/az_selfplay.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -DBIG2_WITH_TORCH -c $< -o $@

AZ_SELFPLAY_OBJS := $(CORE_OBJS) \
                    $(BUILD_DIR)/src/simulation/game_simulator.o \
                    $(TYPED_SEARCH_OBJS) \
                    $(AZ_SEARCH_OBJS) \
                    $(BUILD_DIR)/src/datagen/az_selfplay.o

$(BIN_DIR)/az_selfplay: $(AZ_SELFPLAY_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_TORCH)
	@echo "✓ $(BIN_DIR)/az_selfplay"

az_selfplay: dirs $(BIN_DIR)/az_selfplay

# ============================================================================
# bin/eval_az_match — paired-deal az_search evaluation (needs LibTorch+Arrow)
# ============================================================================

$(BUILD_DIR)/src/datagen/eval_az_match.o: src/datagen/eval_az_match.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -DBIG2_WITH_TORCH -c $< -o $@

EVAL_AZ_MATCH_OBJS := $(CORE_OBJS) \
                      $(BUILD_DIR)/src/simulation/game_simulator.o \
                      $(TYPED_SEARCH_OBJS) \
                      $(AZ_SEARCH_OBJS) \
                      $(BUILD_DIR)/src/datagen/eval_az_match.o

$(BIN_DIR)/eval_az_match: $(EVAL_AZ_MATCH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_TORCH)
	@echo "✓ $(BIN_DIR)/eval_az_match"

eval_az_match: dirs $(BIN_DIR)/eval_az_match

# ============================================================================
# bin/az_nn_check — cross-check C++ NN inference vs Python (needs LibTorch)
# ============================================================================

$(BUILD_DIR)/research/az_nn_check.o: research/az_nn_check.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -DBIG2_WITH_TORCH -c $< -o $@

AZ_NN_CHECK_OBJS := $(CORE_OBJS) \
                    $(TYPED_SEARCH_OBJS) \
                    $(AZ_SEARCH_OBJS) \
                    $(BUILD_DIR)/research/az_nn_check.o

$(BIN_DIR)/az_nn_check: $(AZ_NN_CHECK_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_TORCH)
	@echo "✓ $(BIN_DIR)/az_nn_check"

az_nn_check: dirs $(BIN_DIR)/az_nn_check

# ============================================================================
# bin/az_search_smoke — end-to-end search w/ real NN evaluator (needs LibTorch)
# ============================================================================

$(BUILD_DIR)/research/az_search_smoke.o: research/az_search_smoke.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -DBIG2_WITH_TORCH -c $< -o $@

AZ_SEARCH_SMOKE_OBJS := $(CORE_OBJS) \
                        $(TYPED_SEARCH_OBJS) \
                        $(AZ_SEARCH_OBJS) \
                        $(BUILD_DIR)/research/az_search_smoke.o

$(BIN_DIR)/az_search_smoke: $(AZ_SEARCH_SMOKE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_TORCH)
	@echo "✓ $(BIN_DIR)/az_search_smoke"

az_search_smoke: dirs $(BIN_DIR)/az_search_smoke

# ============================================================================
# bin/az_play_check — AzSearchPlayer vs greedy via GameSimulator (needs LibTorch)
# ============================================================================

$(BUILD_DIR)/research/az_play_check.o: research/az_play_check.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -DBIG2_WITH_TORCH -c $< -o $@

AZ_PLAY_CHECK_OBJS := $(CORE_OBJS) \
                      $(BUILD_DIR)/src/simulation/game_simulator.o \
                      $(TYPED_SEARCH_OBJS) \
                      $(AZ_SEARCH_OBJS) \
                      $(BUILD_DIR)/research/az_play_check.o

$(BIN_DIR)/az_play_check: $(AZ_PLAY_CHECK_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_TORCH)
	@echo "✓ $(BIN_DIR)/az_play_check"

az_play_check: dirs $(BIN_DIR)/az_play_check

# ============================================================================
# bin/move_audit — az_search move-set audit dump + structural checks (no Arrow)
# ============================================================================

$(BUILD_DIR)/research/move_audit.o: research/move_audit.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

MOVE_AUDIT_OBJS := $(CORE_OBJS) \
                   $(BUILD_DIR)/research/move_audit.o

$(BIN_DIR)/move_audit: $(MOVE_AUDIT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/move_audit"

move_audit: dirs $(BIN_DIR)/move_audit

# ============================================================================
# bin/az_compose_gen — dump the player-policy composition map (nn/az_compose.json)
# ============================================================================

$(BUILD_DIR)/research/az_compose_gen.o: research/az_compose_gen.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

AZ_COMPOSE_GEN_OBJS := $(CORE_OBJS) \
                       $(BUILD_DIR)/research/az_compose_gen.o

$(BIN_DIR)/az_compose_gen: $(AZ_COMPOSE_GEN_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/az_compose_gen"

az_compose_gen: dirs $(BIN_DIR)/az_compose_gen

# ============================================================================
# bin/az_vs_teacher_agree — az(net+search) vs typed_search move-disagreement dump
# ============================================================================

$(BUILD_DIR)/research/az_vs_teacher_agree.o: research/az_vs_teacher_agree.cpp | dirs
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -DBIG2_WITH_TORCH -c $< -o $@

AZ_AGREE_OBJS := $(CORE_OBJS) \
                 $(BUILD_DIR)/src/simulation/game_simulator.o \
                 $(TYPED_SEARCH_OBJS) \
                 $(AZ_SEARCH_OBJS) \
                 $(BUILD_DIR)/research/az_vs_teacher_agree.o

$(BIN_DIR)/az_vs_teacher_agree: $(AZ_AGREE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_TORCH)
	@echo "✓ $(BIN_DIR)/az_vs_teacher_agree"

az_vs_teacher_agree: dirs $(BIN_DIR)/az_vs_teacher_agree

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
