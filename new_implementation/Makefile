# Build from new_implementation/ directory.
CXX         = g++
CXXFLAGS    = -O2 -std=c++17 -Wall -Wextra -march=native -fopenmp
INCLUDES    = -Icore -Isimulation -Iplayers -Idatagen -Ifeatures -Itest
LDFLAGS     = -pthread
LDFLAGS_PARQUET = -lparquet -larrow -pthread

BUILD_DIR   = build
BIN_DIR     = bin

CORE_CPP    := $(wildcard core/*.cpp)
SIM_CPP     := simulation/game_simulator.cpp
COORD_CPP   := simulation/game_coordinator.cpp
DATAGEN_CPP := datagen/parquet_export.cpp

TEST_CPP    := $(wildcard test/*.cpp)

.PHONY: all clean dirs test_core coordinator help

all: help

help:
	@echo "Targets:"
	@echo "  make test_core     - unit tests (no Arrow)"
	@echo "  make coordinator   - coordinator + Parquet export (needs libarrow)"
	@echo "  make clean"

dirs:
	@mkdir -p $(BUILD_DIR)/core $(BUILD_DIR)/simulation $(BUILD_DIR)/datagen $(BUILD_DIR)/test $(BIN_DIR)

# --- Test binary (core + simulator only, no Arrow) ---
TEST_OBJS := $(patsubst core/%.cpp,$(BUILD_DIR)/core/%.o,$(CORE_CPP)) \
             $(BUILD_DIR)/simulation/game_simulator.o \
             $(patsubst test/%.cpp,$(BUILD_DIR)/test/%.o,$(TEST_CPP))

$(BUILD_DIR)/test/test_random_games.o: test/test_random_games.cpp | dirs
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/core/%.o: core/%.cpp | dirs
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/simulation/game_simulator.o: simulation/game_simulator.cpp | dirs
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test/%.o: test/%.cpp | dirs
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)/test_core: $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ $(BIN_DIR)/test_core"

test_core: dirs $(BIN_DIR)/test_core

# --- Coordinator + Parquet (compile-only check; needs libarrow-dev) ---
$(BUILD_DIR)/simulation/game_coordinator.o: simulation/game_coordinator.cpp | dirs
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/datagen/parquet_export.o: datagen/parquet_export.cpp | dirs
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

COORD_OBJS := $(patsubst core/%.cpp,$(BUILD_DIR)/core/%.o,$(CORE_CPP)) \
              $(BUILD_DIR)/simulation/game_simulator.o \
              $(BUILD_DIR)/simulation/game_coordinator.o \
              $(BUILD_DIR)/datagen/parquet_export.o

coordinator: dirs $(COORD_OBJS)
	@echo "✓ Coordinator + Parquet objects built (no main binary)."

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
