# ====== CONFIG ======
CXX         = g++
CXXFLAGS    = -O2 -std=c++17 -Wall -Wextra -march=native -fopenmp
LDFLAGS     = -lparquet -larrow -pthread
INCLUDES    = -I. -Icore -Idatagen -Iplayers -Ifeatures

# Directories
SRC_DIRS    = core datagen players features
BUILD_DIR   = build
BIN_DIR     = bin

# Executables
GENERATE_DATA = $(BIN_DIR)/generate_data

# ====== AUTO-DISCOVERY ======
# Find all .cpp files in library directories (excluding generate_data.cpp)
LIB_SOURCES := $(foreach dir,$(SRC_DIRS),$(shell find $(dir) -name '*.cpp' ! -name 'generate_data.cpp' 2>/dev/null))
LIB_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(LIB_SOURCES))

# Main executable source
GENERATE_DATA_SRC = datagen/generate_data.cpp
GENERATE_DATA_OBJ = $(BUILD_DIR)/datagen/generate_data.o

# ====== TARGETS ======
.PHONY: all clean dirs help datagen

# Default: show help
all:
	@echo "Use 'make datagen' to build the data generation binary"
	@echo "Or let the Python scripts handle it automatically"

# Data generation binary
datagen: dirs $(GENERATE_DATA)
	@echo "✓ Data generation binary ready: $(GENERATE_DATA)"

# Create necessary directories
dirs:
	@mkdir -p $(BUILD_DIR)/core $(BUILD_DIR)/datagen
	@mkdir -p $(BUILD_DIR)/players/random $(BUILD_DIR)/players/greedy
	@mkdir -p $(BUILD_DIR)/features/game_level $(BUILD_DIR)/features/turn_level
	@mkdir -p $(BIN_DIR)

# Link generate_data executable
$(GENERATE_DATA): $(GENERATE_DATA_OBJ) $(LIB_OBJECTS)
	@echo "🔗 Linking generate_data"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile library object files
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "📦 Compiling $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@rm -f *.parquet logs/*.txt 2>/dev/null || true
	@echo "✓ Clean complete"

help:
	@echo "Big 2 AI - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make datagen  - Build data generation binary"
	@echo "  make clean    - Remove all build artifacts"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Executables:"
	@echo "  $(GENERATE_DATA) - Self-play data generation"
	@echo ""
	@echo "Note: Python scripts will automatically build what they need"

# ====== DEBUGGING ======
.PHONY: show-sources show-objects

show-sources:
	@echo "Library sources:"
	@echo "$(LIB_SOURCES)" | tr ' ' '\n'
	@echo ""
	@echo "Main source:"
	@echo "  $(GENERATE_DATA_SRC)"

show-objects:
	@echo "Library objects:"
	@echo "$(LIB_OBJECTS)" | tr ' ' '\n'
	@echo ""
	@echo "Main object:"
	@echo "  $(GENERATE_DATA_OBJ)"