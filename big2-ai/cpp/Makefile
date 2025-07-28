# ====== CONFIG ======
CXX         = g++
CXXFLAGS    = -O2 -std=c++17 -Wall -Wextra -march=native -fopenmp
LDFLAGS     = -lparquet -larrow -pthread
INCLUDES    = -I. -Ifeatures -Ifeatures/game_level -Ifeatures/turn_level

SRC_DIRS    = . features features/game_level features/turn_level
BUILD_DIR   = build
TARGET      = big2-trainer

# ====== SRC/OBJ DISCOVERY ======
SOURCES := $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.cpp))
OBJECTS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

# ====== DEFAULT TARGET ======
all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/features/game_level $(BUILD_DIR)/features/turn_level

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET) logs game_records.jsonl

format:
	clang-format -i $(SOURCES) $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.h))

# ====== UTILITIES ======
.PHONY: all clean format

