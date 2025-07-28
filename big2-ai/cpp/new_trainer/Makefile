# ====== CONFIG ======
CXX         = g++
CXXFLAGS    = -O2 -std=c++17 -Wall -Wextra -march=native -fopenmp
LDFLAGS     = -lparquet -larrow -pthread
INCLUDES    = -Icpp        # Path to your headers

SRC_DIR     = .
SRCS        = $(wildcard $(SRC_DIR)/*.cpp)
OBJS        = $(SRCS:.cpp=.o)
TARGET      = big2-trainer

# ====== RULES ======
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

format:
	clang-format -i $(SRC_DIR)/*.cpp $(SRC_DIR)/*.h

# ====== UTILITIES ======
.PHONY: all clean format
