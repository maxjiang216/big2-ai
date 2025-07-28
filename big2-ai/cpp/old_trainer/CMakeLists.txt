cmake_minimum_required(VERSION 3.10)
project(GreedyTrainer)

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find and require zlib (needed by cnpy)
find_package(ZLIB REQUIRED)

# Include the directory containing cnpy.h
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cnpy)

# List your source files. Adjust this list if you have more or fewer files.
set(SOURCES
    main_big.cpp
    game.cpp
    greedyplayer.cpp
    greedytrainer.cpp
    move.cpp
    util.cpp
    cnpy/cnpy.cpp
)

# Create the executable
add_executable(greedy_big ${SOURCES})

# Link against zlib
target_link_libraries(greedy_big ${ZLIB_LIBRARIES})
