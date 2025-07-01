// game_record.h
#ifndef GAME_RECORD_H
#define GAME_RECORD_H

#include <vector>
#include <array>
#include <memory>
#include <string>

#include "move.h"
#include <arrow/api.h>
#include <parquet/arrow/writer.h>

// Forward declaration
class Game;

/**
 * @brief Records the sequence of moves and outcome of a single Big 2 game.
 * Supports conversion to Arrow tables for Parquet serialization.
 */
class GameRecord {
public:
    GameRecord();

    /**
     * @brief Capture the initial state of the game (starting hands).
     * @param game The Game object after deal but before any moves.
     */
    void set_initial_state(const Game &game);

    /**
     * @brief Append a move by a player to the record.
     * @param player Index of the player (0 or 1).
     * @param move   The Move they played.
     */
    void add_move(const Move &move);

    /**
     * @brief Get a JSON string (for debugging).
     */
    std::string to_json() const;

    /**
     * @brief Return the Arrow schema for GameRecord.
     */
    static std::shared_ptr<arrow::Schema> arrow_schema();

    /**
     * @brief Convert a vector of GameRecord into an Arrow Table for Parquet.
     * @param records Records to convert.
     * @return Arrow Table containing all records.
     */
    static arrow::Result<std::shared_ptr<arrow::Table>> ToTable(
        const std::vector<GameRecord>& records);

private:
    // Starting hand distributions for each player (rank counts)
    std::array<std::array<int,13>,2> _initial_hands;

    // Sequence of moves for the game
    std::vector<Move> _moves;
};

#endif // GAME_RECORD_H
