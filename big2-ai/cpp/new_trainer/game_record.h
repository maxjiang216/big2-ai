// game_record.h
#ifndef GAME_RECORD_H
#define GAME_RECORD_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "move.h"
#include "partial_game.h"
#include <arrow/api.h>
#include <parquet/arrow/writer.h>

// Forward declaration
class Game;

struct TurnRecord {

  int current_player;
  Game game;
  std::array<PartialGame, 2> views;
  // Legal moves for current player
  std::vector<int> legal_moves;
  // Possible legal moves for opponent
  std::vector<int> possible_moves;
  // Move played
  Move move;
};

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

  const Game &game() const { return _game; }
  const std::vector<TurnRecord> &turns() const { return _turns; }

private:
  Game _game;
  std::array<PartialGame, 2> _views;
  // Sequence of moves for the game
  std::vector<TurnRecord> _turns;
};

#endif // GAME_RECORD_H
