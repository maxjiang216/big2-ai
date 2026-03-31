#ifndef GAME_RECORD_H
#define GAME_RECORD_H

#include "game.h"
#include "move.h"
#include "partial_game.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

struct TurnRecord {
  int current_player;
  Game game;
  std::array<PartialGame, 2> views;
  std::vector<int> legal_moves;
  std::vector<int> possible_moves;
  Move move;
};

class GameRecord {
public:
  GameRecord();

  void set_initial_state(const Game &game);

  void add_move(const Move &move);

  const Game &game() const { return _game; }
  const std::vector<TurnRecord> &turns() const { return _turns; }

private:
  Game _game;
  std::array<PartialGame, 2> _views;
  std::vector<TurnRecord> _turns;
};

#endif
