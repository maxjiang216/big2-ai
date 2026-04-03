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
  // Which tablebase case fired on this turn (-1 = none).
  // 1 = forced-win sequence (from lead), 2 = opponent has 1 card.
  // Immediate winning moves are not tagged (they always occur on the last turn).
  int tb_case = -1;
  // If tb_case==1: number of moves in the full forced-win plan (including the first).
  int tb_forced_seq_len = 0;
  // If tb_case==2: true when the move follows the opp-1 precomputed straight table.
  bool tb_opp1_table_straight = false;
};

// Per-game counts for tablebase use: each "segment" (consecutive case1 or case2
// turns by the same player) contributes one hit — the first turn of the segment.
// Matches selfplay / benchmark aggregation for mean forced-seq length.
struct TablebaseFirstHitStats {
  int case1 = 0;
  int case2 = 0;
  long long case1_seq_sum = 0;
  int case1_seq_max = 0;
  int case2_table_straight = 0;
};

class GameRecord {
public:
  GameRecord();

  void set_initial_state(const Game &game);

  void add_move(const Move &move, int tb_case = -1, int tb_forced_seq_len = 0,
                bool tb_opp1_table_straight = false);

  const Game &game() const { return _game; }
  const std::vector<TurnRecord> &turns() const { return _turns; }

private:
  Game _game;
  std::array<PartialGame, 2> _views;
  std::vector<TurnRecord> _turns;
};

TablebaseFirstHitStats tablebase_first_hit_stats(const GameRecord &record);

#endif
