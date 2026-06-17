#ifndef PLAYER_H
#define PLAYER_H

#include "game.h"
#include "move.h"
#include "partial_game.h"
#include "tablebase_opp1.h"
#include "tablebase_peek.h"
#include "util.h"

#include <array>
#include <optional>

class Player {
public:
  virtual ~Player() = default;

  virtual void reset() {}

  // Deal lives in Game; each player’s view is built only from public info for
  // that seat (PartialGame(game, player_num)). After that, state stays in sync
  // by applying the same move stream: this player in select_move(), the
  // opponent in accept_opponent_move() — no full-state resync from Game.
  void accept_deal(const Game &game, int player_num) {
    game_ = PartialGame(game, player_num);
    on_deal(game.player_hand(player_num), player_num);
  }

  // Non-virtual: advances shared state, then calls on_opponent_move.
  void accept_opponent_move(const Move &move) {
    game_.apply_move(move);
    on_opponent_move(move);
  }

  // Non-virtual: checks tablebase first, then delegates to select_move_impl.
  // Applies the chosen move to the shared PartialGame before returning.
  // Records which tablebase case fired (accessible via last_tb_case()).
  Move select_move() {
    last_tb_case_ = -1;
    last_tb_forced_seq_len_ = 0;
    last_tb_opp1_table_straight_ = false;
    auto tb = tablebase_move();
    if (tb) {
      game_.apply_move(*tb);
      on_self_move(*tb);  // tablebase skips still enter the move history
      return *tb;
    }
    Move chosen = select_move_impl();
    game_.apply_move(chosen);
    on_self_move(chosen);
    return chosen;
  }

  // Which tablebase case fired on the most recent select_move() call.
  // -1 = no tracked tablebase move, 1 = forced-win sequence, 2 = opp has 1 card.
  // Winning moves that empty the hand are not tagged (see tablebase_move).
  int last_tb_case() const { return last_tb_case_; }
  // When last_tb_case()==1: length of the full forced-win move sequence.
  int last_tb_forced_seq_len() const { return last_tb_forced_seq_len_; }
  // When last_tb_case()==2: true if the move came from the opp-1 straight table.
  bool last_tb_opp1_table_straight() const { return last_tb_opp1_table_straight_; }

protected:
  // Shared view of the game state, kept in sync by the base class.
  PartialGame game_;

  // Called after accept_deal updates game_. Override for subclass init.
  virtual void on_deal(const std::array<int, 13> & /*hand*/, int /*turn*/) {}

  // Called after accept_opponent_move updates game_. Override if needed.
  virtual void on_opponent_move(const Move & /*move*/) {}

  // Called after select_move applies OUR chosen move (search-selected OR
  // tablebase root-skip), so history-tracking players see every applied move.
  virtual void on_self_move(const Move & /*move*/) {}

  // Subclasses implement their move-selection logic here.
  // Must return a legal move. Must NOT call game_.apply_move() — the base does it.
  virtual Move select_move_impl() = 0;

  int last_tb_case_ = -1;
  int last_tb_forced_seq_len_ = 0;
  bool last_tb_opp1_table_straight_ = false;

  // Returns a tablebase move when one exists, nullopt otherwise.
  // Sets last_tb_case_ when a case worth recording fires (1 or 2). Immediate
  // hand-emptying wins are not tagged — they always occur on the final turn.
  std::optional<Move> tablebase_move() {
    TablebasePeekResult r = peek_tablebase_move(game_);
    last_tb_case_ = r.tb_case;
    last_tb_forced_seq_len_ = r.tb_forced_seq_len;
    last_tb_opp1_table_straight_ = r.tb_opp1_table_straight;
    return r.move;
  }
};

#endif
