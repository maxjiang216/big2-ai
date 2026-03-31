#ifndef PLAYER_H
#define PLAYER_H

#include "game.h"
#include "move.h"
#include "partial_game.h"
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
  Move select_move() {
    auto tb = tablebase_move();
    if (tb) {
      game_.apply_move(*tb);
      return *tb;
    }
    Move chosen = select_move_impl();
    game_.apply_move(chosen);
    return chosen;
  }

protected:
  // Shared view of the game state, kept in sync by the base class.
  PartialGame game_;

  // Called after accept_deal updates game_. Override for subclass init.
  virtual void on_deal(const std::array<int, 13> & /*hand*/, int /*turn*/) {}

  // Called after accept_opponent_move updates game_. Override if needed.
  virtual void on_opponent_move(const Move & /*move*/) {}

  // Subclasses implement their move-selection logic here.
  // Must return a legal move. Must NOT call game_.apply_move() — the base does it.
  virtual Move select_move_impl() = 0;

private:
  // Returns a tablebase move when one exists, nullopt otherwise.
  std::optional<Move> tablebase_move() {
    // Case 0: a legal move that empties the hand wins immediately, regardless
    // of what the opponent could theoretically respond with.
    const int hand_size = game_.num_cards();
    for (int mid : game_.get_legal_moves()) {
      if (mid == kPASS)
        continue;
      // Use MOVE_TO_CARDS card count (what apply_move subtracts), not
      // Move::numCards(), so case 0 stays aligned with PartialGame::apply_move.
      if (MOVE_TO_CARDS[mid][13] == hand_size)
        return Move(mid);
    }

    // Case 1: from the lead, search for a guaranteed winning sequence.
    // find_forced_win returns a sequence only when every step either empties
    // our hand or is provably unbeatable; otherwise returns nullopt and we
    // fall through to the subclass strategy.
    if (game_.last_move().combination == Move::Combination::kPass) {
      auto seq = find_forced_win(game_.player_hand(), game_.discard_pile(),
                                 game_.opponent_hand_size());
      if (seq)
        return Move((*seq)[0]);
    }

    // Case 2: opponent has exactly 1 card left and all our legal moves are
    // singles. Playing the highest single is weakly dominant: it beats the
    // opponent's card in every scenario where any single of ours would, and
    // ties in every scenario where none would.
    if (game_.last_move().combination == Move::Combination::kPass &&
        game_.opponent_hand_size() == 1) {
      auto legal = game_.get_legal_moves();
      int best_rank = -1;
      bool all_singles = true;
      for (int mid : legal) {
        Move m(mid);
        if (m.combination != Move::Combination::kSingle) {
          all_singles = false;
          break;
        }
        if (m.rank > best_rank)
          best_rank = m.rank;
      }
      if (all_singles && best_rank != -1)
        return Move(Move::Combination::kSingle, best_rank);
    }
    return std::nullopt;
  }
};

#endif
