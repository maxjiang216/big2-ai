#ifndef TABLEBASE_PEEK_H
#define TABLEBASE_PEEK_H

#include "move.h"
#include "partial_game.h"
#include "tablebase_opp1.h"
#include "util.h"

#include <optional>

// Same decision logic as Player::tablebase_move(), but operates on a const
// PartialGame snapshot (no Player state). Used by analysis tools.
struct TablebasePeekResult {
  std::optional<Move> move;
  int tb_case = -1;
  int tb_forced_seq_len = 0;
  bool tb_opp1_table_straight = false;
};

inline TablebasePeekResult peek_tablebase_move(const PartialGame &game_) {
  TablebasePeekResult r;

  const int hand_size = game_.num_cards();
  for (int mid : game_.get_legal_moves()) {
    if (mid == kPASS)
      continue;
    if (MOVE_TO_CARDS[mid][13] == hand_size) {
      r.move = Move(mid);
      return r;
    }
  }

  if (game_.last_move().combination == Move::Combination::kPass) {

    if (game_.opponent_hand_size() == 1) {
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
      if (all_singles && best_rank != -1) {
        r.tb_case = 2;
        r.tb_opp1_table_straight = false;
        r.move = Move(Move::Combination::kSingle, best_rank);
        return r;
      }

      Opp1Result opp1 = lookup_opp1(game_.player_hand());
      if (opp1.first_move_id != 0) {
        for (int mid : legal) {
          if (mid == opp1.first_move_id) {
            r.tb_case = 2;
            r.tb_opp1_table_straight = true;
            r.move = Move(mid);
            return r;
          }
        }
      }
      if (auto def = opp1_default_strategy_move(game_.player_hand())) {
        r.tb_case = 2;
        r.tb_opp1_table_straight = false;
        r.move = Move(*def);
        return r;
      }
    }

    auto seq = find_forced_win(game_.player_hand(), game_.discard_pile(),
                               game_.opponent_hand_size());
    if (seq) {
      r.tb_case = 1;
      r.tb_forced_seq_len = static_cast<int>(seq->size());
      r.move = Move((*seq)[0]);
      return r;
    }
  }
  return r;
}

#endif
