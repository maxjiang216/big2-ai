#ifndef GREEDY_PLAYER_H
#define GREEDY_PLAYER_H

#include "move.h"
#include "partial_game.h"
#include "player.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

struct GreedyEval {
  int win_now;
  int bombs;
  int neg_num_cards;
  int num_2s;
  int num_as;
  int num_ks;
  int num_qs;
  int num_js;
  int num_10s;
  int num_9s;
  int num_8s;
  int num_7s;
  int num_6s;
  int num_5s;
  int num_4s;

  bool operator<(const GreedyEval &other) const {
    return std::tie(win_now, bombs, neg_num_cards, num_2s, num_as, num_ks,
                    num_qs, num_js, num_10s, num_9s, num_8s, num_7s, num_6s,
                    num_5s, num_4s) <
           std::tie(other.win_now, other.bombs, other.neg_num_cards,
                    other.num_2s, other.num_as, other.num_ks, other.num_qs,
                    other.num_js, other.num_10s, other.num_9s, other.num_8s,
                    other.num_7s, other.num_6s, other.num_5s, other.num_4s);
  }
};

class GreedyPlayer : public Player {
protected:
  Move select_move_impl() override {
    std::vector<int> legal = game_.get_legal_moves();

    std::vector<int> nonpass_moves;
    for (int m : legal) {
      if (Move(m).combination != Move::Combination::kPass)
        nonpass_moves.push_back(m);
    }

    if (nonpass_moves.empty())
      return Move(kPASS);

    auto best_it = nonpass_moves.begin();
    GreedyEval best_eval = evaluate_after_move(game_, Move(*best_it));
    for (auto it = nonpass_moves.begin() + 1; it != nonpass_moves.end(); ++it) {
      GreedyEval eval = evaluate_after_move(game_, Move(*it));
      if (best_eval < eval) {
        best_eval = eval;
        best_it = it;
      }
    }
    return Move(*best_it);
  }

private:
  static GreedyEval evaluate_after_move(const PartialGame &base,
                                        const Move &move) {
    PartialGame sim = base;
    sim.apply_move(move);

    const auto &h = sim.player_hand();
    int n_cards = 0, n_bombs = 0;
    for (int i = 0; i < 13; ++i) {
      n_cards += h[i];
      if ((i < 11 && h[i] == 4) || (i == 11 && h[i] == 3))
        ++n_bombs;
    }

    GreedyEval eval;
    eval.win_now       = (n_cards == 0) ? 1 : 0;
    eval.bombs         = n_bombs;
    eval.neg_num_cards = -n_cards;
    eval.num_2s        = h[12];
    eval.num_as        = h[11];
    eval.num_ks        = h[10];
    eval.num_qs        = h[9];
    eval.num_js        = h[8];
    eval.num_10s       = h[7];
    eval.num_9s        = h[6];
    eval.num_8s        = h[5];
    eval.num_7s        = h[4];
    eval.num_6s        = h[3];
    eval.num_5s        = h[2];
    eval.num_4s        = h[1];
    return eval;
  }
};

#endif
