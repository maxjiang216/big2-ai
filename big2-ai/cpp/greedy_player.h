#ifndef GREEDY_PLAYER_H
#define GREEDY_PLAYER_H

#include "move.h"
#include "partial_game.h"
#include "player.h"
#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

struct GreedyEval {
  int win_now; // 1 if hand is empty, else 0 (most important)
  int bombs;
  int neg_num_cards; // -hand.size(), so fewer cards is "greater"
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

  // Lexicographical comparison, higher tuple is "better"
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
public:
  GreedyPlayer() = default;

  void accept_deal(std::array<int, 13> hand, int turn) override {
    game_ = PartialGame(hand, turn);
  }

  void accept_opponent_move(const Move &move) override {
    game_.apply_move(move);
  }

  Move select_move() override {
    std::vector<int> legal = game_.get_legal_moves();

    // Separate pass moves from real moves
    std::vector<int> nonpass_moves;
    for (int m : legal) {
      Move move(m);
      if (move.combination != Move::Combination::kPass) {
        nonpass_moves.push_back(m);
      }
    }

    if (nonpass_moves.empty()) {
      // Only pass is legal: play pass
      Move pass_move(legal.front());
      game_.apply_move(pass_move);
      return pass_move;
    }

    // Find the candidate move with the best (highest) evaluation tuple
    auto best_it = nonpass_moves.begin();
    GreedyEval best_eval = evaluate_after_move(game_, Move(*best_it));
    for (auto it = nonpass_moves.begin(); it != nonpass_moves.end(); ++it) {
      GreedyEval eval = evaluate_after_move(game_, Move(*it));
      if (best_eval < eval) {
        best_eval = eval;
        best_it = it;
      }
    }
    Move chosen_move(*best_it);
    game_.apply_move(chosen_move);
    return chosen_move;
  }

private:
  PartialGame game_;

  GreedyEval evaluate_after_move(const PartialGame &base,
                                 const Move &move) const {
    PartialGame sim = base;
    sim.apply_move(move);

    const auto &hand_array = sim.player_hand();
    int n_cards = 0, n_bombs = 0;

    for (int i = 0; i < 13; ++i) {
      n_cards += hand_array[i];
      if (i < 11 && hand_array[i] == 4 || i == 11 && hand_array[i] == 3)
        ++n_bombs;
    }

    GreedyEval eval;
    eval.win_now = (n_cards == 0) ? 1 : 0;
    eval.bombs = n_bombs;
    eval.neg_num_cards = -n_cards;
    eval.num_2s = hand_array[12];
    eval.num_as = hand_array[11];
    eval.num_ks = hand_array[10];
    eval.num_qs = hand_array[9];
    eval.num_js = hand_array[8];
    eval.num_10s = hand_array[7];
    eval.num_9s = hand_array[6];
    eval.num_8s = hand_array[5];
    eval.num_7s = hand_array[4];
    eval.num_6s = hand_array[3];
    eval.num_5s = hand_array[2];
    eval.num_4s = hand_array[1];
    // Don't need 3s since we already have num_cards

    return eval;
  }
};

#endif // GREEDY_PLAYER_H
