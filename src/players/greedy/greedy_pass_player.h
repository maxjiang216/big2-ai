#ifndef GREEDY_PASS_PLAYER_H
#define GREEDY_PASS_PLAYER_H

#include "greedy_player.h"
#include "linear_evaluator.h"
#include "player.h"

#include <stdexcept>
#include <string>

// Greedy policy with optional voluntary pass when Ridge model predicts Δ>0
// (Δ ≈ E[win|pass] − E[win|greedy] on training distribution).
class GreedyPassPlayer : public Player {
public:
  explicit GreedyPassPlayer(const std::string &model_path)
      : pass_model_(model_path) {
    if (!pass_model_.loaded())
      throw std::runtime_error("GreedyPassPlayer: model failed to load from '" +
                               model_path + "'");
  }

protected:
  Move select_move_impl() override {
    const std::vector<int> legal = game_.get_legal_moves();
    if (voluntary_pass_legal(legal) && pass_model_.predict(game_) > 0.0)
      return Move(kPASS);
    return greedy_best(game_, legal, greedy_hand_eval);
  }

private:
  LinearEvaluator pass_model_;
};

#endif
