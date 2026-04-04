#ifndef GREEDY_LINEAR_PLAYER_H
#define GREEDY_LINEAR_PLAYER_H

#include "greedy_player.h"
#include "linear_evaluator.h"

#include <string>

// Greedy move choice using Ridge linear value (same pipeline as ``PimcLinearPlayer`` rollout).
class GreedyLinearPlayer : public Player {
public:
  explicit GreedyLinearPlayer(const std::string &model_path) : linear_(model_path) {}

protected:
  Move select_move_impl() override {
    return greedy_best(game_, game_.get_legal_moves(),
                       [this](const PartialGame &sim) { return linear_.predict(sim); });
  }

private:
  LinearEvaluator linear_;
};

#endif
