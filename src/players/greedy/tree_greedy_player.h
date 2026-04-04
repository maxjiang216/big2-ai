#ifndef TREE_GREEDY_PLAYER_H
#define TREE_GREEDY_PLAYER_H

#include "greedy_player.h"
#include "tree_evaluator.h"

#include <stdexcept>
#include <string>

// Greedy player whose position evaluator is a trained decision tree.
// Uses the same greedy_best() template as GreedyPlayer, but with a
// TreeEvaluator that predicts win probability (double) instead of the
// hand-crafted GreedyEval tuple.
class TreeGreedyPlayer : public Player {
public:
  explicit TreeGreedyPlayer(const std::string &model_path)
      : evaluator_(model_path) {
    if (!evaluator_.loaded())
      throw std::runtime_error("TreeGreedyPlayer: model failed to load from '" +
                               model_path + "'");
  }

protected:
  Move select_move_impl() override {
    return greedy_best(game_, game_.get_legal_moves(),
                       [this](const PartialGame &sim) {
                         return evaluator_.predict(sim);
                       });
  }

private:
  TreeEvaluator evaluator_;
};

#endif
