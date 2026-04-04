#ifndef TREE_GREEDY_PLAYER_FACTORY_H
#define TREE_GREEDY_PLAYER_FACTORY_H

#include "player_factory.h"
#include "tree_greedy_player.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

// Factory for TreeGreedyPlayer.
//
// `param` is the max_depth of the model to load.  The model file path is
// constructed as:  data/tree_model_d{int(param)}.txt
//
// Example round-robin config entry:
//   {"type": "tree_greedy", "param": 5, "label": "tree_d5"}
class TreeGreedyPlayerFactory : public PlayerFactory {
public:
  explicit TreeGreedyPlayerFactory(double param)
      : model_path_("data/tree_model_d" + std::to_string(static_cast<int>(std::round(param))) + ".txt") {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<TreeGreedyPlayer>(model_path_);
  }

private:
  std::string model_path_;
};

#endif
