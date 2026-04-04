#ifndef PIMC_TREE_PLAYER_FACTORY_H
#define PIMC_TREE_PLAYER_FACTORY_H

#include "pimc_player.h"
#include "player_factory.h"

#include <cmath>
#include <memory>
#include <string>

// Factory for PimcTreePlayer (PIMC with decision-tree rollout after tablebase).
//
// `param`     — number of determinization samples N (default 10 if 0).
// `tree_depth`— which `data/tree_model_d{tree_depth}.txt` to load (e.g. 10).
// `seed`      — RNG seed for hand sampling.
//
// Example: {"type": "pimc_tree", "param": 10, "label": "pimc_tree_n10_d10"}
class PimcTreePlayerFactory : public PlayerFactory {
public:
  PimcTreePlayerFactory(double param, int tree_depth, unsigned int seed)
      : num_samples_(std::max(1, static_cast<int>(std::round(param)))),
        model_path_("data/tree_model_d" + std::to_string(tree_depth) + ".txt"),
        seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<PimcTreePlayer>(num_samples_, model_path_, seed_++);
  }

private:
  int num_samples_;
  std::string model_path_;
  unsigned int seed_;
};

#endif
