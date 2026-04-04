#ifndef PIMC_TREE_ADAPTIVE_PLAYER_FACTORY_H
#define PIMC_TREE_ADAPTIVE_PLAYER_FACTORY_H

#include "pimc_player.h"
#include "player_factory.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

// Adaptive PIMC with tree rollout after tablebase (data/tree_model_d{depth}.txt).
//
// `param` — max determinizations n_max (default 10 if 0).
// `tree_depth` — model file depth (e.g. 10).
class PimcTreeAdaptivePlayerFactory : public PlayerFactory {
public:
  PimcTreeAdaptivePlayerFactory(double param, int tree_depth, unsigned int seed)
      : n_max_(std::max(1, static_cast<int>(std::round(param)))),
        n_min_(std::min(std::max(3, n_max_ / 10), n_max_)),
        model_path_("data/tree_model_d" + std::to_string(tree_depth) + ".txt"),
        seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<PimcTreePlayer>(n_max_, n_min_, 0.05, model_path_,
                                              seed_++);
  }

private:
  int n_max_;
  int n_min_;
  std::string model_path_;
  unsigned int seed_;
};

#endif
