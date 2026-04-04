#ifndef PIMC_TREE_REDET_PLAYER_FACTORY_H
#define PIMC_TREE_REDET_PLAYER_FACTORY_H

#include "pimc_player.h"
#include "player_factory.h"

#include <cmath>
#include <memory>
#include <string>

// PIMC with tree rollout + per-opponent-turn hand resampling in rollouts.
class PimcTreeRedetPlayerFactory : public PlayerFactory {
public:
  PimcTreeRedetPlayerFactory(double param, int tree_depth, unsigned int seed)
      : num_samples_(std::max(1, static_cast<int>(std::round(param)))),
        model_path_("data/tree_model_d" + std::to_string(tree_depth) + ".txt"),
        seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<PimcTreePlayer>(num_samples_, model_path_, seed_++, true);
  }

private:
  int num_samples_;
  std::string model_path_;
  unsigned int seed_;
};

#endif
