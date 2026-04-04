#ifndef PIMC_LINEAR_REDET_PLAYER_FACTORY_H
#define PIMC_LINEAR_REDET_PLAYER_FACTORY_H

#include "pimc_player.h"
#include "player_factory.h"

#include <cmath>
#include <memory>
#include <string>

// PIMC with Ridge linear rollout + per-opponent-turn hand resampling in rollouts.
class PimcLinearRedetPlayerFactory : public PlayerFactory {
public:
  PimcLinearRedetPlayerFactory(double param, unsigned int seed)
      : num_samples_(std::max(1, static_cast<int>(std::round(param)))),
        seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<PimcLinearPlayer>(num_samples_, model_path_, seed_++, true);
  }

private:
  int num_samples_;
  std::string model_path_{"data/linear_rollout_w.txt"};
  unsigned int seed_;
};

#endif
