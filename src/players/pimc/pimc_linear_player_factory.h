#ifndef PIMC_LINEAR_PLAYER_FACTORY_H
#define PIMC_LINEAR_PLAYER_FACTORY_H

#include "pimc_player.h"
#include "player_factory.h"

#include <cmath>
#include <memory>
#include <string>

// PIMC with Ridge linear rollout after tablebase.
//
// ``param`` — number of determinization samples N (default 10 if 0).
// Weights: ``data/linear_rollout_w.txt`` (from ``analysis/train_linear_rollout.py``).
class PimcLinearPlayerFactory : public PlayerFactory {
public:
  PimcLinearPlayerFactory(double param, unsigned int seed)
      : num_samples_(std::max(1, static_cast<int>(std::round(param)))),
        seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<PimcLinearPlayer>(num_samples_, model_path_, seed_++);
  }

private:
  int num_samples_;
  std::string model_path_{"data/linear_rollout_w.txt"};
  unsigned int seed_;
};

#endif
