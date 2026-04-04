#ifndef PIMC_REDET_PLAYER_FACTORY_H
#define PIMC_REDET_PLAYER_FACTORY_H

#include "pimc_player.h"
#include "player_factory.h"

#include <cmath>
#include <memory>

// PIMC with greedy rollout; opponent hand is re-sampled at each opponent turn
// inside ``policy_rollout`` (see ``pimc_player.h``).
//
// ``param`` — determinization count N (default 10 if 0).
class PimcRedetPlayerFactory : public PlayerFactory {
public:
  explicit PimcRedetPlayerFactory(double param, unsigned int seed)
      : num_samples_(std::max(1, static_cast<int>(std::round(param)))),
        seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<PimcPlayer>(num_samples_, seed_++, true);
  }

private:
  int num_samples_;
  unsigned int seed_;
};

#endif
