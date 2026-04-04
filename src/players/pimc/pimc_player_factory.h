#ifndef PIMC_PLAYER_FACTORY_H
#define PIMC_PLAYER_FACTORY_H

#include "pimc_player.h"
#include "player_factory.h"

#include <cmath>
#include <memory>

// Factory for PimcPlayer.
//
// `param` is the number of determinization samples N.
// `seed`  is the RNG seed for the player's hand sampler.
//
// Example round-robin config entry:
//   {"type": "pimc", "param": 10, "label": "pimc_n10"}
class PimcPlayerFactory : public PlayerFactory {
public:
  explicit PimcPlayerFactory(double param, unsigned int seed)
      : num_samples_(std::max(1, static_cast<int>(std::round(param)))),
        seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<PimcPlayer>(num_samples_, seed_++);
  }

private:
  int num_samples_;
  unsigned int seed_;
};

#endif
