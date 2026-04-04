#ifndef PIMC_ADAPTIVE_PLAYER_FACTORY_H
#define PIMC_ADAPTIVE_PLAYER_FACTORY_H

#include "pimc_player.h"
#include "player_factory.h"

#include <algorithm>
#include <cmath>
#include <memory>

// Adaptive PIMC with greedy rollout after tablebase.
//
// `param` — max determinizations n_max (default 10 if 0).
// n_min   = max(3, n_max/10), clamped to n_max; delta = 0.05 (Bonferroni / M).
class PimcAdaptivePlayerFactory : public PlayerFactory {
public:
  explicit PimcAdaptivePlayerFactory(double param, unsigned int seed)
      : n_max_(std::max(1, static_cast<int>(std::round(param)))),
        n_min_(std::min(std::max(3, n_max_ / 10), n_max_)), seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<PimcPlayer>(n_max_, n_min_, 0.05, seed_++);
  }

private:
  int n_max_;
  int n_min_;
  unsigned int seed_;
};

#endif
