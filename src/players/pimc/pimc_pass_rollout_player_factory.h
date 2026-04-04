#ifndef PIMC_PASS_ROLLOUT_PLAYER_FACTORY_H
#define PIMC_PASS_ROLLOUT_PLAYER_FACTORY_H

#include "pimc_player.h"
#include "player_factory.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

// PIMC with greedy_hand_eval + pass-aware Ridge rollout. param = N determinizations.
class PimcPassRolloutPlayerFactory : public PlayerFactory {
public:
  PimcPassRolloutPlayerFactory(double param, unsigned int seed,
                               const std::string &model_path = "data/pass_ridge_w.txt")
      : n_(std::max(1, static_cast<int>(std::round(param == 0.0 ? 10.0 : param)))),
        seed_(seed),
        path_(model_path) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<PimcPassRolloutPlayer>(n_, path_, seed_++);
  }

private:
  int n_;
  unsigned int seed_;
  std::string path_;
};

// Adaptive variant: n_min = max(3, n_max/10), delta = 0.05.
class PimcPassRolloutAdaptivePlayerFactory : public PlayerFactory {
public:
  PimcPassRolloutAdaptivePlayerFactory(double param, unsigned int seed,
                                       const std::string &model_path = "data/pass_ridge_w.txt")
      : n_max_(std::max(1, static_cast<int>(std::round(param == 0.0 ? 10.0 : param)))),
        n_min_(std::min(std::max(3, n_max_ / 10), n_max_)),
        seed_(seed),
        path_(model_path) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<PimcPassRolloutPlayer>(n_max_, n_min_, 0.05, path_, seed_++);
  }

private:
  int n_max_;
  int n_min_;
  unsigned int seed_;
  std::string path_;
};

#endif
