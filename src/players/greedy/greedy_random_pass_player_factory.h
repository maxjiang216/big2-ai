#ifndef GREEDY_RANDOM_PASS_PLAYER_FACTORY_H
#define GREEDY_RANDOM_PASS_PLAYER_FACTORY_H

#include "greedy_random_pass_player.h"
#include "player_factory.h"

#include <memory>

class GreedyRandomPassPlayerFactory : public PlayerFactory {
public:
  explicit GreedyRandomPassPlayerFactory(float p, unsigned int seed = std::random_device{}())
      : p_(p), next_seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<GreedyRandomPassPlayer>(p_, next_seed_++);
  }

private:
  float p_;
  unsigned int next_seed_;
};

#endif
