#ifndef GREEDY_RANDOM_PLAYER_FACTORY_H
#define GREEDY_RANDOM_PLAYER_FACTORY_H

#include "greedy_random_player.h"
#include "player_factory.h"

#include <memory>

class GreedyRandomPlayerFactory : public PlayerFactory {
public:
  explicit GreedyRandomPlayerFactory(float p, unsigned int seed = std::random_device{}())
      : p_(p), next_seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<GreedyRandomPlayer>(p_, next_seed_++);
  }

private:
  float p_;
  unsigned int next_seed_;
};

#endif
