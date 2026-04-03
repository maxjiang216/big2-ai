#ifndef RANDOM_PLAYER_FACTORY_H
#define RANDOM_PLAYER_FACTORY_H

#include "player_factory.h"
#include "random_player.h"

#include <memory>
#include <random>

class RandomPlayerFactory : public PlayerFactory {
public:
  explicit RandomPlayerFactory(unsigned int seed = std::random_device{}())
      : next_seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    auto player = std::make_unique<RandomPlayer>(next_seed_);
    ++next_seed_;
    return player;
  }

private:
  unsigned int next_seed_;
};

#endif
