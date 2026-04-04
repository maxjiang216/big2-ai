#ifndef GREEDY_NO_BOMB_PLAYER_FACTORY_H
#define GREEDY_NO_BOMB_PLAYER_FACTORY_H

#include "greedy_no_bomb_player.h"
#include "player_factory.h"

#include <memory>

class GreedyNoBombPlayerFactory : public PlayerFactory {
public:
  explicit GreedyNoBombPlayerFactory(float p_bomb, unsigned int seed = std::random_device{}())
      : p_bomb_(p_bomb), next_seed_(seed) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<GreedyNoBombPlayer>(p_bomb_, next_seed_++);
  }

private:
  float p_bomb_;
  unsigned int next_seed_;
};

#endif
