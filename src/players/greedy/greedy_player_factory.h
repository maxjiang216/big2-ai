#ifndef GREEDY_PLAYER_FACTORY_H
#define GREEDY_PLAYER_FACTORY_H

#include "greedy_player.h"
#include "player_factory.h"

#include <memory>

class GreedyPlayerFactory : public PlayerFactory {
public:
  GreedyPlayerFactory() = default;

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<GreedyPlayer>();
  }
};

#endif
