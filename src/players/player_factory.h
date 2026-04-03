#ifndef PLAYER_FACTORY_H
#define PLAYER_FACTORY_H

#include "player.h"

#include <memory>

class PlayerFactory {
public:
  virtual ~PlayerFactory() = default;

  virtual std::unique_ptr<Player> create_player() = 0;
};

#endif
