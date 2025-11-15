#ifndef GREEDY_PLAYER_FACTORY_H
#define GREEDY_PLAYER_FACTORY_H

#include "greedy_player.h"
#include "player_factory.h"
#include <memory>

/**
 * @brief Concrete factory for creating GreedyPlayer instances.
 *
 * Each call to create_player produces a new GreedyPlayer.
 */
class GreedyPlayerFactory : public PlayerFactory {
public:
  GreedyPlayerFactory() = default;

  /**
   * @brief Create a new GreedyPlayer.
   * @return unique_ptr<Player>
   */
  std::unique_ptr<Player> create_player() override {
    return std::make_unique<GreedyPlayer>();
  }
};

#endif // GREEDY_PLAYER_FACTORY_H
