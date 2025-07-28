#ifndef RANDOM_PLAYER_FACTORY_H
#define RANDOM_PLAYER_FACTORY_H

#include "player_factory.h"
#include "random_player.h"
#include <random>

/**
 * @brief Concrete factory for creating RandomPlayer instances.
 *
 * Each call to create_player produces a new RandomPlayer with its own RNG seed.
 */
class RandomPlayerFactory : public PlayerFactory {
public:
  /**
   * @brief Construct a RandomPlayerFactory.
   * @param seed Optional base seed; if not provided, std::random_device is
   * used.
   */
  explicit RandomPlayerFactory(unsigned int seed = std::random_device{}());

  /**
   * @brief Create a new RandomPlayer with a unique seed.
   * @return unique_ptr<Player>
   */
  std::unique_ptr<Player> create_player() override;

private:
  unsigned int next_seed_;
};

#endif // RANDOM_PLAYER_FACTORY_H
