#ifndef RANDOM_PLAYER_FACTORY_H
#define RANDOM_PLAYER_FACTORY_H

#include "player_factory.h"
#include "random_player.h"
#include <memory>
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
  explicit RandomPlayerFactory(unsigned int seed = std::random_device{}())
      : next_seed_(seed) {}

  /**
   * @brief Create a new RandomPlayer with a unique seed.
   * @return unique_ptr<Player>
   */
  std::unique_ptr<Player> create_player() override {
    // Construct a RandomPlayer with the next seed, then increment for
    // uniqueness
    auto player = std::make_unique<RandomPlayer>(next_seed_);
    ++next_seed_;
    return player;
  }

private:
  unsigned int next_seed_;
};

#endif // RANDOM_PLAYER_FACTORY_H
