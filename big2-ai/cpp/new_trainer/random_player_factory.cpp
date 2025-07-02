#include "random_player_factory.h"

RandomPlayerFactory::RandomPlayerFactory(unsigned int seed)
    : next_seed_(seed) {}

std::unique_ptr<Player> RandomPlayerFactory::create_player() {
    // Construct a RandomPlayer with the next seed, then increment for uniqueness
    auto player = std::make_unique<RandomPlayer>(next_seed_);
    ++next_seed_;
    return player;
}
