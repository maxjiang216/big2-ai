#ifndef PLAYER_FACTORY_REGISTRY_H
#define PLAYER_FACTORY_REGISTRY_H

#include "player_factory.h"
#include "greedy/greedy_player_factory.h"
#include "greedy/greedy_random_player_factory.h"
#include "greedy/greedy_random_pass_player_factory.h"
#include "greedy/greedy_no_bomb_player_factory.h"
#include "random/random_player_factory.h"

#include <iostream>
#include <memory>
#include <string>

// Creates a PlayerFactory by name. `param` is used by parameterized variants;
// ignored for "random" and "greedy". `seed` is the RNG seed for stochastic players.
//
// Supported names:
//   "random"             — uniform random legal move
//   "greedy"             — greedy (no randomness)
//   "greedy_random"      — greedy with prob param of playing a random move
//   "greedy_random_pass" — greedy with prob param of passing instead
//   "greedy_no_bomb"     — greedy; plays a bomb only with probability param
inline std::shared_ptr<PlayerFactory> make_player_factory(const std::string &name,
                                                           double param,
                                                           unsigned int seed) {
  if (name == "random")
    return std::make_shared<RandomPlayerFactory>(seed);
  if (name == "greedy")
    return std::make_shared<GreedyPlayerFactory>();
  if (name == "greedy_random")
    return std::make_shared<GreedyRandomPlayerFactory>(static_cast<float>(param), seed);
  if (name == "greedy_random_pass")
    return std::make_shared<GreedyRandomPassPlayerFactory>(static_cast<float>(param), seed);
  if (name == "greedy_no_bomb")
    return std::make_shared<GreedyNoBombPlayerFactory>(static_cast<float>(param), seed);

  std::cerr << "Error: unknown player '" << name << "'\n"
            << "Available: random, greedy, greedy_random, greedy_random_pass, greedy_no_bomb\n";
  return nullptr;
}

#endif
