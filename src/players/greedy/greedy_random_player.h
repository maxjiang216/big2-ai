#ifndef GREEDY_RANDOM_PLAYER_H
#define GREEDY_RANDOM_PLAYER_H

#include "greedy_player.h"

#include <random>
#include <vector>

// Plays greedy, but with probability `p` plays a uniformly-random legal move
// (including pass) instead.
class GreedyRandomPlayer : public GreedyPlayer {
public:
  explicit GreedyRandomPlayer(float p, unsigned int seed)
      : p_(p), rng_(seed) {}

protected:
  Move select_move_impl() override {
    std::vector<int> legal = game_.get_legal_moves();
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    if (coin(rng_) < p_) {
      std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
      return Move(legal[pick(rng_)]);
    }
    return greedy_best(game_, legal);
  }

private:
  float p_;
  std::mt19937 rng_;
};

#endif
