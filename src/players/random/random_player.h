#ifndef RANDOM_PLAYER_H
#define RANDOM_PLAYER_H

#include "move.h"
#include "player.h"

#include <random>
#include <stdexcept>
#include <vector>

// Inherits Player::select_move(), which runs the tablebase (guaranteed-optimal
// trivial moves) before calling select_move_impl(). Randomness applies only when
// the tablebase has no prescribed move — same as any other concrete player.
class RandomPlayer : public Player {
public:
  RandomPlayer() : rng_(std::random_device{}()) {}

  explicit RandomPlayer(unsigned int seed) : rng_(seed) {}

protected:
  Move select_move_impl() override {
    std::vector<int> legal = game_.get_legal_moves();
    if (legal.empty())
      throw std::runtime_error("No legal moves available.");
    std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
    return Move(legal[dist(rng_)]);
  }

private:
  std::mt19937 rng_;
};

#endif
