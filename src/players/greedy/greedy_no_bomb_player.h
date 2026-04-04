#ifndef GREEDY_NO_BOMB_PLAYER_H
#define GREEDY_NO_BOMB_PLAYER_H

#include "greedy_player.h"

#include <random>
#include <vector>

// Plays greedy, but suppresses bomb plays with probability (1 - p_bomb).
// When suppressed:
//   1. Pass if available.
//   2. Otherwise play the greedy-best non-bomb move.
//   3. If no non-bomb moves exist, play the bomb anyway.
class GreedyNoBombPlayer : public GreedyPlayer {
public:
  explicit GreedyNoBombPlayer(float p_bomb, unsigned int seed)
      : p_bomb_(p_bomb), rng_(seed) {}

protected:
  Move select_move_impl() override {
    std::vector<int> legal = game_.get_legal_moves();
    Move chosen = greedy_best(game_, legal);

    if (chosen.combination != Move::Combination::kBomb)
      return chosen;

    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    if (coin(rng_) < p_bomb_)
      return chosen;

    // Suppress the bomb: prefer pass, then best non-bomb, then bomb as fallback.
    for (int m : legal) {
      if (Move(m).combination == Move::Combination::kPass)
        return Move(kPASS);
    }

    std::vector<int> no_bomb;
    no_bomb.reserve(legal.size());
    for (int m : legal) {
      if (Move(m).combination != Move::Combination::kBomb)
        no_bomb.push_back(m);
    }
    if (!no_bomb.empty())
      return greedy_best(game_, no_bomb);

    return chosen;
  }

private:
  float p_bomb_;
  std::mt19937 rng_;
};

#endif
