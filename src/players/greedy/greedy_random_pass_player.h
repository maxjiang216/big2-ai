#ifndef GREEDY_RANDOM_PASS_PLAYER_H
#define GREEDY_RANDOM_PASS_PLAYER_H

#include "greedy_player.h"

#include <random>
#include <vector>

// Plays greedy, but with probability `p` passes instead (if pass is legal).
// When it is the trick leader (pass not available), always plays greedy.
class GreedyRandomPassPlayer : public GreedyPlayer {
public:
  explicit GreedyRandomPassPlayer(float p, unsigned int seed)
      : p_(p), rng_(seed) {}

protected:
  Move select_move_impl() override {
    std::vector<int> legal = game_.get_legal_moves();

    bool pass_available = false;
    for (int m : legal) {
      if (Move(m).combination == Move::Combination::kPass) {
        pass_available = true;
        break;
      }
    }

    if (pass_available) {
      std::uniform_real_distribution<float> coin(0.0f, 1.0f);
      if (coin(rng_) < p_)
        return Move(kPASS);
    }

    return greedy_best(game_, legal);
  }

private:
  float p_;
  std::mt19937 rng_;
};

#endif
