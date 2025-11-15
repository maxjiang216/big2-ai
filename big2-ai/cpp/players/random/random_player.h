#ifndef RANDOM_PLAYER_H
#define RANDOM_PLAYER_H

#include "move.h"
#include "partial_game.h"
#include "player.h"
#include <array>
#include <random>
#include <stdexcept>
#include <vector>

/**
 * @brief Player that selects uniformly random legal moves.
 */
class RandomPlayer : public Player {
public:
  /**
   * @brief Construct with non-deterministic seed.
   */
  RandomPlayer() : rng_(std::random_device{}()), game_() {}

  /**
   * @brief Construct with a specific RNG seed.
   * @param seed Seed for the internal RNG.
   */
  explicit RandomPlayer(unsigned int seed) : rng_(seed), game_() {}

  /**
   * @brief Receive and store the initial hand.
   */
  void accept_deal(std::array<int, 13> hand, int turn) override {
    game_ = PartialGame(hand, turn);
  }

  /**
   * @brief Update internal state after opponent's move.
   */
  void accept_opponent_move(const Move &move) override {
    game_.apply_move(move);
  }

  /**
   * @brief Select a legal move uniformly at random.
   * @return Chosen Move.
   */
  Move select_move() override {
    std::vector<int> legal = game_.get_legal_moves();
    if (legal.empty())
      throw std::runtime_error("No legal moves available.");
    std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
    size_t idx = dist(rng_);
    int move_id = legal[idx];
    Move m(move_id);
    game_.apply_move(m);
    return m;
  }

private:
  std::mt19937 rng_;
  PartialGame game_;
};

#endif // RANDOM_PLAYER_H
