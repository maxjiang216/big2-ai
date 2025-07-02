#ifndef RANDOM_PLAYER_H
#define RANDOM_PLAYER_H

#include "player.h"
#include "move.h"
#include "partial_game.h"
#include <random>
#include <vector>

/**
 * @brief Player that selects uniformly random legal moves.
 */
class RandomPlayer : public Player {
public:
    /**
     * @brief Construct with non-deterministic seed.
     */
    RandomPlayer();

    /**
     * @brief Construct with a specific RNG seed.
     * @param seed Seed for the internal RNG.
     */
    explicit RandomPlayer(unsigned int seed);

    /**
     * @brief Receive and store the initial hand.
     */
    void accept_deal(const std::vector<Card>& hand, bool turn) override;

    /**
     * @brief Update internal state after opponent's move.
     */
    void accept_opponent_move(const Move& move) override;

    /**
     * @brief Select a legal move uniformly at random.
     * @return Chosen Move.
     */
    Move select_move() override;

private:
    std::mt19937 rng_;
    PartialGame game_;

};

#endif // RANDOM_PLAYER_H
