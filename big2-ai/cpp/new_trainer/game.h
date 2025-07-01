// game.h
#ifndef GAME_H
#define GAME_H

#include <random>
#include <vector>
#include <array>

#include "move.h"  // Represents a single play (type, rank, length, etc.)

/**
 * @brief Internal representation of a Big 2 game state.
 * Handles deck, hands, current trick, initiative, and end conditions.
 */
class Game {
public:
    /**
     * @brief Construct a new Game (empty state).
     */
    Game();

    /**
     * @brief Shuffle the deck and deal cards to both players.
     * @param rng Random number generator.
     */
    void shuffle_deal(std::mt19937 &rng);

    /**
     * @brief Get the index (0 or 1) of the player whose turn it is.
     */
    int current_player() const;

    /**
     * @brief Check if the game has ended (one player out of cards).
     */
    bool is_over() const;

    /**
     * @brief Apply a player's move to the game state.
     * @param move The move chosen by the current player.
     */
    void apply_move(const Move &move);

    /**
     * @brief Get the winner (0 or 1) after the game ends.
     * Returns the index of the player who first emptied their hand.
     */
    int get_winner() const;

    /**
     * @brief Get a copy of a player's hand (rank counts) for initialization.
     * @param player 0 or 1.
     * @return vector of counts indexed by rank (0=3, ..., 12=2).
     */
    std::vector<int> get_player_hand(int player) const;

private:
    std::array<std::array<int,13>,2> hands_;
    std::array<int,13>                discard_pile_;
    int                             current_player_;
    Move last_move{Move::Combination::kPass};

    friend std::ostream& operator<<(std::ostream& os, const Game& game);
};

#endif // GAME_H
