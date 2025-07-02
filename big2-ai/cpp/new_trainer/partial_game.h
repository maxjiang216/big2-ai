#ifndef PARTIAL_GAME_H
#define PARTIAL_GAME_H

#include <vector>
#include <array>
#include "move.h"  // Represents a single play (type, rank, length, etc.)

/**
 * @brief Player's partial view of the Big 2 game state.
 * Contains only information available to the player:
 *  - Their own hand
 *  - How many cards opponent has left
 *  - Cards played/discarded so far
 *  - Whose turn it is
 *  - The last move played
 */
class PartialGame {
public:
    /**
     * @brief Construct an empty PartialGame state.
     */
    PartialGame();

    /**
     * @brief Initialize player state from hand, opponent card count, turn, and last move.
     * @param player_hand Initial hand (counts by rank, 0=3, ..., 12=2)
     */
    PartialGame(const std::array<int, 13>& player_hand, bool turn);

    void apply_move(const Move& move);

    std::vector<int> get_legal_moves() const;

private:
    bool turn; // true if player's turn
    std::array<int, 13> player_hand_{};    // This player's cards (by rank)
    int opponent_card_count_{16};          // Opponent's cards left
    std::array<int, 13> discard_pile_{};   // Cards played so far (by rank)
    Move last_move_{Move::Combination::kPass};
};

#endif // PARTIAL_GAME_H
