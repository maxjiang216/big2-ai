#ifndef PLAYER_H
#define PLAYER_H

#include <vector>

#include "move.h"
#include "card.h"

/**
 * @brief Abstract base class for a player agent in Big 2.
 *
 * Defines the interface that all player implementations must provide.
 */
class Player {
public:
    virtual ~Player() = default;

    /**
     * @brief Receive the initial hand at the start of a game.
     * @param hand Vector of cards dealt to this player.
     */
    virtual void accept_deal(const std::vector<Card>& hand) = 0;

    /**
     * @brief Notify the player of the opponent's move.
     * @param move The move played by the opponent.
     */
    virtual void accept_opponent_move(const Move& move) = 0;

    /**
     * @brief Select the next move to play.
     * @return The move chosen by this player.
     */
    virtual Move select_move() = 0;
};

#endif // PLAYER_H
