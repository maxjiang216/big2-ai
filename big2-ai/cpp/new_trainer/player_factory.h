#ifndef PLAYER_FACTORY_H
#define PLAYER_FACTORY_H

#include <memory>

#include "player.h"

/**
 * @brief Abstract factory for creating Player instances.
 *
 * PlayerFactory allows instantiation of new Player agents for self-play.
 */
class PlayerFactory {
public:
    virtual ~PlayerFactory() = default;

    /**
     * @brief Create a new Player instance.
     * @return A unique_ptr to the newly constructed Player.
     */
    virtual std::unique_ptr<Player> create_player() = 0;
};

#endif // PLAYER_FACTORY_H
