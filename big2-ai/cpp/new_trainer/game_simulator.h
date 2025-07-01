// game_simulator.h
#ifndef GAME_SIMULATOR_H
#define GAME_SIMULATOR_H

#include <memory>
#include <random>

#include "player.h"
#include "game.h"      // Internal game state representation
#include "game_record.h"

/**
 * @brief Simulates a single Big 2 game between two players using an internal Game object.
 */
class GameSimulator {
public:
    /**
     * @brief Construct a new GameSimulator.
     * @param player0    The Agent playing as Player 0 (first move).
     * @param player1    The Agent playing as Player 1.
     * @param rng        A random number generator for dealing and randomness.
     */
    GameSimulator(std::unique_ptr<Player> player0,
                  std::unique_ptr<Player> player1,
                  std::mt19937 &rng);

    /**
     * @brief Run the game to completion and return its record.
     * @return GameRecord containing the full move sequence and outcome.
     */
    GameRecord run();

private:
    std::unique_ptr<Player> _player0;
    std::unique_ptr<Player> _player1;
    std::mt19937 &_rng;
    Game _game;  // Internal game state, handles deck, hands, tricks, scoring
    GameRecord _record;

    /**
     * @brief Initialize the game: shuffle, deal, determine initiative.
     */
    void initialize_game();

    /**
     * @brief Main loop: alternate between players until game over.
     */
    void play_loop();

    /**
     * @brief Ask the current player to select and play a move, update game state and record.
     */
    void play_turn();
};

#endif // GAME_SIMULATOR_H
