// game_coordinator.h
#ifndef GAME_COORDINATOR_H
#define GAME_COORDINATOR_H

#include <memory>
#include <random>
#include <string>
#include <vector>

class PlayerFactory;
class GameSimulator;
struct GameRecord;

class GameCoordinator {
public:
  /**
   * @brief Construct a new GameCoordinator.
   * @param player_factory_p0 Factory for Player 0 (first player) Agents.
   * @param player_factory_p1 Factory for Player 1 (second player) Agents.
   * @param num_games Number of self-play games to simulate.
   * @param output_path Path to write the collected GameRecord data.
   * @param num_threads Number of concurrent threads to use.
   * @param random_seed Optional RNG seed (default: random_device).
   */
  GameCoordinator(std::shared_ptr<PlayerFactory> player_factory_p0,
                  std::shared_ptr<PlayerFactory> player_factory_p1,
                  int num_games, const std::string &output_path,
                  int num_threads,
                  unsigned int random_seed = std::random_device{}());

  /**
   * @brief Run all self-play games
   */
  void run_all();

  /**
   * @brief Write all collected GameRecords to disk (e.g. JSON/CSV).
   * Must be called after run_all().
   */
  // void export_data() const;

private:
  std::shared_ptr<PlayerFactory> _player_factory_p0;
  std::shared_ptr<PlayerFactory> _player_factory_p1;
  int _num_games;
  std::string _output_path;
  int _num_threads;
  unsigned int _rng_seed;

  /**
   * @brief Simulate a single game using GameSimulator and RNG.
   * @param rng A thread-local random number generator.
   * @return The GameRecord for one completed game.
   */
  GameRecord simulate_single_game(std::mt19937 &rng);
};

#endif // GAME_COORDINATOR_H
