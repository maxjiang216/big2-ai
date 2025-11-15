// game_coordinator.h
#ifndef GAME_COORDINATOR_H
#define GAME_COORDINATOR_H

#include <arrow/api.h> // Apache Arrow C++ headers
#include <memory>
#include <random>
#include <string>
#include <vector>

class PlayerFactory;
class GameSimulator;
struct GameRecord;
class FeatureExtractor; // <-- forward declared

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
  GameCoordinator(
      std::shared_ptr<PlayerFactory> player_factory_p0,
      std::shared_ptr<PlayerFactory> player_factory_p1, int num_games,
      const std::string &output_path, int num_threads,
      unsigned int random_seed = std::random_device{}(),
      const std::string &log_path = "",
      std::vector<std::shared_ptr<FeatureExtractor>> game_level_features = {},
      std::vector<std::shared_ptr<FeatureExtractor>> turn_level_features = {});

  /**
   * @brief Run all self-play games
   */
  void run_all(const std::string &game_feature_out,
               const std::string &turn_feature_out);

  void export_features(const std::string &game_feature_out,
                       const std::string &turn_feature_out) const;

private:
  std::shared_ptr<PlayerFactory> _player_factory_p0;
  std::shared_ptr<PlayerFactory> _player_factory_p1;
  int _num_games;
  std::string _output_path;
  int _num_threads;
  std::string _log_path;
  unsigned int _rng_seed;

  // Feature extractor lists
  std::vector<std::shared_ptr<FeatureExtractor>> _game_level_features;
  std::vector<std::shared_ptr<FeatureExtractor>> _turn_level_features;

  // Store all game records here (1 per game)
  std::vector<GameRecord> _records;

  /**
   * @brief Simulate a single game using GameSimulator and RNG.
   * @param rng A thread-local random number generator.
   * @return The GameRecord for one completed game.
   */
  GameRecord simulate_single_game(std::mt19937 &rng);

  // Helpers
  void export_game_features(const std::string &out_file) const;
  void export_turn_features(const std::string &out_file) const;
};

#endif // GAME_COORDINATOR_H
