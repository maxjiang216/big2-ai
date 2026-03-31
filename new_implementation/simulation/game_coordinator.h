#ifndef GAME_COORDINATOR_H
#define GAME_COORDINATOR_H

#include "game_record.h"
#include "feature_extractor.h"

#include "../players/player_factory.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

class GameCoordinator {
public:
  GameCoordinator(
      std::shared_ptr<PlayerFactory> player_factory_p0,
      std::shared_ptr<PlayerFactory> player_factory_p1, int num_games,
      const std::string &output_path, int num_threads,
      unsigned int random_seed = std::random_device{}(),
      const std::string &log_path = "",
      std::vector<std::shared_ptr<FeatureExtractor>> game_level_features = {},
      std::vector<std::shared_ptr<FeatureExtractor>> turn_level_features = {});

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

  std::vector<std::shared_ptr<FeatureExtractor>> _game_level_features;
  std::vector<std::shared_ptr<FeatureExtractor>> _turn_level_features;

  std::vector<GameRecord> _records;

  GameRecord simulate_single_game(std::mt19937 &rng);
};

#endif
