#include "game_coordinator.h"
#include "greedy_player_factory.h"
#include "length_feature.h"
#include "next_player_feature.h"
#include "opponent_hand_size_feature.h"
#include "outcome_feature.h"
#include "player_hand_size_feature.h"
#include "random_player_factory.h"
#include "turn_outcome_feature.h"
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

// Helper to instantiate and register features
template <typename... FeatureTypes>
std::vector<std::shared_ptr<FeatureExtractor>> make_feature_list() {
  std::vector<std::shared_ptr<FeatureExtractor>> features;
  (void)std::initializer_list<int>{(
      features.push_back(std::make_shared<FeatureTypes>()),
      std::cout << "Added feature: " << std::make_shared<FeatureTypes>()->name()
                << std::endl,
      0)...};
  return features;
}

int main() {
  // Config
  // Crashes at 400k
  int num_games = 1000000;
  std::string output_path = "game_records.jsonl";
  int num_threads = std::max(1u, std::thread::hardware_concurrency());
  unsigned int seed = std::random_device{}();

  std::cout << "Running " << num_games << " self-play games with "
            << num_threads << " threads, output to '" << output_path
            << "'...\n";

  // Players
  auto factory0 = std::make_shared<GreedyPlayerFactory>();
  auto factory1 = std::make_shared<GreedyPlayerFactory>();

  // ----------- DEFINE FEATURES HERE -------------
  auto game_features =
      make_feature_list<OutcomeFeature, GameLengthExtractor
                        // Add more game-level feature classes here
                        >();

  auto turn_features =
      make_feature_list<TurnOutcomeFeature, NextPlayerFeature,
                        PlayerHandSizeFeature, OpponentHandSizeFeature
                        // Add more turn-level feature classes here
                        >();
  // ------------------------------------------------

  std::cout << "Game features: " << game_features.size() << std::endl;
  std::cout << "Turn features: " << turn_features.size() << std::endl;

  GameCoordinator coordinator(factory0, factory1, num_games, output_path,
                              num_threads, seed, "", game_features,
                              turn_features);

  std::cout << "Starting simulation..." << std::endl;
  coordinator.run_all("game_features.parquet", "turn_features.parquet");
  std::cout << "Simulation completed." << std::endl;

  std::cout << "Done.\n";
  return 0;
}