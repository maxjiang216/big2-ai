#include "game_coordinator.h"
#include "outcome_feature.h"
#include "player_hand_size_feature.h"
#include "random_player_factory.h"
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

int main(int argc, char **argv) {
  // Configuration (could be extended to parse argc/argv)
  int num_games = 1000;
  std::string output_path = "game_records.jsonl";
  int num_threads = 1;
  unsigned int seed = std::random_device{}();

  std::cout << "Running " << num_games << " self-play games with "
            << num_threads << " threads, output to '" << output_path
            << "'...\n";

  // Create two RandomPlayer factories with distinct seeds
  auto factory0 = std::make_shared<RandomPlayerFactory>(seed);
  auto factory1 = std::make_shared<RandomPlayerFactory>(seed + 12345);

  // Instantiate feature extractors
  std::vector<std::shared_ptr<FeatureExtractor>> game_features;
  auto outcome_feature = std::make_shared<OutcomeFeature>();
  if (outcome_feature) {
    game_features.push_back(outcome_feature);
    std::cout << "Added OutcomeFeature: " << outcome_feature->name()
              << std::endl;
  } else {
    std::cerr << "Failed to create OutcomeFeature!" << std::endl;
  }

  std::vector<std::shared_ptr<FeatureExtractor>> turn_features;
  auto hand_size_feature = std::make_shared<PlayerHandSizeFeature>();
  if (hand_size_feature) {
    turn_features.push_back(hand_size_feature);
    std::cout << "Added PlayerHandSizeFeature: " << hand_size_feature->name()
              << std::endl;
  } else {
    std::cerr << "Failed to create PlayerHandSizeFeature!" << std::endl;
  }

  std::cout << "Game features: " << game_features.size() << std::endl;
  std::cout << "Turn features: " << turn_features.size() << std::endl;

  // Initialize coordinator with feature extractors
  GameCoordinator coordinator(factory0, factory1, num_games, output_path,
                              num_threads, seed,
                              "",            // log_path
                              game_features, // game-level features
                              turn_features  // turn-level features
  );

  // Run self-play
  std::cout << "Starting simulation..." << std::endl;
  coordinator.run_all();
  std::cout << "Simulation completed." << std::endl;

  // Export features to Parquet files
  std::cout << "Starting feature export..." << std::endl;
  try {
    coordinator.export_features("game_features.parquet",
                                "turn_features.parquet");
    std::cout << "Feature export completed successfully." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Exception during feature export: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception during feature export!" << std::endl;
    return 1;
  }

  std::cout << "Done.\n";
  return 0;
}