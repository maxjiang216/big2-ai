// datagen/generate_data.cpp
#include "game_coordinator.h"
#include "players/greedy/greedy_player_factory.h"
#include "players/random/random_player_factory.h"
#include "features/game_level/outcome_feature.h"
#include "features/game_level/length_feature.h"
#include "features/turn_level/turn_outcome_feature.h"
#include "features/turn_level/next_player_feature.h"
#include "features/turn_level/player_hand_size_feature.h"
#include "features/turn_level/opponent_hand_size_feature.h"

#include <iostream>
#include <sstream>  // ADDED: needed for std::istringstream
#include <string>
#include <memory>
#include <random>
#include <thread>

// ============================================================================
// PLAYER REGISTRY - Add new players here
// ============================================================================
std::shared_ptr<PlayerFactory> create_player_factory(const std::string& name) {
  if (name == "random") {
    return std::make_shared<RandomPlayerFactory>();
  } else if (name == "greedy") {
    return std::make_shared<GreedyPlayerFactory>();
  }
  // Add more players here as you implement them:
  // else if (name == "mcts") {
  //   return std::make_shared<MCTSPlayerFactory>();
  // }
  
  std::cerr << "Error: Unknown player type '" << name << "'\n";
  std::cerr << "Available players: random, greedy\n";
  return nullptr;
}

// ============================================================================
// FEATURE REGISTRY - Add new features here
// ============================================================================
std::shared_ptr<FeatureExtractor> create_feature(const std::string& name) {
  // Game-level features
  if (name == "outcome") return std::make_shared<OutcomeFeature>();
  if (name == "length") return std::make_shared<GameLengthExtractor>();
  
  // Turn-level features
  if (name == "turn_outcome") return std::make_shared<TurnOutcomeFeature>();
  if (name == "next_player") return std::make_shared<NextPlayerFeature>();
  if (name == "player_hand_size") return std::make_shared<PlayerHandSizeFeature>();
  if (name == "opponent_hand_size") return std::make_shared<OpponentHandSizeFeature>();
  
  std::cerr << "Warning: Unknown feature '" << name << "', skipping\n";
  return nullptr;
}

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [options]\n"
            << "Options:\n"
            << "  --player <name>           Player type (required, used for both sides)\n"
            << "  --games <num>             Number of games to simulate (required)\n"
            << "  --output <path>           Output path prefix (required)\n"
            << "  --game-features <f1,f2>   Comma-separated game-level features\n"
            << "  --turn-features <f1,f2>   Comma-separated turn-level features\n"
            << "  --seed <num>              Random seed (default: random)\n"
            << "  --threads <num>           Number of threads (default: hardware max - 2)\n"
            << "  --help                    Show this help\n"
            << "\nExample:\n"
            << "  " << program_name << " --player greedy --games 10000 \\\n"
            << "    --output data/greedy_10k \\\n"
            << "    --game-features outcome,length \\\n"
            << "    --turn-features turn_outcome,player_hand_size\n";
}

std::vector<std::string> split_string(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream token_stream(str);
  while (std::getline(token_stream, token, delimiter)) {
    if (!token.empty()) {
      tokens.push_back(token);
    }
  }
  return tokens;
}

int main(int argc, char** argv) {
  // Parse command-line arguments
  std::string player_name;
  int num_games = 0;
  std::string output_path;
  std::vector<std::string> game_feature_names;
  std::vector<std::string> turn_feature_names;
  unsigned int seed = std::random_device{}();
  int num_threads = std::max(1u, std::thread::hardware_concurrency() - 2);

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--player" && i + 1 < argc) {
      player_name = argv[++i];
    } else if (arg == "--games" && i + 1 < argc) {
      num_games = std::stoi(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--game-features" && i + 1 < argc) {
      game_feature_names = split_string(argv[++i], ',');
    } else if (arg == "--turn-features" && i + 1 < argc) {
      turn_feature_names = split_string(argv[++i], ',');
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoul(argv[++i]);
    } else if (arg == "--threads" && i + 1 < argc) {
      num_threads = std::stoi(argv[++i]);
    } else {
      std::cerr << "Error: Unknown argument '" << arg << "'\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  // Validate required arguments
  if (player_name.empty() || num_games == 0 || output_path.empty()) {
    std::cerr << "Error: Missing required arguments\n\n";
    print_usage(argv[0]);
    return 1;
  }

  // Create player factories
  auto factory = create_player_factory(player_name);
  if (!factory) {
    return 1;
  }

  // Create features
  std::vector<std::shared_ptr<FeatureExtractor>> game_features;
  for (const auto& name : game_feature_names) {
    auto feature = create_feature(name);
    if (feature) {
      game_features.push_back(feature);
      std::cout << "Added game feature: " << name << "\n";
    }
  }

  std::vector<std::shared_ptr<FeatureExtractor>> turn_features;
  for (const auto& name : turn_feature_names) {
    auto feature = create_feature(name);
    if (feature) {
      turn_features.push_back(feature);
      std::cout << "Added turn feature: " << name << "\n";
    }
  }

  // Print configuration
  std::cout << "\n=== Data Generation Configuration ===\n";
  std::cout << "Player: " << player_name << " vs " << player_name << "\n";
  std::cout << "Games: " << num_games << "\n";
  std::cout << "Output: " << output_path << "_game.parquet, " 
            << output_path << "_turn.parquet\n";
  std::cout << "Seed: " << seed << "\n";
  std::cout << "Threads: " << num_threads << "\n";
  std::cout << "Game features: " << game_features.size() << "\n";
  std::cout << "Turn features: " << turn_features.size() << "\n";
  std::cout << "=====================================\n\n";

  // Run simulation
  GameCoordinator coordinator(
    factory,  // player 0
    factory,  // player 1 (self-play)
    num_games,
    "",       // unused old output_path parameter
    num_threads,
    seed,
    "",       // no log file
    game_features,
    turn_features
  );

  std::string game_output = output_path + "_game.parquet";
  std::string turn_output = output_path + "_turn.parquet";

  // run_all handles batching and exporting internally
  coordinator.run_all(game_output, turn_output);

  std::cout << "\n✓ Data generation complete!\n";
  std::cout << "Game features: " << game_output << "\n";
  std::cout << "Turn features: " << turn_output << "\n";

  return 0;
}