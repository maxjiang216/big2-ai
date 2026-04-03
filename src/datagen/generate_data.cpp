#include "feature_registry.h"
#include "game_coordinator.h"
#include "greedy/greedy_player_factory.h"
#include "random/random_player_factory.h"
#include "player_factory.h"

#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Helpers
// ============================================================================

static std::vector<std::string> split_csv(const std::string &s) {
  std::vector<std::string> tokens;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (!tok.empty()) tokens.push_back(tok);
  return tokens;
}

static std::shared_ptr<PlayerFactory> make_factory(const std::string &name,
                                                    unsigned int seed) {
  if (name == "random") return std::make_shared<RandomPlayerFactory>(seed);
  if (name == "greedy") return std::make_shared<GreedyPlayerFactory>();
  std::cerr << "Error: unknown player '" << name << "'\n";
  return nullptr;
}

static void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "Options:\n"
      << "  --player <name>            Player type for both sides (required)\n"
      << "  --games <N>                Number of games (required)\n"
      << "  --output <path>            Output path prefix (required)\n"
      << "                             Writes <path>_game.parquet and <path>_turn.parquet\n"
      << "  --game-features <f1,f2>    Comma-separated game-level feature names\n"
      << "  --turn-features <f1,f2>    Comma-separated turn-level feature names\n"
      << "  --seed <S>                 RNG seed (default: random)\n"
      << "  --threads <T>              Threads (default: hardware - 2)\n"
      << "\nAvailable players: random, greedy\n"
      << "\nExample:\n"
      << "  " << prog
      << " --player random --games 100000 --output data/random_100k"
         " --game-features outcome,length,tb_hits,tb_case1,tb_case2,"
         "tb_forced_seq_len,tb_opp1_table_straight"
         " --turn-features turn_outcome,player_hand_size\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  std::string player_name;
  int num_games = 0;
  std::string output_path;
  std::vector<std::string> game_feat_names, turn_feat_names;
  unsigned int seed = std::random_device{}();
  int num_threads = std::max(1u, std::thread::hardware_concurrency() - 2);

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    else if (arg == "--player"        && i + 1 < argc) player_name     = argv[++i];
    else if (arg == "--games"         && i + 1 < argc) num_games       = std::stoi(argv[++i]);
    else if (arg == "--output"        && i + 1 < argc) output_path     = argv[++i];
    else if (arg == "--game-features" && i + 1 < argc) game_feat_names = split_csv(argv[++i]);
    else if (arg == "--turn-features" && i + 1 < argc) turn_feat_names = split_csv(argv[++i]);
    else if (arg == "--seed"          && i + 1 < argc) seed            = std::stoul(argv[++i]);
    else if (arg == "--threads"       && i + 1 < argc) num_threads     = std::stoi(argv[++i]);
    else { std::cerr << "Unknown argument: " << arg << "\n"; print_usage(argv[0]); return 1; }
  }

  if (player_name.empty() || num_games <= 0 || output_path.empty()) {
    std::cerr << "Error: --player, --games, and --output are required\n\n";
    print_usage(argv[0]);
    return 1;
  }

  // Build feature objects.
  std::vector<std::shared_ptr<FeatureExtractor>> game_features, turn_features;
  for (const auto &n : game_feat_names) {
    auto f = create_feature(n);
    if (f) { game_features.push_back(f); std::cout << "Game feature: " << n << "\n"; }
  }
  for (const auto &n : turn_feat_names) {
    auto f = create_feature(n);
    if (f) { turn_features.push_back(f); std::cout << "Turn feature: " << n << "\n"; }
  }

  auto factory = make_factory(player_name, seed);
  if (!factory) return 1;

  std::cout << "\n=== Data Generation ===\n"
            << "Player:   " << player_name << " vs " << player_name << "\n"
            << "Games:    " << num_games << "\n"
            << "Output:   " << output_path << "_game.parquet\n"
            << "          " << output_path << "_turn.parquet\n"
            << "Seed:     " << seed << "\n"
            << "Threads:  " << num_threads << "\n\n";

  GameCoordinator coordinator(
      factory, factory,
      num_games, "",
      num_threads, seed, "",
      game_features, turn_features);

  coordinator.run_all(output_path + "_game.parquet", output_path + "_turn.parquet");

  std::cout << "\nDone.\n";
  return 0;
}
