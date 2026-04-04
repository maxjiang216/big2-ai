#include "feature_registry.h"
#include "game_coordinator.h"
#include "samples_md.h"
#include "player_factory_registry.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
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


static std::string format_eta(double seconds) {
  if (seconds < 0 || !std::isfinite(seconds))
    return "?";
  int s = static_cast<int>(seconds + 0.5);
  if (s < 60)
    return std::to_string(s) + "s";
  int m = s / 60;
  s %= 60;
  if (m < 60)
    return std::to_string(m) + "m " + std::to_string(s) + "s";
  int h = m / 60;
  m %= 60;
  return std::to_string(h) + "h " + std::to_string(m) + "m";
}

static void print_progress_line(int done, int total, double elapsed_sec) {
  if (total <= 0)
    return;
  int pct = static_cast<int>(100.0 * done / total);
  int bar_w = 24;
  int filled = static_cast<int>(bar_w * done / static_cast<double>(total));
  std::string bar;
  bar.reserve(bar_w);
  for (int i = 0; i < bar_w; ++i)
    bar += (i < filled) ? '#' : '.';

  double rate = (elapsed_sec > 0) ? done / elapsed_sec : 0;
  double remaining = total - done;
  double eta_sec = (done > 0 && rate > 0) ? remaining / rate : 0.0;

  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "\rPlaying games: [%s] %d/%d (%d%%) | %.1f games/s | ETA %s   ",
                bar.c_str(), done, total, pct, rate, format_eta(eta_sec).c_str());
  std::cerr << buf << std::flush;
}

static void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "Options:\n"
      << "  --player <name>            Player type for both sides (required)\n"
      << "  --games <N>                Number of games (required)\n"
      << "  --output <path>            Optional. Writes <path>_game.parquet and "
         "<path>_turn.parquet\n"
      << "                             (requires at least one feature list)\n"
      << "  --game-features <f1,f2>    Comma-separated game-level feature names\n"
      << "  --turn-features <f1,f2>    Comma-separated turn-level feature names\n"
      << "  --seed <S>                 RNG seed (default: random)\n"
      << "  --threads <T>              Threads (default: hardware - 2)\n"
      << "  --samples-md <path>        Markdown: anomaly games with hands + history\n"
      << "\nAvailable players: random, greedy, greedy_random, greedy_random_pass, greedy_no_bomb\n"
      << "  (greedy_random, greedy_random_pass, greedy_no_bomb are only valid in eval_match)\n"
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
  std::string samples_md_path;
  std::vector<std::string> game_feat_names, turn_feat_names;
  unsigned int seed = std::random_device{}();
  int num_threads = std::max(1u, std::thread::hardware_concurrency() - 2);

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--player" && i + 1 < argc)
      player_name = argv[++i];
    else if (arg == "--games" && i + 1 < argc)
      num_games = std::stoi(argv[++i]);
    else if (arg == "--output" && i + 1 < argc)
      output_path = argv[++i];
    else if (arg == "--game-features" && i + 1 < argc)
      game_feat_names = split_csv(argv[++i]);
    else if (arg == "--turn-features" && i + 1 < argc)
      turn_feat_names = split_csv(argv[++i]);
    else if (arg == "--seed" && i + 1 < argc)
      seed = std::stoul(argv[++i]);
    else if (arg == "--threads" && i + 1 < argc)
      num_threads = std::stoi(argv[++i]);
    else if (arg == "--samples-md" && i + 1 < argc)
      samples_md_path = argv[++i];
    else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (player_name.empty() || num_games <= 0) {
    std::cerr << "Error: --player and --games are required\n\n";
    print_usage(argv[0]);
    return 1;
  }

  if (!output_path.empty() && game_feat_names.empty() && turn_feat_names.empty()) {
    std::cerr << "Error: --output requires at least one of --game-features or "
                 "--turn-features\n\n";
    print_usage(argv[0]);
    return 1;
  }

  std::vector<std::shared_ptr<FeatureExtractor>> game_features, turn_features;
  for (const auto &n : game_feat_names) {
    auto f = create_feature(n);
    if (f) {
      game_features.push_back(f);
      std::cout << "Game feature: " << n << "\n";
    }
  }
  for (const auto &n : turn_feat_names) {
    auto f = create_feature(n);
    if (f) {
      turn_features.push_back(f);
      std::cout << "Turn feature: " << n << "\n";
    }
  }

  auto factory = make_player_factory(player_name, 0.0, seed);
  if (!factory) return 1;

  std::cout << "\n=== Data generation / self-play ===\n"
            << "Player:   " << player_name << " vs " << player_name << "\n"
            << "Games:    " << num_games << "\n";
  if (!output_path.empty()) {
    std::cout << "Output:   " << output_path << "_game.parquet\n"
              << "          " << output_path << "_turn.parquet\n";
  } else {
    std::cout << "Output:   (none — stats only)\n";
  }
  std::cout << "Seed:     " << seed << "\n"
            << "Threads:  " << num_threads << "\n\n";

  GameCoordinator coordinator(factory, factory, num_games, "", num_threads, seed, "",
                              game_features, turn_features);

  std::string game_out;
  std::string turn_out;
  if (!output_path.empty()) {
    game_out = output_path + "_game.parquet";
    turn_out = output_path + "_turn.parquet";
  }

  std::atomic<bool> progress_done{false};
  auto t_prog_start = std::chrono::steady_clock::now();
  std::thread progress_thread([&]() {
    while (!progress_done.load()) {
      int done = coordinator.games_completed();
      auto now = std::chrono::steady_clock::now();
      double elapsed =
          std::chrono::duration<double>(now - t_prog_start).count();
      print_progress_line(done, num_games, elapsed);
      if (done >= num_games)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });

  coordinator.run_all(game_out, turn_out);

  progress_done = true;
  if (progress_thread.joinable())
    progress_thread.join();
  std::cerr << "\n";

  coordinator.print_run_summary(std::cout);

  if (!samples_md_path.empty()) {
    if (write_samples_md(coordinator.anomaly_indexed_games(), samples_md_path))
      std::cout << "Wrote sample games (hands + history) to " << samples_md_path
                << "\n";
  }

  std::cout << "Done.\n";
  return 0;
}
