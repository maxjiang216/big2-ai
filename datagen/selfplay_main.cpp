#include "feature_registry.h"
#include "game_record.h"
#include "game_simulator.h"
#include "move.h"
#include "greedy/greedy_player_factory.h"
#include "random/random_player_factory.h"
#include "player_factory.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
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
  if (name == "random")
    return std::make_shared<RandomPlayerFactory>(seed);
  if (name == "greedy")
    return std::make_shared<GreedyPlayerFactory>();
  std::cerr << "Error: unknown player '" << name << "'\n";
  return nullptr;
}

// ============================================================================
// Feature aggregation
// ============================================================================

struct FeatureAgg {
  double sum = 0;
  int mn = INT_MAX, mx = INT_MIN;
  long count = 0;

  void add(int v) {
    sum += v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    ++count;
  }

  double mean() const { return count > 0 ? sum / count : 0.0; }
};

// ============================================================================
// JSONL writer
// ============================================================================

// Sample turns where tablebase case 1 or 2 fired (for debugging / inspection).
// Returns false if the file could not be written.
static bool write_tb_examples(const std::vector<GameRecord> &records,
                              const std::string &path, int max_per_case) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Error: cannot open --tb-examples path '" << path << "'\n";
    return false;
  }
  out << "# Tablebase examples: case1 = forced-win sequence; case2 = opp has 1 card "
         "and move from precomputed straight table only (default/singles-only "
         "case2 are omitted).\n"
         "# State is the full game snapshot *before* the played move.\n\n";

  int n1 = 0, n2 = 0;
  for (size_t g = 0; g < records.size(); ++g) {
    const auto &rec = records[g];
    for (size_t ti = 0; ti < rec.turns().size(); ++ti) {
      const auto &turn = rec.turns()[ti];
      if (turn.tb_case == 1 && n1 < max_per_case) {
        out << "---------- case1  (game " << g << "  turn " << ti << ") ----------\n";
        out << "current_player: " << turn.current_player << "\n";
        out << "forced_seq_len: " << turn.tb_forced_seq_len << "\n";
        out << turn.game;
        out << "played_move: " << turn.move
            << "  move_id=" << encodeMove(turn.move) << "\n\n";
        ++n1;
      } else if (turn.tb_case == 2 && turn.tb_opp1_table_straight &&
                 n2 < max_per_case) {
        out << "---------- case2  (table-straight, game " << g << "  turn " << ti
            << ") ----------\n";
        out << "current_player: " << turn.current_player << "\n";
        out << turn.game;
        out << "played_move: " << turn.move
            << "  move_id=" << encodeMove(turn.move) << "\n\n";
        ++n2;
      }
      if (n1 >= max_per_case && n2 >= max_per_case) {
        out.close();
        return true;
      }
    }
  }
  return true;
}

static void write_jsonl(const std::vector<GameRecord> &records,
                        const std::vector<std::shared_ptr<FeatureExtractor>> &turn_features,
                        const std::string &path) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Error: cannot open output file '" << path << "'\n";
    return;
  }

  // Extract all feature columns upfront for each game.
  // turnExtract returns 2 * num_turns values (p0 then p1 interleaved per turn).
  for (size_t game_idx = 0; game_idx < records.size(); ++game_idx) {
    const GameRecord &rec = records[game_idx];
    const auto &turns = rec.turns();
    size_t num_turns = turns.size();

    // Pre-extract each feature's column for this game.
    std::vector<std::vector<int>> cols;
    cols.reserve(turn_features.size());
    for (const auto &feat : turn_features)
      cols.push_back(feat->turnExtract(rec));

    for (size_t t = 0; t < num_turns; ++t) {
      for (int p = 0; p < 2; ++p) {
        size_t idx = t * 2 + p;
        out << "{\"game_id\":" << game_idx
            << ",\"turn_idx\":" << t
            << ",\"perspective\":" << p;
        for (size_t fi = 0; fi < turn_features.size(); ++fi) {
          out << ",\"" << turn_features[fi]->name() << "\":" << cols[fi][idx];
        }
        out << "}\n";
      }
    }
  }
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "Options:\n"
      << "  --player <name>            Player type for both sides (default: random)\n"
      << "  --games <N>                Number of games (required)\n"
      << "  --threads <T>              Threads (default: hardware - 2)\n"
      << "  --seed <S>                 RNG seed (default: random)\n"
      << "  --game-features <f1,f2>    Game-level features to aggregate + print\n"
      << "  --turn-features <f1,f2>    Turn-level features to aggregate + print\n"
      << "  --output <path>            Write turn data to path_turns.jsonl\n"
      << "  --game-jsonl <path>        One JSON object per game (needs --game-features)\n"
      << "  --tb-examples <path>       Sample turns: case1 (forced win), case2 "
         "(opp-1 table straight only)\n"
      << "  --tb-examples-max <N>      Max examples per bucket (default: 20)\n"
      << "\nAvailable players: random, greedy\n"
      << "\nExample:\n"
      << "  " << prog
      << " --games 10000 --game-features outcome,length,tb_hits,tb_case1,"
         "tb_case2,tb_forced_seq_len,tb_opp1_table_straight"
         " --turn-features turn_outcome,player_hand_size\n";
}

int main(int argc, char **argv) {
  std::string player_name = "random";
  int num_games = 0;
  int num_threads = std::max(1u, std::thread::hardware_concurrency() - 2);
  unsigned int seed = std::random_device{}();
  std::vector<std::string> game_feat_names, turn_feat_names;
  std::string output_path;
  std::string game_jsonl_path;
  std::string tb_examples_path;
  int tb_examples_max = 20;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    else if (arg == "--player"        && i + 1 < argc) player_name = argv[++i];
    else if (arg == "--games"         && i + 1 < argc) num_games   = std::stoi(argv[++i]);
    else if (arg == "--threads"       && i + 1 < argc) num_threads = std::stoi(argv[++i]);
    else if (arg == "--seed"          && i + 1 < argc) seed        = std::stoul(argv[++i]);
    else if (arg == "--game-features" && i + 1 < argc) game_feat_names = split_csv(argv[++i]);
    else if (arg == "--turn-features" && i + 1 < argc) turn_feat_names = split_csv(argv[++i]);
    else if (arg == "--output"        && i + 1 < argc) output_path = argv[++i];
    else if (arg == "--game-jsonl"    && i + 1 < argc) game_jsonl_path = argv[++i];
    else if (arg == "--tb-examples"   && i + 1 < argc) tb_examples_path = argv[++i];
    else if (arg == "--tb-examples-max" && i + 1 < argc)
      tb_examples_max = std::stoi(argv[++i]);
    else { std::cerr << "Unknown argument: " << arg << "\n"; print_usage(argv[0]); return 1; }
  }

  if (num_games <= 0) {
    std::cerr << "Error: --games is required and must be > 0\n";
    print_usage(argv[0]);
    return 1;
  }

  if (!game_jsonl_path.empty() && game_feat_names.empty()) {
    std::cerr << "Error: --game-jsonl requires at least one --game-features name\n";
    return 1;
  }

  if (tb_examples_max < 0) {
    std::cerr << "Error: --tb-examples-max must be >= 0\n";
    return 1;
  }

  // Build feature objects.
  std::vector<std::shared_ptr<FeatureExtractor>> game_features, turn_features;
  for (const auto &n : game_feat_names) {
    auto f = create_feature(n);
    if (f) game_features.push_back(f);
  }
  for (const auto &n : turn_feat_names) {
    auto f = create_feature(n);
    if (f) turn_features.push_back(f);
  }

  // Create factories (each thread gets its own player via the factory).
  auto factory_p0 = make_factory(player_name, seed);
  auto factory_p1 = make_factory(player_name, seed + 1000000u);
  if (!factory_p0 || !factory_p1) return 1;

  std::cout << "\n=== Self-play: " << player_name << " vs " << player_name
            << " (" << num_games << " games, " << num_threads << " threads) ===\n";

  // -------------------------------------------------------------------------
  // Run games (threaded).
  // -------------------------------------------------------------------------
  auto t_start = std::chrono::high_resolution_clock::now();

  std::atomic<int> next_game{0};
  std::vector<std::vector<GameRecord>> thread_records(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&, t]() {
      std::mt19937 rng(seed + static_cast<unsigned>(t) * 999983u);
      auto &out = thread_records[t];
      int idx;
      while ((idx = next_game.fetch_add(1)) < num_games) {
        auto p0 = factory_p0->create_player();
        auto p1 = factory_p1->create_player();
        GameSimulator sim(std::move(p0), std::move(p1), rng);
        out.push_back(sim.run());
      }
    });
  }
  for (auto &th : workers) th.join();

  // Flatten into one vector.
  std::vector<GameRecord> records;
  records.reserve(num_games);
  for (auto &v : thread_records)
    records.insert(records.end(), std::move_iterator(v.begin()),
                   std::move_iterator(v.end()));

  auto t_end = std::chrono::high_resolution_clock::now();
  long elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

  // -------------------------------------------------------------------------
  // Aggregate built-in stats (win rate, game length).
  // -------------------------------------------------------------------------
  int p0_wins = 0;
  long total_turns = 0;
  long tb_total = 0;
  long tb_case1 = 0, tb_case2 = 0;
  long long tb_case1_seq_sum = 0;
  long tb_case1_seq_max = 0;
  long tb_case2_table_straight = 0;
  int games_with_tb = 0;
  int games_with_case1 = 0;
  int games_with_case2 = 0;

  for (const auto &rec : records) {
    if (rec.game().get_winner() == 0) ++p0_wins;
    total_turns += static_cast<long>(rec.turns().size());
    auto s = tablebase_first_hit_stats(rec);
    int seg = s.case1 + s.case2;
    tb_total += seg;
    if (seg > 0)
      ++games_with_tb;
    tb_case1 += s.case1;
    tb_case2 += s.case2;
    if (s.case1 > 0)
      ++games_with_case1;
    if (s.case2 > 0)
      ++games_with_case2;
    tb_case1_seq_sum += s.case1_seq_sum;
    if (s.case1_seq_max > tb_case1_seq_max)
      tb_case1_seq_max = s.case1_seq_max;
    tb_case2_table_straight += s.case2_table_straight;
  }

  int p1_wins = num_games - p0_wins;
  double avg_turns = total_turns > 0 ? static_cast<double>(total_turns) / num_games : 0.0;

  std::cout << "\n"
            << "P0 wins: " << p0_wins << " (" << 100.0 * p0_wins / num_games << "%)"
            << "   P1 wins: " << p1_wins << " (" << 100.0 * p1_wins / num_games << "%)\n"
            << "Avg game length: " << avg_turns << " turns\n";

  // -------------------------------------------------------------------------
  // Game-level feature stats.
  // -------------------------------------------------------------------------
  if (!game_features.empty()) {
    std::cout << "\n--- Game features ---\n";
    for (const auto &feat : game_features) {
      FeatureAgg agg;
      for (const auto &rec : records)
        agg.add(feat->gameExtract(rec));
      std::cout << "  " << feat->name()
                << ": mean=" << agg.mean()
                << "  min=" << agg.mn
                << "  max=" << agg.mx << "\n";
    }
  }

  // -------------------------------------------------------------------------
  // Turn-level feature stats.
  // -------------------------------------------------------------------------
  if (!turn_features.empty()) {
    std::cout << "\n--- Turn features (mean over all turns x perspectives) ---\n";
    for (const auto &feat : turn_features) {
      FeatureAgg agg;
      for (const auto &rec : records) {
        for (int v : feat->turnExtract(rec))
          agg.add(v);
      }
      std::cout << "  " << feat->name()
                << ": mean=" << agg.mean()
                << "  min=" << agg.mn
                << "  max=" << agg.mx << "\n";
    }
  }

  // -------------------------------------------------------------------------
  // Tablebase usage (always shown; rates are vs games, not turns).
  // -------------------------------------------------------------------------
  std::cout << "\n--- Tablebase usage (game-level first-hit segments) ---\n";
  if (num_games > 0) {
    std::cout << "  Games with ≥1 TB segment: " << games_with_tb << " / " << num_games
              << " (" << 100.0 * games_with_tb / num_games << "%)\n"
              << "  Total TB segments:        " << tb_total << "  (avg "
              << static_cast<double>(tb_total) / num_games << " per game)\n"
              << "    case1 (forced win seq): " << tb_case1 << " segments in "
              << games_with_case1 << " games";
    if (tb_case1 > 0) {
      double mean_len =
          static_cast<double>(tb_case1_seq_sum) / static_cast<double>(tb_case1);
      std::cout << "  mean seq len=" << mean_len << "  max=" << tb_case1_seq_max;
    }
    std::cout << "\n"
              << "    case2 (opp has 1 card): " << tb_case2 << " segments in "
              << games_with_case2 << " games";
    if (tb_case2 > 0) {
      std::cout << "  table-straight: " << tb_case2_table_straight << " ("
                << 100.0 * tb_case2_table_straight / tb_case2 << "% of case2 segments)";
    }
    std::cout << "\n";
  }

  // -------------------------------------------------------------------------
  // Timing.
  // -------------------------------------------------------------------------
  double games_per_sec = elapsed_ms > 0 ? 1000.0 * num_games / elapsed_ms : 0.0;
  std::cout << "\nElapsed: " << elapsed_ms << " ms  ("
            << static_cast<long>(games_per_sec) << " games/s)\n\n";

  // -------------------------------------------------------------------------
  // Per-game JSONL (game-level features only).
  // -------------------------------------------------------------------------
  if (!game_jsonl_path.empty()) {
    std::ofstream gj(game_jsonl_path);
    if (!gj) {
      std::cerr << "Error: cannot open --game-jsonl path '" << game_jsonl_path
                << "'\n";
      return 1;
    }
    for (const auto &rec : records) {
      gj << "{";
      for (size_t fi = 0; fi < game_features.size(); ++fi) {
        if (fi)
          gj << ",";
        gj << "\"" << game_features[fi]->name() << "\":"
           << game_features[fi]->gameExtract(rec);
      }
      gj << "}\n";
    }
    std::cout << "Wrote " << records.size() << " games to " << game_jsonl_path
              << "\n";
  }

  // -------------------------------------------------------------------------
  // JSONL output.
  // -------------------------------------------------------------------------
  if (!output_path.empty() && !turn_features.empty()) {
    std::string jsonl_path = output_path + "_turns.jsonl";
    std::cout << "Writing JSONL to " << jsonl_path << " ...\n";
    write_jsonl(records, turn_features, jsonl_path);
    std::cout << "Done.\n";
  }

  if (!tb_examples_path.empty()) {
    if (write_tb_examples(records, tb_examples_path, tb_examples_max))
      std::cout << "Wrote tablebase examples (up to " << tb_examples_max
                << " each: case1 forced-win, case2 table-straight) to "
                << tb_examples_path << "\n";
  }

  return 0;
}
