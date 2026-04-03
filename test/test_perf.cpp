#include "game_record.h"
#include "game_simulator.h"
#include "random/random_player_factory.h"

#include <chrono>
#include <climits>
#include <iomanip>
#include <iostream>
#include <random>

// Perf benchmark: runs N games single-threaded and prints timing + tablebase stats.
// Build with: make benchmark  →  bin/benchmark

static void run_benchmark(int num_games, unsigned int seed) {
  RandomPlayerFactory f0(seed), f1(seed + 1u);
  std::mt19937 rng(seed);

  long total_turns = 0;
  long tb_total = 0;
  long tb_case1 = 0, tb_case2 = 0;
  long long tb_case1_seq_sum = 0;
  long tb_case1_seq_max = 0;
  long tb_case2_table_straight = 0;
  int games_with_tb = 0;
  int games_with_case1 = 0;
  int games_with_case2 = 0;

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_games; ++i) {
    GameSimulator sim(f0.create_player(), f1.create_player(), rng);
    GameRecord rec = sim.run();

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

  auto t1 = std::chrono::high_resolution_clock::now();
  long elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  double games_per_sec = elapsed_ms > 0 ? 1000.0 * num_games / elapsed_ms : 0.0;
  double ms_per_game   = elapsed_ms > 0 ? static_cast<double>(elapsed_ms) / num_games : 0.0;
  double avg_turns     = static_cast<double>(total_turns) / num_games;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "\n=== Perf benchmark: random vs random ===\n"
            << "Games:          " << num_games << "\n"
            << "Elapsed:        " << elapsed_ms << " ms\n"
            << "Throughput:     " << static_cast<long>(games_per_sec) << " games/s\n"
            << "Latency:        " << ms_per_game << " ms/game\n"
            << "Avg turns/game: " << avg_turns << "\n"
            << "\n--- Tablebase usage (game-level first-hit segments) ---\n"
            << "Games with ≥1 TB segment: " << games_with_tb << " / " << num_games
            << " (" << 100.0 * games_with_tb / num_games << "%)\n"
            << "Total TB segments:        " << tb_total << "  (avg "
            << static_cast<double>(tb_total) / num_games << " per game)\n"
            << "  case1 (forced win seq): " << tb_case1 << " segments in "
            << games_with_case1 << " games";
  if (tb_case1 > 0) {
    double mean_len =
        static_cast<double>(tb_case1_seq_sum) / static_cast<double>(tb_case1);
    std::cout << "  mean seq len=" << mean_len << "  max=" << tb_case1_seq_max;
  }
  std::cout << "\n"
            << "  case2 (opp has 1 card): " << tb_case2 << " segments in "
            << games_with_case2 << " games";
  if (tb_case2 > 0) {
    std::cout << "  table-straight: " << tb_case2_table_straight << " ("
              << 100.0 * tb_case2_table_straight / tb_case2
              << "% of case2 segments)";
  }
  std::cout << "\n\n";
}

// Called from test_main so it appears in the test suite, and also usable
// standalone via bin/benchmark which defines its own main below.
void run_perf_tests() {
  constexpr int QUICK = 5'000;
  std::cout << "[perf] Quick run (" << QUICK << " games)...\n";
  run_benchmark(QUICK, 42u);
}

#ifdef BENCHMARK_MAIN
int main(int argc, char **argv) {
  int num_games = 100'000;
  unsigned int seed = 42u;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--games"  || arg == "-n") && i + 1 < argc)
      num_games = std::stoi(argv[++i]);
    else if ((arg == "--seed" || arg == "-s") && i + 1 < argc)
      seed = static_cast<unsigned>(std::stoul(argv[++i]));
    else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: benchmark [--games N] [--seed S]\n";
      return 0;
    }
  }

  run_benchmark(num_games, seed);
  return 0;
}
#endif
