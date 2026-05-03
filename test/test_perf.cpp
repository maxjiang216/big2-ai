#include "game_record.h"
#include "game_simulator.h"
#include "game.h"
#include "random/random_player_factory.h"
#include "util.h"

#include <algorithm>
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

// Micro-benchmark: compute_legal_moves hot paths (PASS vs trailing, into vs
// return-by-value). Rebuild with e.g. CXXFLAGS='-O1 ...' or '-O3 ...' to compare.
static void microbench_compute_legal_moves() {
  std::mt19937 rng(7);
  constexpr int N_POS = 400;
  std::vector<std::array<int, 13>> pass_hands;
  std::vector<int> pass_lids;
  std::vector<std::array<int, 13>> resp_hands;
  std::vector<int> resp_lids;
  pass_hands.reserve(N_POS);
  pass_lids.reserve(N_POS);
  resp_hands.reserve(N_POS);
  resp_lids.reserve(N_POS);

  while (static_cast<int>(pass_hands.size()) < N_POS) {
    Game g;
    g.shuffle_deal(rng);
    if (g.last_move_id() != kPASS)
      continue;
    pass_hands.push_back(g.player_hand(g.current_player()));
    pass_lids.push_back(kPASS);
  }
  while (static_cast<int>(resp_hands.size()) < N_POS) {
    Game g;
    g.shuffle_deal(rng);
    for (int s = 0; s < 5 && !g.is_over(); ++s) {
      auto leg = g.get_legal_moves();
      if (leg.size() <= 1)
        break;
      g.apply_move(leg[1 + static_cast<int>(rng() % (leg.size() - 1))]);
    }
    if (g.last_move_id() == kPASS || g.is_over())
      continue;
    resp_hands.push_back(g.player_hand(g.current_player()));
    resp_lids.push_back(g.last_move_id());
  }

  constexpr int TRIALS = 250'000;
  std::vector<int> buf;
  buf.reserve(64);
  int sink = 0;

  auto t_pass_in = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < TRIALS; ++i) {
    int j = i % N_POS;
    compute_legal_moves_into(pass_hands[j], pass_lids[j], buf);
    sink += static_cast<int>(buf.size());
  }
  auto t_pass_in_end = std::chrono::high_resolution_clock::now();

  auto t_pass_val = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < TRIALS; ++i) {
    int j = i % N_POS;
    auto v = compute_legal_moves(pass_hands[j], pass_lids[j]);
    sink += static_cast<int>(v.size());
  }
  auto t_pass_val_end = std::chrono::high_resolution_clock::now();

  auto t_resp_in = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < TRIALS; ++i) {
    int j = i % N_POS;
    compute_legal_moves_into(resp_hands[j], resp_lids[j], buf);
    sink += static_cast<int>(buf.size());
  }
  auto t_resp_in_end = std::chrono::high_resolution_clock::now();

  auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };

  std::cout << "\n=== Microbench: compute_legal_moves (" << TRIALS
            << " calls each; sink=" << sink << ") ===\n";
  std::cout << "  kPASS + into/reuse:     " << ms(t_pass_in, t_pass_in_end)
            << " ms\n";
  std::cout << "  kPASS + return vector:  " << ms(t_pass_val, t_pass_val_end)
            << " ms\n";
  std::cout << "  response + into/reuse:  " << ms(t_resp_in, t_resp_in_end)
            << " ms\n";
}

static void collect_kpass_rec(int rank, int rem, std::array<int, 13> &h,
                              std::vector<std::array<int, 13>> &out) {
  if (rank == 12) {
    if (rem >= 0 && rem <= max_cards_in_deck_for_rank(12)) {
      h[12] = rem;
      out.push_back(h);
    }
    return;
  }
  const int mx = max_cards_in_deck_for_rank(rank);
  for (int c = 0; c <= rem && c <= mx; ++c) {
    h[rank] = c;
    collect_kpass_rec(rank + 1, rem - c, h, out);
  }
}

static void collect_kpass_small_hands(std::vector<std::array<int, 13>> &out) {
  std::array<int, 13> h{};
  for (int X = 1; X <= KPASS_LEGAL_TABLE_MAX_CARDS_BUILT; ++X)
    collect_kpass_rec(0, X, h, out);
}

// kPASS legal microbench: varies g_kpass_legal_max_cards (0 = table off).
static void microbench_kpass_table_caps() {
  std::vector<std::array<int, 13>> hands;
  collect_kpass_small_hands(hands);
  if (hands.empty())
    return;

  std::vector<int> buf;
  buf.reserve(static_cast<std::size_t>(LEGAL_MOVES_SIZE));
  int sink = 0;
  constexpr int TRIALS = 200'000;

  std::cout << "\n=== Microbench: kPASS precomputed table by card cap\n"
            << "    (all " << hands.size()
            << " deck-capped hands with 1.." << KPASS_LEGAL_TABLE_MAX_CARDS_BUILT
            << " cards; " << TRIALS << " compute_legal_moves_into calls each)\n";

  const int saved = g_kpass_legal_max_cards;
  for (int cap = 0; cap <= KPASS_LEGAL_TABLE_MAX_CARDS_BUILT; ++cap) {
    g_kpass_legal_max_cards = cap;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TRIALS; ++i) {
      const auto &hh = hands[static_cast<std::size_t>(i) % hands.size()];
      compute_legal_moves_into(hh, kPASS, buf);
      sink += static_cast<int>(buf.size());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "  g_kpass_legal_max_cards=" << cap << "  ";
    if (cap == 0)
      std::cout << "(table off):  ";
    else
      std::cout << "(table on when sum(hand)<=" << cap << "):  ";
    std::cout << ms << " ms\n";
  }
  g_kpass_legal_max_cards = saved;
  std::cout << "  sink=" << sink << "\n";
}

// Worst-case-ish: discard accounts for all known cards so unseen[r]=0 and
// opponent_can_respond scans many beating candidates before concluding false.
static void microbench_opponent_can_respond() {
  std::array<int, 13> hand{};
  std::array<int, 13> discard{};
  for (int r = 0; r < 13; ++r)
    discard[r] = max_cards_in_deck_for_rank(r);
  constexpr int TRIALS = 40'000;
  int sink = 0;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < TRIALS; ++i) {
    for (int mid = kSINGLE_START; mid < kSINGLE_START + 13; ++mid)
      sink += opponent_can_respond(mid, hand, discard, 13);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  const long ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::cout << "\n=== Microbench: opponent_can_respond (hard; discard=max)\n"
            << "    " << TRIALS << " × 13 move ids, sink=" << sink << "  " << ms
            << " ms\n";
}

// Called from test_main so it appears in the test suite, and also usable
// standalone via bin/benchmark which defines its own main below.
void run_perf_tests() {
  constexpr int QUICK = 5'000;
  std::cout << "[perf] Quick run (" << QUICK << " games)...\n";
  run_benchmark(QUICK, 42u);
  microbench_compute_legal_moves();
  microbench_kpass_table_caps();
  microbench_opponent_can_respond();
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
    else if (arg == "--microbench-legal") {
      microbench_compute_legal_moves();
      return 0;
    } else if (arg == "--bench-kpass-caps") {
      microbench_kpass_table_caps();
      return 0;
    } else if (arg == "--bench-opponent-can-respond") {
      microbench_opponent_can_respond();
      return 0;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: benchmark [--games N] [--seed S] [--microbench-legal] "
                   "[--bench-kpass-caps] [--bench-opponent-can-respond]\n";
      return 0;
    }
  }

  run_benchmark(num_games, seed);
  microbench_compute_legal_moves();
  microbench_kpass_table_caps();
  microbench_opponent_can_respond();
  return 0;
}
#endif
