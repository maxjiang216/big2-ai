#include "game_simulator.h"
#include "player_factory_registry.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if BIG2_PIMC_STATS
#include <inttypes.h>
#endif

// ============================================================================
// Wilson score 95% confidence interval for a proportion
// ============================================================================

struct WilsonCI {
  double lo, hi;
};

static WilsonCI wilson_ci(double p_hat, long n, double z = 1.96) {
  double z2 = z * z;
  double n_inv = 1.0 / static_cast<double>(n);
  double center = (p_hat + z2 * n_inv / 2.0) / (1.0 + z2 * n_inv);
  double half = z * std::sqrt(p_hat * (1.0 - p_hat) * n_inv + z2 * n_inv * n_inv / 4.0) /
                (1.0 + z2 * n_inv);
  return {center - half, center + half};
}

// ============================================================================
// Paired deal simulation
//
// For each deal index i, we seed an rng with (base_seed + i). We run two
// games from that seed: one where A=P0, B=P1, and one where A=P1, B=P0.
// Because both rngs start from the same seed, both games see the same card
// distribution, which cancels out luck from the deal.
// ============================================================================

static int run_paired_deal(unsigned int deal_seed,
                            PlayerFactory &factory_a, PlayerFactory &factory_b) {
  int a_wins = 0;

  for (int swap = 0; swap < 2; ++swap) {
    std::mt19937 rng(deal_seed);
    auto p0 = (swap == 0) ? factory_a.create_player() : factory_b.create_player();
    auto p1 = (swap == 0) ? factory_b.create_player() : factory_a.create_player();
    GameSimulator sim(std::move(p0), std::move(p1), rng);
    GameRecord rec = sim.run();
    int winner = rec.game().get_winner();
    // winner==0 means P0 won; A is P0 when swap==0, P1 when swap==1
    if ((swap == 0 && winner == 0) || (swap == 1 && winner == 1))
      ++a_wins;
  }
  return a_wins;
}

// ============================================================================
// CLI helpers
// ============================================================================

static void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "Options:\n"
      << "  --p0 <name>          Player A type (required)\n"
      << "  --p0-param <f>       Parameter for player A (default: 0.0)\n"
      << "  --p1 <name>          Player B type (required)\n"
      << "  --p1-param <f>       Parameter for player B (default: 0.0)\n"
      << "  --deals <N>          Number of unique deals (total games = 2*N) (required)\n"
      << "  --seed <S>           Base RNG seed (default: random)\n"
      << "  --threads <T>        Worker threads (default: hardware - 2)\n"
      << "\nAvailable players: random, greedy, greedy_random, greedy_random_pass, greedy_no_bomb\n"
      << "\nExample:\n"
      << "  " << prog
      << " --p0 greedy_no_bomb --p0-param 0.3 --p1 greedy --deals 50000\n";
}

static std::string player_label(const std::string &name, double param) {
  if (name == "greedy" || name == "random")
    return name;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%s(%.2g)", name.c_str(), param);
  return buf;
}

static std::string format_elapsed(long long ms) {
  if (ms < 1000) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld ms", ms);
    return buf;
  }
  char buf[64];
  double s = ms / 1000.0;
  if (s < 60)
    std::snprintf(buf, sizeof(buf), "%.1f s", s);
  else
    std::snprintf(buf, sizeof(buf), "%dm %ds", (int)(s / 60), (int)s % 60);
  return buf;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  std::string p0_name, p1_name;
  double p0_param = 0.0, p1_param = 0.0;
  int num_deals = 0;
  unsigned int seed = std::random_device{}();
  int num_threads = std::max(1u, std::thread::hardware_concurrency() - 2);

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--p0" && i + 1 < argc)
      p0_name = argv[++i];
    else if (arg == "--p0-param" && i + 1 < argc)
      p0_param = std::stod(argv[++i]);
    else if (arg == "--p1" && i + 1 < argc)
      p1_name = argv[++i];
    else if (arg == "--p1-param" && i + 1 < argc)
      p1_param = std::stod(argv[++i]);
    else if (arg == "--deals" && i + 1 < argc)
      num_deals = std::stoi(argv[++i]);
    else if (arg == "--seed" && i + 1 < argc)
      seed = static_cast<unsigned int>(std::stoul(argv[++i]));
    else if (arg == "--threads" && i + 1 < argc)
      num_threads = std::stoi(argv[++i]);
    else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (p0_name.empty() || p1_name.empty() || num_deals <= 0) {
    std::cerr << "Error: --p0, --p1, and --deals are required\n\n";
    print_usage(argv[0]);
    return 1;
  }

  std::string label0 = player_label(p0_name, p0_param);
  std::string label1 = player_label(p1_name, p1_param);
  long total_games = 2L * num_deals;

  std::cout << "\n=== Eval Match: " << label0 << " vs " << label1 << " ===\n"
            << "Deals:   " << num_deals << "  |  Total games: " << total_games << "\n"
            << "Seed:    " << seed << "\n"
            << "Threads: " << num_threads << "\n\n";

  // Each thread gets factories seeded differently to avoid seed collisions
  // across threads (factories increment seeds per player created).
  // We give thread t an offset of t * 1000003 from the base seed.
  std::atomic<int> next_deal{0};
  std::atomic<long> a_wins_total{0};

  auto t_start = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&, t]() {
      // Per-thread factories with distinct seeds so player RNGs don't overlap.
      unsigned int thread_seed = seed + static_cast<unsigned int>(t) * 1000003u;
      auto fa = make_player_factory(p0_name, p0_param, thread_seed);
      auto fb = make_player_factory(p1_name, p1_param, thread_seed + 500009u);
      if (!fa || !fb) return;

      int deal_idx;
      long local_wins = 0;
      while ((deal_idx = next_deal.fetch_add(1)) < num_deals) {
        // Deal seed is deterministic from the global seed + deal index so
        // that every thread arrives at the same card distribution for deal i.
        unsigned int deal_seed = seed + static_cast<unsigned int>(deal_idx);
        local_wins += run_paired_deal(deal_seed, *fa, *fb);
      }
      a_wins_total.fetch_add(local_wins);
    });
  }

  // Progress monitor
  std::atomic<bool> progress_done{false};
  auto t_prog = std::chrono::steady_clock::now();
  std::thread progress_thread([&]() {
    while (!progress_done.load()) {
      int done = next_deal.load();
      if (done > num_deals) done = num_deals;
      auto now = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(now - t_prog).count();
      double rate = elapsed > 0 ? (2.0 * done) / elapsed : 0;
      int pct = num_deals > 0 ? static_cast<int>(100.0 * done / num_deals) : 0;
      std::fprintf(stderr, "\rDeals: %d/%d (%d%%)  |  %.0f games/s   ",
                   done, num_deals, pct, rate);
      std::fflush(stderr);
      if (done >= num_deals) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });

  for (auto &w : workers)
    if (w.joinable()) w.join();

  progress_done = true;
  if (progress_thread.joinable()) progress_thread.join();
  std::fprintf(stderr, "\n");

  auto t_end = std::chrono::high_resolution_clock::now();
  long long elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

  long a_wins = a_wins_total.load();
  double p_hat = static_cast<double>(a_wins) / static_cast<double>(total_games);
  WilsonCI ci = wilson_ci(p_hat, total_games);
  double games_per_sec = elapsed_ms > 0 ? 1000.0 * total_games / elapsed_ms : 0;

  std::cout << "P0 (" << label0 << ") wins: " << a_wins << " / " << total_games
            << " (" << std::fixed;
  std::cout.precision(2);
  std::cout << 100.0 * p_hat << "%)\n";
  std::cout << "95% Wilson CI: [" << 100.0 * ci.lo << "%, " << 100.0 * ci.hi << "%]\n";

  if (ci.lo > 0.50)
    std::cout << "Result: " << label0 << " is significantly better (CI excludes 50%)\n";
  else if (ci.hi < 0.50)
    std::cout << "Result: " << label1 << " is significantly better (CI excludes 50%)\n";
  else
    std::cout << "Result: no significant difference (CI includes 50%)\n";

  std::cout << "Elapsed: " << format_elapsed(elapsed_ms)
            << "  (" << static_cast<long>(games_per_sec) << " games/s)\n\n";

#if BIG2_PIMC_STATS
  {
    const auto &st = pimc_global_stats();
    const uint64_t calls = st.total_select_calls.load();
    const uint64_t dets = st.total_dets_used.load();
    std::fprintf(stderr,
                 "[PIMC stats] select_calls=%" PRIu64 " total_dets=%" PRIu64
                 " avg_dets_per_call=%.3f\n",
                 calls, dets,
                 calls ? static_cast<double>(dets) / static_cast<double>(calls)
                       : 0.0);
    std::fprintf(stderr,
                 "[PIMC stats] total_dets_saved=%" PRIu64
                 " rollout_equiv_saved=%" PRIu64
                 " (candidate×det cells skipped vs n_max per call)\n",
                 st.total_dets_saved.load(),
                 st.total_rollout_equiv_saved.load());
  }
#endif

  return 0;
}
