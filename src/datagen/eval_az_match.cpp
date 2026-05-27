// eval_az_match: paired-deal evaluation for az_search.
//
// Player A is always az_search (--player-a / --opp-a). Player B is either
// another az_search model pair (--player-b / --opp-b) or a classic policy
// (--classic NAME [--classic-param P], via the player factory registry).
//
// For each deal we play two games from the same shuffle with seats swapped, so
// card luck cancels (mirrors eval_match). Reports A's win count, win rate, a
// Wilson 95% CI, and a significance line.
//
//   make eval_az_match
//   ./bin/eval_az_match --player-a models/az_player_gen2.pt --opp-a models/az_opp_gen2.pt \
//       --player-b models/az_player_gen1.pt --opp-b models/az_opp_gen1.pt --deals 200 --sims 200
//   ./bin/eval_az_match --player-a models/az_player.pt --opp-a models/az_opp.pt \
//       --classic greedy --deals 500 --sims 200

#include "az_search/az_search_player_factory.h"
#include "eval_helpers.h"  // wilson_ci, format_elapsed
#include "game_record.h"
#include "game_simulator.h"
#include "player_factory_registry.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>

static const char *arg(int argc, char **argv, const char *key, const char *def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return def;
}

int main(int argc, char **argv) {
  const std::string player_a = arg(argc, argv, "--player-a", "models/az_player.pt");
  const std::string opp_a = arg(argc, argv, "--opp-a", "models/az_opp.pt");
  const std::string player_b = arg(argc, argv, "--player-b", "");
  const std::string opp_b = arg(argc, argv, "--opp-b", "");
  const std::string classic = arg(argc, argv, "--classic", "");
  const double classic_param = std::atof(arg(argc, argv, "--classic-param", "0"));
  const int deals = std::atoi(arg(argc, argv, "--deals", "200"));
  const int sims = std::atoi(arg(argc, argv, "--sims", "200"));
  const unsigned base_seed = (unsigned)std::strtoul(arg(argc, argv, "--seed", "42"), nullptr, 10);

  // Play mode (training=false) -> deterministic opponent representative.
  auto factory_a = std::make_shared<az_search::AzSearchPlayerFactory>(
      sims, base_seed, player_a, opp_a, /*training=*/false);

  std::shared_ptr<PlayerFactory> factory_b;
  std::string b_label;
  if (!classic.empty()) {
    factory_b = make_player_factory(classic, classic_param, base_seed + 1);
    if (!factory_b) return 1;
    b_label = "classic:" + classic;
  } else if (!player_b.empty() && !opp_b.empty()) {
    factory_b = std::make_shared<az_search::AzSearchPlayerFactory>(
        sims, base_seed + 1, player_b, opp_b, /*training=*/false);
    b_label = "az:" + player_b;
  } else {
    std::fprintf(stderr, "need either --classic NAME or --player-b/--opp-b\n");
    return 2;
  }

  std::printf("eval_az_match: A=az:%s  vs  B=%s  (%d deals x2, sims=%d)\n",
              player_a.c_str(), b_label.c_str(), deals, sims);

  auto t0 = std::chrono::steady_clock::now();
  long a_wins = 0, games = 0;
  for (int d = 0; d < deals; ++d) {
    const unsigned deal_seed = base_seed + (unsigned)d;
    for (int swap = 0; swap < 2; ++swap) {
      std::mt19937 rng(deal_seed);
      auto p0 = (swap == 0) ? factory_a->create_player() : factory_b->create_player();
      auto p1 = (swap == 0) ? factory_b->create_player() : factory_a->create_player();
      GameSimulator sim(std::move(p0), std::move(p1), rng);
      GameRecord rec = sim.run();
      const int winner = rec.game().get_winner();
      if ((swap == 0 && winner == 0) || (swap == 1 && winner == 1)) ++a_wins;
      ++games;
    }
  }
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();

  const double p = (double)a_wins / (double)games;
  WilsonCI ci = wilson_ci(p, games);
  std::printf("A wins %ld / %ld = %.3f  (Wilson 95%% CI [%.3f, %.3f])  [%s]\n",
              a_wins, games, p, ci.lo, ci.hi, format_elapsed(ms).c_str());
  if (ci.lo > 0.5)
    std::printf("  A is significantly stronger (CI above 50%%).\n");
  else if (ci.hi < 0.5)
    std::printf("  B is significantly stronger (CI below 50%%).\n");
  else
    std::printf("  not significant at 95%% (CI straddles 50%%).\n");
  return 0;
}
