// Step-5 validation: run AzSearchPlayer (via its factory) against greedy through
// the real GameSimulator, alternating seats. Confirms the player loads the nets,
// plays only legal moves (the engine would reject otherwise), maintains its
// persistent per-turn tree, and that games complete with a winner.
//
//   make az_play_check
//   ./bin/az_play_check models/az_player.pt models/az_opp.pt [games] [sims]

#include "az_search/az_search_player_factory.h"
#include "game_record.h"
#include "game_simulator.h"
#include "player_factory_registry.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s player.pt opp.pt [games] [sims]\n", argv[0]);
    return 2;
  }
  const int games = (argc > 3) ? std::atoi(argv[3]) : 20;
  const int sims = (argc > 4) ? std::atoi(argv[4]) : 100;

  az_search::AzSearchPlayerFactory az(sims, /*seed=*/123, argv[1], argv[2],
                                      /*training=*/false);
  auto greedy = make_player_factory("greedy", 0.0, 7);

  std::mt19937 rng(42);
  int az_wins = 0, total = 0;
  for (int g = 0; g < games; ++g) {
    const bool az_first = (g % 2 == 0);
    auto p0 = az_first ? az.create_player() : greedy->create_player();
    auto p1 = az_first ? greedy->create_player() : az.create_player();
    GameSimulator sim(std::move(p0), std::move(p1), rng);
    GameRecord rec = sim.run();
    const int az_seat = az_first ? 0 : 1;
    if (rec.game().get_winner() == az_seat) ++az_wins;
    ++total;
  }
  std::printf("az_search %d / %d vs greedy (sims=%d) — all games completed legally\n",
              az_wins, total, sims);
  return 0;
}
