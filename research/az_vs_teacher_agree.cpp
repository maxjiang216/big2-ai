// Where do az_search (net + 100-sim search) and its teacher (typed_search)
// disagree on the move? Plays typed_search self-play to get game trajectories,
// then at each REAL decision runs the FULL az search (so the value lookahead can
// override the raw policy) and dumps (teacher_move_id, az_searched_move_id) pairs
// to a CSV for breakdown in Python.
//
//   make az_vs_teacher_agree && ./bin/az_vs_teacher_agree \
//       --player models/az_player_gen0.pt --opp models/az_opp_gen0.pt \
//       --games 500 --sims 100 --out /tmp/agree_pairs.csv

#include "az_search/az_search.h"
#include "az_search/eval_cache.h"
#include "az_search/nn_eval.h"

#include "game.h"
#include "game_record.h"
#include "game_simulator.h"
#include "move.h"
#include "player_factory_registry.h"
#include "util.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

using namespace az_search;

static const char *arg(int c, char **v, const char *k, const char *d) {
  for (int i = 1; i + 1 < c; ++i)
    if (std::strcmp(v[i], k) == 0) return v[i + 1];
  return d;
}

int main(int argc, char **argv) {
  const std::string player = arg(argc, argv, "--player", "models/az_player_gen0.pt");
  const std::string opp = arg(argc, argv, "--opp", "models/az_opp_gen0.pt");
  const int games = std::atoi(arg(argc, argv, "--games", "500"));
  const int sims = std::atoi(arg(argc, argv, "--sims", "100"));
  const unsigned seed = (unsigned)std::strtoul(arg(argc, argv, "--seed", "1"), nullptr, 10);
  const std::string out = arg(argc, argv, "--out", "/tmp/agree_pairs.csv");

  NNEvaluator nn(player, opp, torch::Device(torch::kCPU));
  auto factory = make_player_factory("typed_search", 0, seed);
  std::mt19937 rng(seed);

  FILE *f = std::fopen(out.c_str(), "w");
  std::fprintf(f, "teacher,az\n");
  long decisions = 0, agree = 0;

  for (int g = 0; g < games; ++g) {
    GameSimulator sim(factory->create_player(), factory->create_player(), rng);
    GameRecord rec = sim.run();
    for (const TurnRecord &tr : rec.turns()) {
      // REAL decisions only: >=2 legal moves and no tablebase/forced tag (those
      // resolve trivially and az would match the teacher by construction).
      if (tr.legal_moves.size() < 2 || tr.tb_case != -1) continue;
      const int mover = tr.current_player;
      SearchState st;
      st.our_hand = tr.game.player_hand(mover);
      st.opp_size = tr.game.get_player_hand_size(1 - mover);
      st.discard = tr.game.discard_pile();
      st.last_move = encodeMove(tr.game.last_move());
      st.side = kUs;
      CachingEvaluator cache(nn);
      Search s(st, SearchConfig{1.5f, sims, seed + (unsigned)g, /*training=*/false});
      s.run(cache);
      const int az = s.best_move();
      const int teach = encodeMove(tr.move);
      std::fprintf(f, "%d,%d\n", teach, az);
      ++decisions;
      if (az == teach) ++agree;
    }
  }
  std::fclose(f);
  std::printf("decisions=%ld  agree=%ld (%.1f%%)  -> %s\n", decisions, agree,
              100.0 * (double)agree / (double)decisions, out.c_str());
  return 0;
}
