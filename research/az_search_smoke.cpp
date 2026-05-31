// End-to-end smoke for the az_search core driven by the real NN evaluator.
//
//   make az_search_smoke && ./bin/az_search_smoke models/az_player.pt models/az_opp.pt [sims]
//
// Builds a mid-game root, runs the search through a CachingEvaluator wrapping the
// LibTorch NNEvaluator, and prints the chosen move, root value, node/cache stats,
// then exercises subtree reuse via advance_root.

#include "az_search/az_search.h"
#include "az_search/eval_cache.h"
#include "az_search/nn_eval.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace az_search;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s player.pt opp.pt [sims]\n", argv[0]);
    return 2;
  }
  const int sims = (argc > 3) ? std::atoi(argv[3]) : 200;

  NNEvaluator nn(argv[1], argv[2], torch::Device(torch::kCPU));
  CachingEvaluator cache(nn);

  // Mid-game lead position: non-consecutive singles 3,5,7,9,J,K,2 (7 cards, no
  // straight / no single hand-emptying move) so the search builds a real tree.
  SearchState root{};
  for (int idx : {0, 2, 4, 6, 8, 10, 12}) root.our_hand[idx] = 1;
  root.opp_size = 7;
  root.last_move = kPASS;  // lead
  root.side = kUs;

  Search s(root, {1.5f, sims});
  s.run(cache);

  const int bm = s.best_move();
  std::cout << "best_move id=" << bm << " (" << Move(bm) << ")\n";
  std::printf("root_value = %.4f\n", s.root_value());
  std::printf("nodes = %zu  player_misses = %ld  opp_misses = %ld  cache_size = %zu\n",
              s.num_nodes(), cache.player_misses(), cache.opp_misses(), cache.size());

  // Subtree reuse: re-root onto the chosen move's resulting opp position.
  SearchState next = az_transition(root, bm);
  normalize_forced_pass(next);
  const long before = (long)s.num_nodes();
  s.advance_root(next);
  std::printf("after advance_root: root_n = %ld  nodes_delta = %ld\n",
              s.root_n(), (long)s.num_nodes() - before);
  return 0;
}
