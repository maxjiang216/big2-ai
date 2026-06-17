// End-to-end smoke for the az_search core driven by the real seq NN evaluator.
//
//   make az_search_smoke && ./bin/az_search_smoke models/az_seq.pt [sims]
//
// Builds a mid-game root (with a short fabricated history), runs the search
// through the LibTorch NNEvaluator (KV-cache path), prints the chosen move,
// root value and node stats, then exercises subtree reuse via advance_root,
// and cross-checks the KV-cache path against --no-kv-cache full recompute.

#include "az_search/az_search.h"
#include "az_search/nn_eval.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace az_search;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s az_seq.pt [sims]\n", argv[0]);
    return 2;
  }
  const int sims = (argc > 2) ? std::atoi(argv[2]) : 200;

  // Mid-game lead position: non-consecutive singles 3,5,7,9,J,K,2 (7 cards, no
  // straight / no single hand-emptying move) so the search builds a real tree.
  // History kept empty for the smoke (the net doesn't validate consistency);
  // in-tree path tokens still exercise the KV-cache leaf path.
  SearchState root{};
  for (int idx : {0, 2, 4, 6, 8, 10, 12}) root.our_hand[idx] = 1;
  root.opp_size = 7;
  root.last_move = kPASS;  // lead
  root.side = kUs;

  auto run_once = [&](bool use_kv) {
    NNEvaluator nn(argv[1], torch::Device(torch::kCPU), /*max_slots=*/1, use_kv);
    nn.set_prefix({});
    Search s(root, {}, {1.5f, sims});
    s.run(nn);
    return std::pair<int, float>(s.best_move(), s.root_value());
  };

  NNEvaluator nn(argv[1], torch::Device(torch::kCPU), /*max_slots=*/1, true);
  nn.set_prefix({});
  Search s(root, {}, {1.5f, sims});
  s.run(nn);

  const int bm = s.best_move();
  std::cout << "best_move id=" << bm << " (" << Move(bm) << ")\n";
  std::printf("root_value = %.4f\n", s.root_value());
  std::printf("nodes = %zu\n", s.num_nodes());

  // Subtree reuse: re-root onto the chosen move's resulting position with the
  // matching one-move history.
  SearchState next = az_transition(root, bm);
  SearchState raw = next;
  normalize_forced_pass(next);
  std::vector<int> hist{bm};
  if (next.side != raw.side) hist.push_back(kPASS);  // fused forced pass
  const long before = (long)s.num_nodes();
  s.advance_root(next, hist);
  std::printf("after advance_root: root_n = %ld  nodes_delta = %ld\n",
              s.root_n(), (long)s.num_nodes() - before);

  // KV-cache vs full-recompute consistency (deterministic play mode).
  auto kv = run_once(true);
  auto full = run_once(false);
  std::printf("kv: move=%d value=%.6f   full: move=%d value=%.6f   %s\n",
              kv.first, kv.second, full.first, full.second,
              (kv.first == full.first && std::fabs(kv.second - full.second) < 1e-3)
                  ? "MATCH" : "MISMATCH");
  return 0;
}
