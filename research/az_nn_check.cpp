// Cross-check az_search C++ seq-net inference against the Python forward pass.
//
//   make az_nn_check
//   ./bin/az_nn_check models/az_seq.pt
//   uv run python -m nn.az_nn_check models/az_seq.pt
//
// Both print value / first logits / argmax per head for a fixed set of
// (history, side-input) cases defined identically on both sides; the numbers
// must agree to fp tolerance. Cases cover: empty history (BOS only), a history
// containing a forced pass, and a longer line — each evaluated through BOTH
// the KV-cache path (prefix split at a fixed point) and full recompute.

#include "az_search/nn_eval.h"

#include <cstdio>
#include <vector>

using namespace az_search;

struct Case {
  std::vector<int> prefix;       // game history before the search root
  std::vector<int> path;         // in-tree path tokens
  EvalFeatures f;                // side inputs (path_tokens filled from `path`)
};

// Fixed test cases (mirrored in nn/az_nn_check.py). Move ids: 1..13 singles by
// rank (1='3'), 0=pass; the histories need not be game-consistent — both sides
// feed the identical token stream, which is all the parity check needs.
static std::vector<Case> cases() {
  std::vector<Case> cs(3);
  // 0: empty history, empty path (opening decision, BOS readout).
  cs[0].prefix = {};
  cs[0].path = {};
  cs[0].f = {{2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1},
             {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             16, 16, true, {}};
  // 1: short history with a forced pass; owner waiting (opp heads).
  cs[1].prefix = {3, 7, 0};        // single 5, single 9, pass
  cs[1].path = {5, 0, 2};          // single 7, pass, single 4
  cs[1].f = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
             {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0},
             {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
             11, 13, false, {}};
  // 2: longer mixed line, owner to move.
  cs[2].prefix = {1, 14, 0, 26, 0, 4, 8, 12, 0};
  cs[2].path = {2, 6};
  cs[2].f = {{0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 1, 0},
             {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0},
             {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
             8, 6, true, {}};
  return cs;
}

template <std::size_t N>
static int argmax(const std::array<float, N> &a) {
  int best = 0;
  for (int i = 1; i < (int)N; ++i)
    if (a[i] > a[best]) best = i;
  return best;
}

static void print_eval(const char *tag, std::size_t i, const NetEval &e) {
  std::printf("  %s case%zu value=%.6f policy[0..4]=%.6f %.6f %.6f %.6f %.6f "
              "pamax=%d behavior[0..2]=%.6f %.6f %.6f bamax=%d qa[0..2]=%.6f "
              "%.6f %.6f\n",
              tag, i, e.value, e.policy[0], e.policy[1], e.policy[2],
              e.policy[3], e.policy[4], argmax(e.policy), e.behavior[0],
              e.behavior[1], e.behavior[2], argmax(e.behavior), e.qa[0],
              e.qa[1], e.qa[2]);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s az_seq.pt\n", argv[0]);
    return 2;
  }
  NNEvaluator kv(argv[1], torch::Device(torch::kCPU), /*max_slots=*/1, true);
  NNEvaluator full(argv[1], torch::Device(torch::kCPU), /*max_slots=*/1, false);

  auto cs = cases();
  for (std::size_t i = 0; i < cs.size(); ++i) {
    Case c = cs[i];
    c.f.path_tokens = c.path;
    kv.set_prefix(c.prefix);
    print_eval("kv  ", i, kv.eval(c.f));
    full.set_prefix(c.prefix);
    print_eval("full", i, full.eval(c.f));
  }
  return 0;
}
