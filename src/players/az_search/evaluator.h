#ifndef AZ_SEARCH_EVALUATOR_H
#define AZ_SEARCH_EVALUATOR_H

// Abstract leaf evaluator for the search core. Torch-free: the real NN-backed
// evaluator (nn_eval.h) and a deterministic test stub both implement it, so the
// search core compiles and unit-tests without LibTorch.
//
// UNIFIED interface for the history-transformer net: one eval per leaf returns
// all four heads; the search reads (policy, value) at our-turn leaves and
// (behavior, qa) at opponent-turn leaves. The history enters as `path_tokens`
// — the move ids from the SEARCH ROOT to the leaf; the evaluator owns the
// game-history prefix (set once per real turn) and the KV cache.

#include "features.h"

#include <vector>

namespace az_search {

// One leaf position. Counts are raw; the evaluator builds the 48-dim float
// encodings and normalises sizes. `trick` is retained for the legacy two-net
// adapter (eval_az_match --legacy-b) and debugging; the seq net ignores it
// (the history implies the trick).
struct EvalFeatures {
  std::array<int, 13> hand;     // searcher's exact hand counts
  std::array<int, 13> opp_max;  // opponent upper-bound counts (thermo)
  std::array<int, 13> trick;    // current trick rank counts (all 0 == lead)
  int opp_size;                 // opponent hand size
  int our_size;                 // searcher hand size
  bool owner_to_move;           // true at our-turn leaves, false at opp-turn
  std::vector<int> path_tokens; // move ids root -> leaf (after the game prefix)
};

// All four heads of the unified net, from the searcher's (hand-owner's) POV.
struct NetEval {
  float value = 0.5f;  // P(searcher wins), read at our-turn leaves
  std::array<float, AZ_PLAYER_HEAD_DIM> policy{};   // factored player logits
  std::array<float, AZ_OPP_HEAD_DIM> behavior{};    // opp behavior logits
  std::array<float, AZ_OPP_HEAD_DIM> qa{};          // per-move q_a in [0,1]
};

class Evaluator {
public:
  virtual ~Evaluator() = default;
  virtual NetEval eval(const EvalFeatures &f) = 0;
};

}  // namespace az_search

#endif  // AZ_SEARCH_EVALUATOR_H
