#ifndef AZ_SEARCH_FEATURES_H
#define AZ_SEARCH_FEATURES_H

// Position -> NN feature derivations for az_search. Torch-free (only rank-count
// arrays); the float-tensor encoding (encode_exact / encode_thermo) lives in
// nn_eval. Scalar hand sizes are normalised by 16 to match the Python training
// pipeline (nn/dataset.py divides sizes by 16).

#include "considered_moves.h"
#include "move.h"
#include "util.h"

#include <algorithm>
#include <array>

namespace az_search {

constexpr float kSizeNorm = 16.0f;

// NN input/output structs live in evaluator.h (unified EvalFeatures/NetEval);
// this header keeps the torch-free feature derivations shared by the search
// core, the evaluators, and datagen.

// Upper bound on the opponent's per-rank holdings, given our hand and the
// discard pile: max in deck minus what we hold minus what's been played. This
// is the union (our cards + discard), so it is invariant to which of OUR cards
// we hypothetically play within a search line — the basis for sharing opponent
// NN evals across nodes that differ only in our hidden hand.
inline std::array<int, 13> opp_max_counts(const std::array<int, 13> &our_hand,
                                          const std::array<int, 13> &discard) {
  std::array<int, 13> o{};
  for (int r = 0; r < 13; ++r)
    o[r] = std::max(0, max_cards_in_deck_for_rank(r) - our_hand[r] - discard[r]);
  return o;
}

// Rank counts of the move on the table (the last move that must be beaten).
// An empty trick (we hold the initiative) is represented by kPASS -> all zeros,
// which encode_exact turns into the all-zero "initiative" vector.
inline std::array<int, 13> trick_counts(int last_move_id) {
  std::array<int, 13> t{};
  if (last_move_id != kPASS)
    for (int r = 0; r < 13; ++r)
      t[r] = MOVE_TO_CARDS[last_move_id][r];
  return t;
}

}  // namespace az_search

#endif  // AZ_SEARCH_FEATURES_H
