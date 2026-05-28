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

// ---------------------------------------------------------------------------
// NN input / output structs (torch-free, so the search core and a test stub
// can use them without LibTorch). nn_eval.h fills these from TorchScript.
// ---------------------------------------------------------------------------

// Raw per-rank counts + sizes for one position. Sizes are RAW (un-normalised);
// the evaluator divides by kSizeNorm when building tensors.
struct PlayerFeatures {
  std::array<int, 13> hand;     // our exact hand counts
  std::array<int, 13> opp_max;  // opponent upper-bound counts (thermo)
  std::array<int, 13> trick;    // current trick rank counts (all 0 == lead)
  int opp_size;
  int our_size;
};

struct OppFeatures {
  std::array<int, 13> hand;     // observer's (searcher's) exact hand counts
  std::array<int, 13> opp_max;  // mover upper-bound counts (thermo)
  std::array<int, 13> trick;    // current trick rank counts
  int opp_size;                 // mover (opponent) hand size
  int our_size;                 // observer (searcher) hand size
};

struct PlayerEval {
  float value;  // P(player-to-move wins), in [0,1]
  std::array<float, AZ_PLAYER_HEAD_DIM> logits;
};

// Opponent net output. The opponent net now takes our exact hand too, so each
// head slot carries a PER-MOVE value q_a = P(observer/searcher wins after the
// opponent plays move a). The node's scalar value is not a head output — the
// search derives it as Sum_a prior(a) * q_a (control-variate baseline).
struct OppEval {
  std::array<float, AZ_OPP_HEAD_DIM> move_value;  // per-slot q_a, each in [0,1]
  std::array<float, AZ_OPP_HEAD_DIM> logits;      // behavior logits
};

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
