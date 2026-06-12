#ifndef AZ_PI_PI_EVAL_H
#define AZ_PI_PI_EVAL_H

// Perfect-information AlphaZero leaf-evaluation interface. Torch-free so the
// search core links into test_core; the production evaluator (pi_nn_eval) is a
// TorchScript-backed subclass.
//
// The PI net is MEMORYLESS: it sees only the mover's exact hand, the opponent's
// exact hand, and the current trick (the move on the table, all-zero = lead).
// No discard pile, no move history. Value is P(the side to move wins this game)
// in [0,1] (sigmoid). Policy is the factored 138-dim player head (composed to
// concrete moves via az_search::player_composed_logit / the C-matrix).

#include "az_search/considered_moves.h"  // AZ_PLAYER_HEAD_DIM

#include <array>
#include <vector>

namespace az_pi {

struct PiEvalFeatures {
  std::array<int, 13> hand;      // mover's exact remaining cards
  std::array<int, 13> opp_hand;  // opponent's exact remaining cards
  std::array<int, 13> trick;     // trick_counts(last_move_id); all-zero = lead
  int our_size = 0;              // mover hand size (informational; net input)
  int opp_size = 0;             // opponent hand size
};

struct PiNetEval {
  float value = 0.5f;  // P(mover wins), in [0,1]
  std::array<float, az_search::AZ_PLAYER_HEAD_DIM> policy{};  // raw logits
};

// Batched leaf evaluator. The search parks each pending leaf, then flushes one
// batch per turn (no KV cache — the net is memoryless).
class PiEvaluator {
 public:
  virtual ~PiEvaluator() = default;
  virtual std::vector<PiNetEval> eval_batch(
      const std::vector<PiEvalFeatures> &feats) = 0;
};

}  // namespace az_pi

#endif  // AZ_PI_PI_EVAL_H
