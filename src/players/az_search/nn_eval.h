#ifndef AZ_SEARCH_NN_EVAL_H
#define AZ_SEARCH_NN_EVAL_H

// Batched LibTorch inference for the two az_search nets. Emits RAW value +
// policy/behavior logits; the legal-move mask and the softmax over the legal
// subset are applied by the search/training, never here (see considered_moves.h
// for the head layout). Torch-dependent — only compiled into torch-enabled
// targets (-DBIG2_WITH_TORCH).

#include "considered_moves.h"
#include "evaluator.h"  // Evaluator interface + Player/Opp Features/Eval structs

#include <torch/script.h>

#include <array>
#include <string>
#include <vector>

namespace az_search {

// Loads both TorchScript nets once; shared (read-only) across threads.
// Implements the synchronous Evaluator interface (used by the single-game
// player path) and exposes batched methods for parallel self-play.
class NNEvaluator : public Evaluator {
public:
  NNEvaluator(const std::string &player_path, const std::string &opp_path,
              torch::Device device);

  // Batched (used by parallel self-play).
  std::vector<PlayerEval> eval_players(const std::vector<PlayerFeatures> &feats) const;
  std::vector<OppEval> eval_opps(const std::vector<OppFeatures> &feats) const;

  // Synchronous Evaluator interface (single-game player path).
  PlayerEval eval_player(const PlayerFeatures &f) override {
    return eval_players({f})[0];
  }
  OppEval eval_opp(const OppFeatures &f) override { return eval_opps({f})[0]; }

  torch::Device device() const { return device_; }

private:
  mutable torch::jit::Module player_;
  mutable torch::jit::Module opp_;
  torch::Device device_;
};

}  // namespace az_search

#endif  // AZ_SEARCH_NN_EVAL_H
