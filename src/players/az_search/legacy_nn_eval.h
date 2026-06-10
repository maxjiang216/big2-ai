#ifndef AZ_SEARCH_LEGACY_NN_EVAL_H
#define AZ_SEARCH_LEGACY_NN_EVAL_H

// Adapter that drives the OLD two-net (Big2NetAZ + Big2NetOpp) TorchScript
// models through the NEW unified Evaluator interface, so the pre-memory
// champion can play paired deals against the seq net (eval_az_match
// --legacy-b). Ignores path_tokens (the old nets are history-free); reads
// hand/opp_max/trick/sizes from EvalFeatures — the exact inputs the old
// pipeline fed. Per eval only the heads the search will read are computed:
// the player net at owner-to-move leaves, the opp net otherwise.
//
// Torch-dependent — only compiled into torch-enabled targets.

#include "considered_moves.h"
#include "evaluator.h"

#include <torch/script.h>

#include <string>
#include <vector>

namespace az_search {

class LegacyNNEvaluator : public Evaluator {
public:
  LegacyNNEvaluator(const std::string &player_path, const std::string &opp_path,
                    torch::Device device);

  // Batched: splits rows by owner_to_move internally, one forward per net.
  std::vector<NetEval> eval_batch(const std::vector<EvalFeatures> &feats);

  NetEval eval(const EvalFeatures &f) override { return eval_batch({f})[0]; }

  torch::Device device() const { return device_; }

private:
  mutable torch::jit::Module player_;
  mutable torch::jit::Module opp_;
  torch::Device device_;
};

}  // namespace az_search

#endif  // AZ_SEARCH_LEGACY_NN_EVAL_H
