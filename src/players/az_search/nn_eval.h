#ifndef AZ_SEARCH_NN_EVAL_H
#define AZ_SEARCH_NN_EVAL_H

// Batched LibTorch inference for the unified history-transformer net
// (nn/model_az_seq.py, exported TorchScript). Emits RAW value/logits per the
// four heads; the legal-move mask and softmax are applied by the search.
//
// ROOT-PREFIX KV CACHE: per slot (one slot per concurrent game seat), the real
// game-history prefix is encoded ONCE per turn (set_prefixes) into a device-
// resident KV pool; each leaf eval then runs the transformer only over the
// in-tree path tokens attending to [cached prefix + path]. `--no-kv-cache`
// (use_kv_cache=false) instead recomputes the full sequence per eval — the
// correctness/debug reference.
//
// Torch-dependent — only compiled into torch-enabled targets
// (-DBIG2_WITH_TORCH).

#include "considered_moves.h"
#include "evaluator.h"

#include <torch/script.h>

#include <string>
#include <vector>

namespace az_search {

class NNEvaluator : public Evaluator {
public:
  NNEvaluator(const std::string &model_path, torch::Device device,
              int max_slots = 1, bool use_kv_cache = true,
              bool fp16_pool = true);

  // Re-encode the game-history prefixes of the given slots (batched). Call
  // once per real turn per slot, BEFORE that slot's leaf evals.
  void set_prefixes(const std::vector<int> &slot_ids,
                    const std::vector<const std::vector<int> *> &histories);
  void set_prefix(const std::vector<int> &history) {
    set_prefixes({0}, {&history});
  }

  // Batched leaf evals; feats[i] belongs to slot_ids[i]'s prefix.
  std::vector<NetEval> eval_batch(const std::vector<int> &slot_ids,
                                  const std::vector<EvalFeatures> &feats);

  // Synchronous Evaluator interface (single-game path, slot 0).
  NetEval eval(const EvalFeatures &f) override {
    return eval_batch({0}, {f})[0];
  }

  torch::Device device() const { return device_; }
  int seq_cap() const { return seq_cap_; }

private:
  mutable torch::jit::Module net_;
  torch::Device device_;
  int n_layers_, n_heads_, head_dim_, d_model_, seq_cap_;
  bool use_kv_cache_;
  torch::Tensor kv_pool_;     // [slots, L, 2, H, seq_cap, Dh] (fp16 or fp32)
  torch::Tensor hlast_pool_;  // [slots, d_model] fp32
  std::vector<int64_t> plen_; // per-slot valid prefix length INCL. BOS
  std::vector<std::vector<int>> raw_hist_;  // per-slot prefix (no-cache mode)
};

}  // namespace az_search

#endif  // AZ_SEARCH_NN_EVAL_H
