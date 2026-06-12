#ifndef AZ_PI_PI_NN_EVAL_H
#define AZ_PI_PI_NN_EVAL_H

// TorchScript-backed PiEvaluator for the perfect-information AZ net. Memoryless:
// each leaf is a single forward over (hand, opp, trick, opp_size, our_size); no
// KV cache. Only compiled with -DBIG2_WITH_TORCH.

#include "az_pi/pi_eval.h"

#include <string>

#include <torch/script.h>

namespace az_pi {

class PiNNEvaluator : public PiEvaluator {
 public:
  PiNNEvaluator(const std::string &model_path, torch::Device device);
  std::vector<PiNetEval> eval_batch(
      const std::vector<PiEvalFeatures> &feats) override;

 private:
  torch::jit::script::Module model_;
  torch::Device device_;
};

}  // namespace az_pi

#endif  // AZ_PI_PI_NN_EVAL_H
