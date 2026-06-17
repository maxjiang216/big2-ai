#include "az_pi/pi_nn_eval.h"

#include "nn_encode.h"  // encode_exact, ENCODING_DIM

#include <algorithm>

namespace az_pi {

PiNNEvaluator::PiNNEvaluator(const std::string &model_path,
                             torch::Device device, bool pts_inputs)
    : model_(torch::jit::load(model_path, device)),
      device_(device),
      pts_inputs_(pts_inputs) {
  model_.eval();
}

std::vector<PiNetEval> PiNNEvaluator::eval_batch(
    const std::vector<PiEvalFeatures> &feats) {
  const int B = static_cast<int>(feats.size());
  std::vector<PiNetEval> result(B);
  if (B == 0) return result;

  auto fopts = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor hand = torch::zeros({B, ENCODING_DIM}, fopts);
  torch::Tensor opp = torch::zeros({B, ENCODING_DIM}, fopts);
  torch::Tensor trick = torch::zeros({B, ENCODING_DIM}, fopts);
  torch::Tensor mpts = torch::zeros({B}, fopts);
  torch::Tensor opts = torch::zeros({B}, fopts);

  float *h = hand.data_ptr<float>();
  float *o = opp.data_ptr<float>();
  float *tr = trick.data_ptr<float>();
  float *mp = mpts.data_ptr<float>();
  float *op = opts.data_ptr<float>();
  for (int i = 0; i < B; ++i) {
    const PiEvalFeatures &f = feats[i];
    encode_exact(f.hand, h + i * ENCODING_DIM);
    encode_exact(f.opp_hand, o + i * ENCODING_DIM);  // exact (perfect info)
    encode_exact(f.trick, tr + i * ENCODING_DIM);
    if (pts_inputs_) {  // series-aware net: scores (already /kSeriesTarget)
      mp[i] = f.my_pts;
      op[i] = f.opp_pts;
    } else {  // legacy score-blind net: hand sizes, positional (opp, our)
      mp[i] = f.opp_size / 16.0f;
      op[i] = f.our_size / 16.0f;
    }
  }

  torch::NoGradGuard ng;
  auto out = model_
                 .forward({hand.to(device_), opp.to(device_), trick.to(device_),
                           mpts.to(device_), opts.to(device_)})
                 .toTuple();
  auto value = out->elements()[0].toTensor().to(torch::kCPU).contiguous();
  auto logits = out->elements()[1].toTensor().to(torch::kCPU).contiguous();
  const float *vp = value.data_ptr<float>();
  const float *lp = logits.data_ptr<float>();
  for (int i = 0; i < B; ++i) {
    result[i].value = vp[i];
    std::copy(lp + i * az_search::AZ_PLAYER_HEAD_DIM,
              lp + (i + 1) * az_search::AZ_PLAYER_HEAD_DIM,
              result[i].policy.begin());
  }
  return result;
}

}  // namespace az_pi
