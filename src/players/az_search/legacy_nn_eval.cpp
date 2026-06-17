#include "legacy_nn_eval.h"

#include "features.h"
#include "nn_encode.h"

#include <algorithm>

namespace az_search {

LegacyNNEvaluator::LegacyNNEvaluator(const std::string &player_path,
                                     const std::string &opp_path,
                                     torch::Device device)
    : player_(torch::jit::load(player_path, device)),
      opp_(torch::jit::load(opp_path, device)), device_(device) {
  player_.eval();
  opp_.eval();
}

namespace {

// The old nets share the input signature (hand, opp, trick, opp_size, our_size).
struct LegacyInputs {
  torch::Tensor hand, opp, trick, osz, usz;
};

LegacyInputs build_inputs(const std::vector<EvalFeatures> &feats,
                          const std::vector<int> &rows) {
  const int B = (int)rows.size();
  auto fopts = torch::TensorOptions().dtype(torch::kFloat32);
  LegacyInputs in{torch::zeros({B, ENCODING_DIM}, fopts),
                  torch::zeros({B, ENCODING_DIM}, fopts),
                  torch::zeros({B, ENCODING_DIM}, fopts),
                  torch::zeros({B}, fopts), torch::zeros({B}, fopts)};
  float *h = in.hand.data_ptr<float>();
  float *o = in.opp.data_ptr<float>();
  float *tr = in.trick.data_ptr<float>();
  for (int i = 0; i < B; ++i) {
    const EvalFeatures &f = feats[rows[i]];
    encode_exact(f.hand, h + i * ENCODING_DIM);
    encode_thermo(f.opp_max, o + i * ENCODING_DIM);
    encode_exact(f.trick, tr + i * ENCODING_DIM);
    in.osz.data_ptr<float>()[i] = f.opp_size / kSizeNorm;
    in.usz.data_ptr<float>()[i] = f.our_size / kSizeNorm;
  }
  return in;
}

}  // namespace

std::vector<NetEval>
LegacyNNEvaluator::eval_batch(const std::vector<EvalFeatures> &feats) {
  std::vector<int> prow, orow;
  for (int i = 0; i < (int)feats.size(); ++i)
    (feats[i].owner_to_move ? prow : orow).push_back(i);
  std::vector<NetEval> result(feats.size());

  torch::NoGradGuard ng;
  if (!prow.empty()) {
    LegacyInputs in = build_inputs(feats, prow);
    auto out = player_
                   .forward({in.hand.to(device_), in.opp.to(device_),
                             in.trick.to(device_), in.osz.to(device_),
                             in.usz.to(device_)})
                   .toTuple();
    auto value = out->elements()[0].toTensor().to(torch::kCPU).contiguous();
    auto logits = out->elements()[1].toTensor().to(torch::kCPU).contiguous();
    const float *vp = value.data_ptr<float>();
    const float *lp = logits.data_ptr<float>();
    for (int i = 0; i < (int)prow.size(); ++i) {
      result[prow[i]].value = vp[i];
      std::copy(lp + i * AZ_PLAYER_HEAD_DIM, lp + (i + 1) * AZ_PLAYER_HEAD_DIM,
                result[prow[i]].policy.begin());
    }
  }
  if (!orow.empty()) {
    LegacyInputs in = build_inputs(feats, orow);
    auto out = opp_
                   .forward({in.hand.to(device_), in.opp.to(device_),
                             in.trick.to(device_), in.osz.to(device_),
                             in.usz.to(device_)})
                   .toTuple();
    auto mv = out->elements()[0].toTensor().to(torch::kCPU).contiguous();
    auto logits = out->elements()[1].toTensor().to(torch::kCPU).contiguous();
    const float *mp = mv.data_ptr<float>();
    const float *lp = logits.data_ptr<float>();
    for (int i = 0; i < (int)orow.size(); ++i) {
      std::copy(mp + i * AZ_OPP_HEAD_DIM, mp + (i + 1) * AZ_OPP_HEAD_DIM,
                result[orow[i]].qa.begin());
      std::copy(lp + i * AZ_OPP_HEAD_DIM, lp + (i + 1) * AZ_OPP_HEAD_DIM,
                result[orow[i]].behavior.begin());
    }
  }
  return result;
}

}  // namespace az_search
