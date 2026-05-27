#include "nn_eval.h"

#include "features.h"
#include "nn_encode.h"  // encode_exact / encode_thermo / ENCODING_DIM

namespace az_search {

NNEvaluator::NNEvaluator(const std::string &player_path,
                         const std::string &opp_path, torch::Device device)
    : player_(torch::jit::load(player_path, device)),
      opp_(torch::jit::load(opp_path, device)), device_(device) {
  player_.eval();
  opp_.eval();
}

std::vector<PlayerEval>
NNEvaluator::eval_players(const std::vector<PlayerFeatures> &feats) const {
  const int B = static_cast<int>(feats.size());
  if (B == 0) return {};

  auto fopts = torch::TensorOptions().dtype(torch::kFloat32);
  auto t_hand = torch::zeros({B, ENCODING_DIM}, fopts);
  auto t_opp = torch::zeros({B, ENCODING_DIM}, fopts);
  auto t_trick = torch::zeros({B, ENCODING_DIM}, fopts);
  auto t_osz = torch::zeros({B}, fopts);
  auto t_usz = torch::zeros({B}, fopts);

  float *h = t_hand.data_ptr<float>();
  float *o = t_opp.data_ptr<float>();
  float *tr = t_trick.data_ptr<float>();
  float *osz = t_osz.data_ptr<float>();
  float *usz = t_usz.data_ptr<float>();

  for (int i = 0; i < B; ++i) {
    encode_exact(feats[i].hand, h + i * ENCODING_DIM);
    encode_thermo(feats[i].opp_max, o + i * ENCODING_DIM);
    encode_exact(feats[i].trick, tr + i * ENCODING_DIM);
    osz[i] = feats[i].opp_size / kSizeNorm;
    usz[i] = feats[i].our_size / kSizeNorm;
  }

  torch::NoGradGuard ng;
  auto out = player_
                 .forward({t_hand.to(device_), t_opp.to(device_),
                           t_trick.to(device_), t_osz.to(device_),
                           t_usz.to(device_)})
                 .toTuple();
  auto value = out->elements()[0].toTensor().to(torch::kCPU).contiguous();
  auto logits = out->elements()[1].toTensor().to(torch::kCPU).contiguous();
  const float *vp = value.data_ptr<float>();
  const float *lp = logits.data_ptr<float>();

  std::vector<PlayerEval> result(B);
  for (int i = 0; i < B; ++i) {
    result[i].value = vp[i];
    std::copy(lp + i * AZ_PLAYER_HEAD_DIM, lp + (i + 1) * AZ_PLAYER_HEAD_DIM,
              result[i].logits.begin());
  }
  return result;
}

std::vector<OppEval>
NNEvaluator::eval_opps(const std::vector<OppFeatures> &feats) const {
  const int B = static_cast<int>(feats.size());
  if (B == 0) return {};

  auto fopts = torch::TensorOptions().dtype(torch::kFloat32);
  auto t_opp = torch::zeros({B, ENCODING_DIM}, fopts);
  auto t_trick = torch::zeros({B, ENCODING_DIM}, fopts);
  auto t_osz = torch::zeros({B}, fopts);
  auto t_usz = torch::zeros({B}, fopts);

  float *o = t_opp.data_ptr<float>();
  float *tr = t_trick.data_ptr<float>();
  float *osz = t_osz.data_ptr<float>();
  float *usz = t_usz.data_ptr<float>();

  for (int i = 0; i < B; ++i) {
    encode_thermo(feats[i].opp_max, o + i * ENCODING_DIM);
    encode_exact(feats[i].trick, tr + i * ENCODING_DIM);
    osz[i] = feats[i].opp_size / kSizeNorm;
    usz[i] = feats[i].our_size / kSizeNorm;
  }

  torch::NoGradGuard ng;
  auto out = opp_
                 .forward({t_opp.to(device_), t_trick.to(device_),
                           t_osz.to(device_), t_usz.to(device_)})
                 .toTuple();
  auto value = out->elements()[0].toTensor().to(torch::kCPU).contiguous();
  auto logits = out->elements()[1].toTensor().to(torch::kCPU).contiguous();
  const float *vp = value.data_ptr<float>();
  const float *lp = logits.data_ptr<float>();

  std::vector<OppEval> result(B);
  for (int i = 0; i < B; ++i) {
    result[i].value = vp[i];
    std::copy(lp + i * AZ_OPP_HEAD_DIM, lp + (i + 1) * AZ_OPP_HEAD_DIM,
              result[i].logits.begin());
  }
  return result;
}

}  // namespace az_search
