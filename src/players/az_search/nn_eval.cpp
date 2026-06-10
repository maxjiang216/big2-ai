#include "nn_eval.h"

#include "features.h"
#include "nn_encode.h"  // encode_exact / encode_thermo / ENCODING_DIM

#include <algorithm>
#include <stdexcept>

namespace az_search {

NNEvaluator::NNEvaluator(const std::string &model_path, torch::Device device,
                         int max_slots, bool use_kv_cache, bool fp16_pool)
    : net_(torch::jit::load(model_path, device)),
      device_(device),
      use_kv_cache_(use_kv_cache) {
  net_.eval();
  // config() = [n_layers, n_heads, head_dim, d_model, seq_cap]
  auto cfg = net_.get_method("config")({}).toIntList();
  n_layers_ = (int)cfg[0];
  n_heads_ = (int)cfg[1];
  head_dim_ = (int)cfg[2];
  d_model_ = (int)cfg[3];
  seq_cap_ = (int)cfg[4];

  plen_.assign(max_slots, 1);  // empty prefix = BOS only
  raw_hist_.assign(max_slots, {});
  if (use_kv_cache_) {
    auto kv_dtype = fp16_pool ? torch::kFloat16 : torch::kFloat32;
    kv_pool_ = torch::zeros(
        {max_slots, n_layers_, 2, n_heads_, seq_cap_, head_dim_},
        torch::TensorOptions().dtype(kv_dtype).device(device_));
    hlast_pool_ = torch::zeros({max_slots, d_model_},
                               torch::TensorOptions().dtype(torch::kFloat32)
                                   .device(device_));
  }
}

void NNEvaluator::set_prefixes(
    const std::vector<int> &slot_ids,
    const std::vector<const std::vector<int> *> &histories) {
  const int B = (int)slot_ids.size();
  if (B == 0) return;
  int pmax = 0;
  for (const auto *h : histories) pmax = std::max(pmax, (int)h->size());
  if (pmax + 1 > seq_cap_)
    throw std::runtime_error("history exceeds net seq_cap");

  for (int i = 0; i < B; ++i) raw_hist_[slot_ids[i]] = *histories[i];
  for (int i = 0; i < B; ++i)
    plen_[slot_ids[i]] = (int64_t)histories[i]->size() + 1;  // + BOS
  if (!use_kv_cache_) return;

  auto lopts = torch::TensorOptions().dtype(torch::kInt64);
  auto tokens = torch::zeros({B, pmax}, lopts);
  auto lens = torch::zeros({B}, lopts);
  auto *tp = tokens.data_ptr<int64_t>();
  auto *lp = lens.data_ptr<int64_t>();
  for (int i = 0; i < B; ++i) {
    const auto &h = *histories[i];
    for (int j = 0; j < (int)h.size(); ++j) tp[i * pmax + j] = h[j];
    lp[i] = (int64_t)h.size();
  }

  torch::NoGradGuard ng;
  auto out = net_.get_method("encode_prefix")({tokens.to(device_),
                                               lens.to(device_)})
                 .toTuple();
  auto kv = out->elements()[0].toTensor();          // [B, L, 2, H, pmax+1, Dh]
  auto h_last = out->elements()[1].toTensor();      // [B, d_model]

  auto slots = torch::from_blob((void *)slot_ids.data(), {B},
                                torch::TensorOptions().dtype(torch::kInt32))
                   .to(device_, torch::kInt64);
  // Write into the leading [.., :pmax+1, ..] slice of the pool rows; the stale
  // tail beyond plen is masked out by forward_leaf.
  kv_pool_.slice(4, 0, pmax + 1)
      .index_copy_(0, slots, kv.to(kv_pool_.scalar_type()));
  hlast_pool_.index_copy_(0, slots, h_last);
}

std::vector<NetEval>
NNEvaluator::eval_batch(const std::vector<int> &slot_ids,
                        const std::vector<EvalFeatures> &feats) {
  const int B = (int)feats.size();
  if (B == 0) return {};

  auto fopts = torch::TensorOptions().dtype(torch::kFloat32);
  auto lopts = torch::TensorOptions().dtype(torch::kInt64);
  auto t_hand = torch::zeros({B, ENCODING_DIM}, fopts);
  auto t_opp = torch::zeros({B, ENCODING_DIM}, fopts);
  auto t_osz = torch::zeros({B}, fopts);
  auto t_usz = torch::zeros({B}, fopts);
  auto t_otm = torch::zeros({B}, fopts);
  float *h = t_hand.data_ptr<float>();
  float *o = t_opp.data_ptr<float>();
  for (int i = 0; i < B; ++i) {
    encode_exact(feats[i].hand, h + i * ENCODING_DIM);
    encode_thermo(feats[i].opp_max, o + i * ENCODING_DIM);
    t_osz.data_ptr<float>()[i] = feats[i].opp_size / kSizeNorm;
    t_usz.data_ptr<float>()[i] = feats[i].our_size / kSizeNorm;
    t_otm.data_ptr<float>()[i] = feats[i].owner_to_move ? 1.0f : 0.0f;
  }

  torch::NoGradGuard ng;
  c10::intrusive_ptr<c10::ivalue::Tuple> out;
  if (use_kv_cache_) {
    int tmax = 1;
    for (const auto &f : feats) tmax = std::max(tmax, (int)f.path_tokens.size());
    auto path = torch::zeros({B, tmax}, lopts);
    auto path_len = torch::zeros({B}, lopts);
    auto plen = torch::zeros({B}, lopts);
    auto *pp = path.data_ptr<int64_t>();
    for (int i = 0; i < B; ++i) {
      const auto &t = feats[i].path_tokens;
      for (int j = 0; j < (int)t.size(); ++j) pp[i * tmax + j] = t[j];
      path_len.data_ptr<int64_t>()[i] = (int64_t)t.size();
      plen.data_ptr<int64_t>()[i] = plen_[slot_ids[i]];
    }
    auto slots = torch::from_blob((void *)slot_ids.data(), {B},
                                  torch::TensorOptions().dtype(torch::kInt32))
                     .to(device_, torch::kInt64);
    auto kv = kv_pool_.index_select(0, slots);
    auto h_last = hlast_pool_.index_select(0, slots);
    out = net_.get_method("forward_leaf")({kv, plen.to(device_), h_last,
                                           path.to(device_),
                                           path_len.to(device_),
                                           t_hand.to(device_),
                                           t_opp.to(device_),
                                           t_osz.to(device_),
                                           t_usz.to(device_),
                                           t_otm.to(device_)})
              .toTuple();
  } else {
    // Full recompute: tokens = prefix + path per row, readout at the end.
    int lmax = 1;
    for (int i = 0; i < B; ++i)
      lmax = std::max(lmax, (int)(raw_hist_[slot_ids[i]].size() +
                                  feats[i].path_tokens.size()));
    auto tokens = torch::zeros({B, lmax}, lopts);
    auto idx = torch::zeros({B}, lopts);
    auto *tp = tokens.data_ptr<int64_t>();
    for (int i = 0; i < B; ++i) {
      const auto &pre = raw_hist_[slot_ids[i]];
      const auto &t = feats[i].path_tokens;
      int k = 0;
      for (int m : pre) tp[i * lmax + k++] = m;
      for (int m : t) tp[i * lmax + k++] = m;
      idx.data_ptr<int64_t>()[i] = k;  // hidden index = total move count
    }
    auto H = net_.get_method("forward_full")({tokens.to(device_)}).toTensor();
    auto gather_idx = idx.to(device_).view({B, 1, 1}).expand({B, 1, d_model_});
    auto hsel = H.gather(1, gather_idx).squeeze(1);
    out = net_.get_method("readout")({hsel, t_hand.to(device_),
                                      t_opp.to(device_), t_osz.to(device_),
                                      t_usz.to(device_), t_otm.to(device_)})
              .toTuple();
  }

  auto value = out->elements()[0].toTensor().to(torch::kCPU).contiguous();
  auto policy = out->elements()[1].toTensor().to(torch::kCPU).contiguous();
  auto behavior = out->elements()[2].toTensor().to(torch::kCPU).contiguous();
  auto qa = out->elements()[3].toTensor().to(torch::kCPU).contiguous();
  const float *vp = value.data_ptr<float>();
  const float *pp = policy.data_ptr<float>();
  const float *bp = behavior.data_ptr<float>();
  const float *qp = qa.data_ptr<float>();

  std::vector<NetEval> result(B);
  for (int i = 0; i < B; ++i) {
    result[i].value = vp[i];
    std::copy(pp + i * AZ_PLAYER_HEAD_DIM, pp + (i + 1) * AZ_PLAYER_HEAD_DIM,
              result[i].policy.begin());
    std::copy(bp + i * AZ_OPP_HEAD_DIM, bp + (i + 1) * AZ_OPP_HEAD_DIM,
              result[i].behavior.begin());
    std::copy(qp + i * AZ_OPP_HEAD_DIM, qp + (i + 1) * AZ_OPP_HEAD_DIM,
              result[i].qa.begin());
  }
  return result;
}

}  // namespace az_search
