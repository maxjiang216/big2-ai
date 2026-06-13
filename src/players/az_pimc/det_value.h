#ifndef AZ_PIMC_DET_VALUE_H
#define AZ_PIMC_DET_VALUE_H

// Determinization-in-tree (Design 1): replace an az_search leaf's mushy II value
// with a SHARP perfect-information value, averaged over belief-sampled opponent
// worlds. The search tree, PUCT, and opponent expansion are unchanged — only the
// leaf scalar `NetEval.value` (P(searcher wins)) is overridden, so one action per
// infoset is preserved (no strategy fusion at the decision).
//
// First cut: UNIFORM belief — sample opponent hands of the right size from the
// leaf thermo (EvalFeatures.opp_max = deck - searcher_hand - discard). Needs only
// the leaf's EvalFeatures (no history prefix), so it slots in as a batch
// post-process on the searcher's leaf evals. AR-belief sampling is the upgrade.

#include "az_pi/pi_eval.h"
#include "az_search/evaluator.h"
#include "nn_encode.h"  // encode_thermo, ENCODING_DIM

#include <torch/script.h>

#include <algorithm>
#include <array>
#include <random>
#include <vector>

namespace az_pimc {

inline std::array<int, 13> uniform_world(const std::array<int, 13> &thermo,
                                         int sz, std::mt19937 &rng) {
  std::vector<int> pool;
  for (int r = 0; r < 13; ++r)
    for (int c = 0; c < thermo[r]; ++c) pool.push_back(r);
  std::shuffle(pool.begin(), pool.end(), rng);
  std::array<int, 13> h{};
  const int take = std::min<int>(sz, static_cast<int>(pool.size()));
  for (int i = 0; i < take; ++i) ++h[pool[i]];
  return h;
}

// Determinized P(searcher wins) for each leaf. Batches all (leaf x N) worlds into
// one PI eval call. The PI value is P(mover wins); at opp-turn leaves the searcher
// is the mover's opponent, so we use 1 - value.
inline std::vector<float> determinized_values(
    const std::vector<az_search::EvalFeatures> &feats, az_pi::PiEvaluator &pi,
    int N, std::mt19937 &rng) {
  std::vector<az_pi::PiEvalFeatures> pf;
  pf.reserve(feats.size() * N);
  for (const auto &f : feats) {
    for (int i = 0; i < N; ++i) {
      auto world = uniform_world(f.opp_max, f.opp_size, rng);
      az_pi::PiEvalFeatures p;
      p.trick = f.trick;
      if (f.owner_to_move) {  // our turn: mover = searcher
        p.hand = f.hand;
        p.opp_hand = world;
        p.our_size = f.our_size;
        p.opp_size = f.opp_size;
        p.my_pts = f.my_pts / 50.0f;
        p.opp_pts = f.opp_pts / 50.0f;
      } else {  // opp turn: mover = opponent (sampled world hand)
        p.hand = world;
        p.opp_hand = f.hand;
        p.our_size = f.opp_size;
        p.opp_size = f.our_size;
        p.my_pts = f.opp_pts / 50.0f;
        p.opp_pts = f.my_pts / 50.0f;
      }
      pf.push_back(p);
    }
  }
  const auto ev = pi.eval_batch(pf);
  std::vector<float> out(feats.size(), 0.5f);
  for (size_t k = 0; k < feats.size(); ++k) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) {
      const float v = ev[k * N + i].value;  // P(mover wins)
      s += feats[k].owner_to_move ? v : (1.0 - v);
    }
    out[k] = static_cast<float>(s / N);
  }
  return out;
}

// Score PI features (mover/opp already oriented) -> P(searcher wins) per leaf.
inline std::vector<float> score_worlds(
    const std::vector<az_search::EvalFeatures> &feats,
    const std::vector<std::array<int, 13>> &worlds, az_pi::PiEvaluator &pi,
    int N) {
  std::vector<az_pi::PiEvalFeatures> pf;
  pf.reserve(feats.size() * N);
  for (size_t k = 0; k < feats.size(); ++k) {
    const auto &f = feats[k];
    for (int i = 0; i < N; ++i) {
      const auto &world = worlds[k * N + i];
      az_pi::PiEvalFeatures p;
      p.trick = f.trick;
      if (f.owner_to_move) {
        p.hand = f.hand; p.opp_hand = world;
        p.our_size = f.our_size; p.opp_size = f.opp_size;
      } else {
        p.hand = world; p.opp_hand = f.hand;
        p.our_size = f.opp_size; p.opp_size = f.our_size;
      }
      pf.push_back(p);
    }
  }
  const auto ev = pi.eval_batch(pf);
  std::vector<float> out(feats.size(), 0.5f);
  for (size_t k = 0; k < feats.size(); ++k) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) {
      const float v = ev[k * N + i].value;
      s += feats[k].owner_to_move ? v : (1.0 - v);
    }
    out[k] = static_cast<float>(s / N);
  }
  return out;
}

// AR-belief determinized values: sample opp worlds from the scripted belief net
// (Big2NetII.sample_opp), conditioned on each leaf's FULL history (game prefix +
// in-tree path), γ-mixed with uniform. Then score with the PI evaluator.
inline std::vector<float> determinized_values_ar(
    const std::vector<az_search::EvalFeatures> &feats,
    const std::vector<std::vector<int>> &full_hist, torch::jit::Module &belief,
    az_pi::PiEvaluator &pi, int N, float gamma, std::mt19937 &rng,
    torch::Device device) {
  const int B = static_cast<int>(feats.size());
  int maxT = 1;
  for (const auto &h : full_hist) maxT = std::max(maxT, (int)h.size());
  auto i64 = torch::TensorOptions().dtype(torch::kInt64);
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor tokens = torch::zeros({B, maxT}, i64);
  torch::Tensor hist_idx = torch::zeros({B}, i64);
  torch::Tensor oppenc = torch::zeros({B, ENCODING_DIM}, f32);
  torch::Tensor thermo13 = torch::zeros({B, 13}, f32);
  torch::Tensor oppsz = torch::zeros({B}, f32);
  int64_t *tk = tokens.data_ptr<int64_t>();
  int64_t *hi = hist_idx.data_ptr<int64_t>();
  float *oe = oppenc.data_ptr<float>();
  float *th = thermo13.data_ptr<float>();
  float *os = oppsz.data_ptr<float>();
  for (int k = 0; k < B; ++k) {
    const auto &h = full_hist[k];
    for (int j = 0; j < (int)h.size(); ++j) tk[k * maxT + j] = h[j];
    hi[k] = (int)h.size();
    encode_thermo(feats[k].opp_max, oe + k * ENCODING_DIM);
    for (int r = 0; r < 13; ++r) th[k * 13 + r] = (float)feats[k].opp_max[r];
    os[k] = (float)feats[k].opp_size;
  }
  torch::Tensor hands;
  {
    torch::NoGradGuard ng;
    hands = belief
                .run_method("sample_opp", tokens.to(device), hist_idx.to(device),
                            oppenc.to(device), thermo13.to(device),
                            oppsz.to(device), (int64_t)N)
                .toTensor().to(torch::kCPU).contiguous();  // [B, N, 13]
  }
  const int64_t *hp = hands.data_ptr<int64_t>();
  std::vector<std::array<int, 13>> worlds(B * N);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for (int k = 0; k < B; ++k)
    for (int i = 0; i < N; ++i) {
      std::array<int, 13> w{};
      if (gamma > 0.0f && u(rng) < gamma) {
        w = uniform_world(feats[k].opp_max, feats[k].opp_size, rng);
      } else {
        for (int r = 0; r < 13; ++r) w[r] = (int)hp[(k * N + i) * 13 + r];
      }
      worlds[k * N + i] = w;
    }
  return score_worlds(feats, worlds, pi, N);
}

}  // namespace az_pimc

#endif  // AZ_PIMC_DET_VALUE_H
