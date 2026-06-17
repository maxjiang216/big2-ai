#include "az_ii/az_ii_player.h"

#include "az_search/considered_moves.h"  // player_composed_logit, AZ_PLAYER_HEAD_DIM
#include "az_search/features.h"          // opp_max_counts
#include "move.h"
#include "nn_encode.h"  // encode_exact / encode_thermo / ENCODING_DIM
#include "util.h"

namespace az_ii {

void AzIiPlayer::on_opponent_move(const Move &move) {
  history_.push_back(encodeMove(move));
}

void AzIiPlayer::on_self_move(const Move &move) {
  history_.push_back(encodeMove(move));
}

Move AzIiPlayer::select_move_impl() {
  const auto legal = game_.get_legal_moves();
  if (legal.size() == 1) return Move(legal[0]);

  const auto hand = game_.player_hand();
  const auto discard = game_.discard_pile();
  const auto oppmax = az_search::opp_max_counts(hand, discard);
  const float our_size = static_cast<float>(game_.num_cards());
  const float opp_size = static_cast<float>(game_.opponent_hand_size());

  auto i64 = torch::TensorOptions().dtype(torch::kInt64);
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
  const int T = static_cast<int>(history_.size());
  // T==0 (game lead, no history): a length-1 dummy token read at hist_idx 0 maps
  // to BOS (causal), so the dummy never contaminates the readout.
  torch::Tensor tokens = torch::zeros({1, T > 0 ? T : 1}, i64);
  int64_t *tk = tokens.data_ptr<int64_t>();
  for (int i = 0; i < T; ++i) tk[i] = history_[i];
  torch::Tensor hist_idx = torch::full({1}, static_cast<int64_t>(T), i64);

  torch::Tensor handt = torch::zeros({1, ENCODING_DIM}, f32);
  torch::Tensor oppt = torch::zeros({1, ENCODING_DIM}, f32);
  encode_exact(hand, handt.data_ptr<float>());
  encode_thermo(oppmax, oppt.data_ptr<float>());
  torch::Tensor osz = torch::full({1}, opp_size / 16.0f, f32);
  torch::Tensor usz = torch::full({1}, our_size / 16.0f, f32);
  torch::Tensor otm = torch::ones({1}, f32);
  // Series state unknown to a PartialGame; eval at the fresh-series (0,0) point.
  torch::Tensor mpts = torch::zeros({1}, f32);
  torch::Tensor opts = torch::zeros({1}, f32);

  torch::NoGradGuard ng;
  auto out = model_
                 ->forward({tokens.to(device_), hist_idx.to(device_),
                            handt.to(device_), oppt.to(device_), osz.to(device_),
                            usz.to(device_), otm.to(device_), mpts.to(device_),
                            opts.to(device_)})
                 .toTuple();
  auto policy = out->elements()[1].toTensor().to(torch::kCPU).contiguous();
  const float *lp = policy.data_ptr<float>();

  int best = legal[0];
  float best_v = -1e30f;
  for (int m : legal) {
    const float s = az_search::player_composed_logit(m, lp);
    if (s > best_v) {
      best_v = s;
      best = m;
    }
  }
  return Move(best);
}

}  // namespace az_ii
