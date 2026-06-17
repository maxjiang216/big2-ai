#include "az_pimc/pimc_player.h"

#include "az_search/features.h"  // opp_max_counts, trick_counts
#include "move.h"
#include "nn_encode.h"  // encode_exact / encode_thermo / ENCODING_DIM

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace az_pimc {

static const bool kDebug = std::getenv("PIMC_DEBUG") != nullptr;
// PIMC_AGG=aom -> Average-of-Max (per-world argmax plurality vote = the FUSED,
// strategy-fusion aggregator). Default = Max-of-Average (commit one move, then
// average over worlds = no root fusion).
static const bool kFused = [] {
  const char *e = std::getenv("PIMC_AGG");
  return e && std::string(e) == "aom";
}();

void AzPimcPlayer::on_opponent_move(const Move &move) {
  history_.push_back(encodeMove(move));
}
void AzPimcPlayer::on_self_move(const Move &move) {
  history_.push_back(encodeMove(move));
}

std::array<int, 13> AzPimcPlayer::uniform_hand(const std::array<int, 13> &thermo,
                                               int opp_size) {
  // The unseen pool is exactly the thermo (deck - our_hand - discard). Draw
  // opp_size physical cards uniformly from that multiset.
  std::vector<int> pool;
  for (int r = 0; r < 13; ++r)
    for (int c = 0; c < thermo[r]; ++c) pool.push_back(r);
  std::shuffle(pool.begin(), pool.end(), rng_);
  std::array<int, 13> h{};
  const int take = std::min<int>(opp_size, static_cast<int>(pool.size()));
  for (int i = 0; i < take; ++i) ++h[pool[i]];
  return h;
}

std::vector<std::array<int, 13>> AzPimcPlayer::sample_hands(
    const std::array<int, 13> &thermo, int opp_size) {
  auto i64 = torch::TensorOptions().dtype(torch::kInt64);
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
  const int T = static_cast<int>(history_.size());
  torch::Tensor tokens = torch::zeros({1, T > 0 ? T : 1}, i64);
  int64_t *tk = tokens.data_ptr<int64_t>();
  for (int i = 0; i < T; ++i) tk[i] = history_[i];
  torch::Tensor hist_idx = torch::full({1}, static_cast<int64_t>(T), i64);

  torch::Tensor oppt = torch::zeros({1, ENCODING_DIM}, f32);
  encode_thermo(thermo, oppt.data_ptr<float>());
  torch::Tensor thermo13 = torch::zeros({1, 13}, f32);
  float *th = thermo13.data_ptr<float>();
  for (int r = 0; r < 13; ++r) th[r] = static_cast<float>(thermo[r]);
  torch::Tensor osz = torch::full({1}, static_cast<float>(opp_size), f32);

  std::vector<std::array<int, 13>> hands(n_);
  {
    torch::NoGradGuard ng;
    auto out = belief_
                   ->run_method("sample_opp", tokens.to(device_),
                                hist_idx.to(device_), oppt.to(device_),
                                thermo13.to(device_), osz.to(device_), (int64_t)n_)
                   .toTensor()
                   .to(torch::kCPU)
                   .contiguous();  // [1, N, 13] long
    const int64_t *hp = out.data_ptr<int64_t>();
    for (int i = 0; i < n_; ++i)
      for (int r = 0; r < 13; ++r)
        hands[i][r] = static_cast<int>(hp[i * 13 + r]);
  }

  // Uniform γ-floor: replace each sampled hand with a uniform draw w.p. γ.
  if (gamma_ > 0.0f) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int i = 0; i < n_; ++i)
      if (u(rng_) < gamma_) hands[i] = uniform_hand(thermo, opp_size);
  }
  return hands;
}

Move AzPimcPlayer::select_move_impl() {
  const auto legal = game_.get_legal_moves();
  if (legal.size() == 1) return Move(legal[0]);

  const auto hand = game_.player_hand();
  const auto discard = game_.discard_pile();
  const auto thermo = az_search::opp_max_counts(hand, discard);
  const int opp_size = game_.opponent_hand_size();

  // Per-move resulting our-hand + the trick the opponent then faces. An emptying
  // move wins immediately — no determinization needed.
  std::vector<az_pi::PiEvalFeatures> feats;
  feats.reserve(legal.size() * n_);
  std::vector<int> move_of_block;  // legal index per N-block, in feats order
  move_of_block.reserve(legal.size());

  auto hands = sample_hands(thermo, opp_size);

  for (int m : legal) {
    const auto tc = az_search::trick_counts(m);
    std::array<int, 13> our_after = hand;
    int our_after_size = 0;
    for (int r = 0; r < 13; ++r) {
      our_after[r] -= tc[r];
      our_after_size += our_after[r];
    }
    if (our_after_size == 0) return Move(m);  // this move wins now

    move_of_block.push_back(m);
    for (const auto &oh : hands) {
      az_pi::PiEvalFeatures f;
      f.hand = oh;            // mover = opponent (full sampled hand)
      f.opp_hand = our_after; // opponent-of-mover = us, after our move
      f.trick = tc;           // the move we played is the trick to beat
      f.our_size = opp_size;
      f.opp_size = our_after_size;
      f.my_pts = 0.0f;        // no series state in a PartialGame
      f.opp_pts = 0.0f;
      feats.push_back(f);
    }
  }

  const auto evals = pi_->eval_batch(feats);

  const int nb = static_cast<int>(move_of_block.size());
  if (kFused) {
    // Average-of-Max: each world votes for its own best move (the optimistic /
    // strategy-fusion rule). Pick the plurality winner.
    std::vector<int> votes(nb, 0);
    for (int i = 0; i < n_; ++i) {
      int wbest = 0;
      double wbest_v = 1e30;  // minimise P(opp wins) in this world
      for (int b = 0; b < nb; ++b) {
        const double v = evals[b * n_ + i].value;
        if (v < wbest_v) { wbest_v = v; wbest = b; }
      }
      ++votes[wbest];
    }
    int best = move_of_block[0], best_votes = -1;
    for (int b = 0; b < nb; ++b)
      if (votes[b] > best_votes) { best_votes = votes[b]; best = move_of_block[b]; }
    return Move(best);
  }

  // Max-of-Average: commit one move across all worlds, then average.
  int best = legal[0];
  float best_score = -1e30f;
  for (int b = 0; b < nb; ++b) {
    double sum = 0.0;
    for (int i = 0; i < n_; ++i) sum += evals[b * n_ + i].value;  // P(opp wins)
    const float score = 1.0f - static_cast<float>(sum / n_);      // P(we win)
    if (kDebug)
      std::fprintf(stderr, "  move %3d  score(P_we_win)=%.3f\n",
                   move_of_block[b], score);
    if (score > best_score) {
      best_score = score;
      best = move_of_block[b];
    }
  }
  return Move(best);
}

}  // namespace az_pimc
