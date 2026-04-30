#ifndef PIMC_PLAYER_H
#define PIMC_PLAYER_H

#include "game.h"
#include "greedy/greedy_player.h"
#include "greedy/linear_evaluator.h"
#include "greedy/tree_evaluator.h"
#include "partial_game.h"
#include "player.h"
#include "tablebase_peek.h"
#include "util.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(BIG2_PIMC_STATS)
#define BIG2_PIMC_STATS 0
#endif

#if BIG2_PIMC_STATS
#include <atomic>
#endif

// ---------------------------------------------------------------------------
// Optional global counters (enable with -DBIG2_PIMC_STATS=1).
// total_dets_used: sum of determinizations across all PIMC decisions.
// total_dets_saved, total_rollout_equiv_saved: savings vs always using n_max.
// ---------------------------------------------------------------------------
#if BIG2_PIMC_STATS
struct PimcGlobalStats {
  std::atomic<uint64_t> total_select_calls{0};
  std::atomic<uint64_t> total_dets_used{0};
  std::atomic<uint64_t> total_dets_saved{0};
  std::atomic<uint64_t> total_rollout_equiv_saved{0};
  std::atomic<uint64_t> voluntary_pass_opportunities{0};
  std::atomic<uint64_t> voluntary_pass_chosen{0};
};

inline PimcGlobalStats &pimc_global_stats() {
  static PimcGlobalStats s;
  return s;
}
#endif

// ---------------------------------------------------------------------------
// Hand sampler
//
// Returns a plausible opponent hand of size opp_count drawn uniformly from
// the pool of cards not in my_hand and not in the discard pile.
// Cards not drawn into the opponent hand implicitly represent the unseen pile.
// ---------------------------------------------------------------------------
inline std::array<int, 13>
sample_opponent_hand(const std::array<int, 13> &my_hand,
                     const std::array<int, 13> &discard, int opp_count,
                     std::mt19937 &rng) {
  std::vector<int> pool;
  pool.reserve(16);
  for (int r = 0; r < 13; ++r) {
    int available = max_cards_in_deck_for_rank(r) - my_hand[r] - discard[r];
    for (int i = 0; i < available; ++i)
      pool.push_back(r);
  }

  std::array<int, 13> opp_hand{};
  int pool_size = static_cast<int>(pool.size());
  for (int i = 0; i < opp_count && i < pool_size; ++i) {
    std::uniform_int_distribution<int> dist(i, pool_size - 1);
    int j = dist(rng);
    std::swap(pool[i], pool[j]);
    opp_hand[pool[i]]++;
  }
  return opp_hand;
}

// ---------------------------------------------------------------------------
// Hoeffding confidence: Bonferroni delta' = delta / M; symmetric radius
// r = sqrt(log(2/delta') / (2n)) for [0,1] means.
// Batched: scalar r, vector p_i = wins[i]/n, lower_i = p_i - r, upper_i = p_i + r.
// Stop if some arm b has lower[b] > max_{j!=b} upper[j] (unique top count).
// ---------------------------------------------------------------------------
inline bool confident_leader_hoeffding(const std::vector<int> &wins, int n, int M,
                                       double delta) {
  if (n < 1 || M < 2)
    return false;

  int max_w = wins[0];
  for (int i = 1; i < M; ++i) {
    if (wins[i] > max_w)
      max_w = wins[i];
  }
  int leader_count = 0;
  int b = 0;
  for (int i = 0; i < M; ++i) {
    if (wins[i] == max_w) {
      ++leader_count;
      b = i;
    }
  }
  if (leader_count != 1)
    return false;

  const double delta_prime = delta / static_cast<double>(M);
  if (delta_prime <= 0.0 || delta_prime >= 1.0)
    return false;

  const double r =
      std::sqrt(std::log(2.0 / delta_prime) / (2.0 * static_cast<double>(n)));
  const double inv_n = 1.0 / static_cast<double>(n);

  double max_upper_other = -1.0;
  for (int j = 0; j < M; ++j) {
    if (j == b)
      continue;
    const double pj = static_cast<double>(wins[j]) * inv_n;
    const double upper = pj + r;
    if (upper > max_upper_other)
      max_upper_other = upper;
  }

  const double pb = static_cast<double>(wins[b]) * inv_n;
  const double lower_b = pb - r;
  return lower_b > max_upper_other;
}

// ---------------------------------------------------------------------------
// Rollout policy: mirrors Player::select_move order.
//
// If ``resample_opponent_each_turn`` is true, before each opponent move in the
// simulated game we draw a new opponent hand from the remaining unseen pool
// (same distribution as ``sample_opponent_hand``) and reconstruct ``Game``.
// Moves already played stay in the discard pile; the new hand is consistent
// with public information but may be inconsistent with greedy choices made
// under earlier hypothetical hands.
// ---------------------------------------------------------------------------
template <typename EvalFn>
inline int policy_rollout(Game g, int my_player, EvalFn eval_fn,
                          bool resample_opponent_each_turn, std::mt19937 &rng,
                          std::vector<int> &legal_scratch) {
  while (!g.is_over()) {
    int cp = g.current_player();
    if (resample_opponent_each_turn && cp != my_player) {
      const std::array<int, 13> my_h = g.player_hand(my_player);
      const std::array<int, 13> disc = g.discard_pile();
      const int opp = 1 - my_player;
      const int opp_count = g.get_player_hand_size(opp);
      const std::array<int, 13> new_opp =
          sample_opponent_hand(my_h, disc, opp_count, rng);
      std::array<int, 13> h0 = g.player_hand(0);
      std::array<int, 13> h1 = g.player_hand(1);
      if (opp == 0)
        h0 = new_opp;
      else
        h1 = new_opp;
      g = Game(h0, h1, disc, g.last_move(), cp);
    }
    PartialGame pg(g, cp);
    TablebasePeekResult tb = peek_tablebase_move(pg);
    if (tb.move) {
      g.apply_move(*tb.move);
    } else {
      pg.get_legal_moves_into(legal_scratch);
      Move best = greedy_best(pg, legal_scratch, eval_fn);
      g.apply_move(best);
    }
  }
  return g.get_winner() == my_player ? 1 : 0;
}

// Rollout policy: like policy_rollout, but when pass and non-pass are both legal,
// pass if Ridge model predicts Δ>0 (same semantics as GreedyPassPlayer).
template <typename EvalFn>
inline int policy_rollout_pass_aware(Game g, int my_player, EvalFn eval_fn,
                                     LinearEvaluator &pass_model,
                                     bool resample_opponent_each_turn,
                                     std::mt19937 &rng,
                                     std::vector<int> &legal_scratch) {
  while (!g.is_over()) {
    int cp = g.current_player();
    if (resample_opponent_each_turn && cp != my_player) {
      const std::array<int, 13> my_h = g.player_hand(my_player);
      const std::array<int, 13> disc = g.discard_pile();
      const int opp = 1 - my_player;
      const int opp_count = g.get_player_hand_size(opp);
      const std::array<int, 13> new_opp =
          sample_opponent_hand(my_h, disc, opp_count, rng);
      std::array<int, 13> h0 = g.player_hand(0);
      std::array<int, 13> h1 = g.player_hand(1);
      if (opp == 0)
        h0 = new_opp;
      else
        h1 = new_opp;
      g = Game(h0, h1, disc, g.last_move(), cp);
    }
    PartialGame pg(g, cp);
    TablebasePeekResult tb = peek_tablebase_move(pg);
    if (tb.move) {
      g.apply_move(*tb.move);
    } else {
      pg.get_legal_moves_into(legal_scratch);
      if (voluntary_pass_legal(legal_scratch) && pass_model.loaded() &&
          pass_model.predict(pg) > 0.0)
        g.apply_move(Move(kPASS));
      else
        g.apply_move(greedy_best(pg, legal_scratch, eval_fn));
    }
  }
  return g.get_winner() == my_player ? 1 : 0;
}

// Shared PIMC move selection.
//
// Common random numbers (CRN): one opponent hand per det, all candidates scored.
//
// If adaptive: loop up to n_max; after n_min dets, stop early when
// confident_leader_hoeffding fires. If not adaptive: exactly n_max dets.
//
// rollout_fn: (Game, my_player, eval_fn, resample_opponent_each_turn, rng,
//              legal_scratch) -> win {0,1}
template <typename EvalFn, typename RolloutFn>
inline Move pimc_select_move_impl(const PartialGame &game_, int player_num_,
                                  int n_max, int n_min, bool adaptive,
                                  double delta, std::mt19937 &rng_,
                                  EvalFn eval_fn, bool resample_opponent_each_turn,
                                  RolloutFn rollout_fn,
                                  std::vector<int> &legal_work,
                                  std::vector<int> &candidates_work) {
  game_.get_legal_moves_into(legal_work);

  candidates_work.clear();
  candidates_work.reserve(legal_work.size());
  bool has_pass = false;
  for (int mid : legal_work) {
    if (Move(mid).combination == Move::Combination::kPass)
      has_pass = true;
    else
      candidates_work.push_back(mid);
  }

  if (candidates_work.empty())
    return Move(kPASS);

  [[maybe_unused]] const bool voluntary_ctx =
      has_pass && !candidates_work.empty();
#if BIG2_PIMC_STATS
  if (voluntary_ctx)
    pimc_global_stats().voluntary_pass_opportunities += 1;
#endif

  if (has_pass)
    candidates_work.push_back(kPASS);

  const std::array<int, 13> my_hand = game_.player_hand();
  const std::array<int, 13> discard = game_.discard_pile();
  const int opp_count = game_.opponent_hand_size();
  const Move last_mv = game_.last_move();

  const int M = static_cast<int>(candidates_work.size());
  std::vector<int> wins(M, 0);

#if BIG2_PIMC_STATS
  pimc_global_stats().total_select_calls += 1;
#endif

  int dets_used = 0;
  for (int det = 0; det < n_max; ++det) {
    const std::array<int, 13> opp_hand =
        sample_opponent_hand(my_hand, discard, opp_count, rng_);

    const std::array<int, 13> hand0 =
        (player_num_ == 0) ? my_hand : opp_hand;
    const std::array<int, 13> hand1 =
        (player_num_ == 0) ? opp_hand : my_hand;

    for (int ci = 0; ci < M; ++ci) {
      Game g(hand0, hand1, discard, last_mv, player_num_);
      g.apply_move(candidates_work[ci]);
      wins[ci] += rollout_fn(g, player_num_, eval_fn,
                             resample_opponent_each_turn, rng_, legal_work);
    }

    dets_used = det + 1;

    if (adaptive && dets_used >= n_min) {
      if (M == 1)
        break;
      if (confident_leader_hoeffding(wins, dets_used, M, delta))
        break;
    }
  }

#if BIG2_PIMC_STATS
  pimc_global_stats().total_dets_used += static_cast<uint64_t>(dets_used);
  if (adaptive) {
    const uint64_t saved =
        static_cast<uint64_t>(n_max - dets_used);
    pimc_global_stats().total_dets_saved += saved;
    pimc_global_stats().total_rollout_equiv_saved +=
        saved * static_cast<uint64_t>(M);
  }
#endif

  int best_ci = 0;
  for (int ci = 1; ci < M; ++ci) {
    if (wins[ci] > wins[best_ci])
      best_ci = ci;
    else if (wins[ci] == wins[best_ci] &&
             candidates_work[ci] < candidates_work[best_ci])
      best_ci = ci;
  }

#if BIG2_PIMC_STATS
  if (voluntary_ctx &&
      Move(candidates_work[best_ci]).combination == Move::Combination::kPass)
    pimc_global_stats().voluntary_pass_chosen += 1;
#endif

  return Move(candidates_work[best_ci]);
}

// ---------------------------------------------------------------------------
// PimcPlayer — greedy heuristic rollout (after tablebase).
// ---------------------------------------------------------------------------
class PimcPlayer : public Player {
public:
  // Fixed N determinizations (legacy).
  explicit PimcPlayer(int num_samples, unsigned int seed,
                    bool resample_opponent_each_turn = false)
      : n_max_(num_samples), n_min_(0), delta_(0.05), adaptive_(false),
        resample_opponent_each_turn_(resample_opponent_each_turn), rng_(seed) {}

  // Adaptive: up to n_max dets; stop early after n_min when Hoeffding test passes.
  PimcPlayer(int n_max, int n_min, double delta, unsigned int seed,
             bool resample_opponent_each_turn = false)
      : n_max_(n_max), n_min_(n_min), delta_(delta), adaptive_(true),
        resample_opponent_each_turn_(resample_opponent_each_turn), rng_(seed) {}

protected:
  void on_deal(const std::array<int, 13> & /*hand*/, int player_num) override {
    player_num_ = player_num;
  }

  Move select_move_impl() override {
    return pimc_select_move_impl(
        game_, player_num_, n_max_, n_min_, adaptive_, delta_, rng_, greedy_hand_eval,
        resample_opponent_each_turn_,
        [](Game g, int my_player, const auto &eval_fn, bool redet, std::mt19937 &rng,
           std::vector<int> &legal_scratch) {
          return policy_rollout(g, my_player, eval_fn, redet, rng, legal_scratch);
        },
        legal_work_, candidates_work_);
  }

private:
  int n_max_;
  int n_min_;
  double delta_;
  bool adaptive_;
  bool resample_opponent_each_turn_;
  int player_num_{0};
  std::mt19937 rng_;
  std::vector<int> legal_work_;
  std::vector<int> candidates_work_;
};

// ---------------------------------------------------------------------------
// PimcTreePlayer — tree evaluator rollout (after tablebase).
// ---------------------------------------------------------------------------
class PimcTreePlayer : public Player {
public:
  PimcTreePlayer(int num_samples, const std::string &model_path,
                 unsigned int seed, bool resample_opponent_each_turn = false)
      : n_max_(num_samples), n_min_(0), delta_(0.05), adaptive_(false),
        resample_opponent_each_turn_(resample_opponent_each_turn),
        evaluator_(model_path), rng_(seed) {
    if (!evaluator_.loaded())
      throw std::runtime_error("PimcTreePlayer: model failed to load from '" +
                               model_path + "'");
  }

  PimcTreePlayer(int n_max, int n_min, double delta,
                 const std::string &model_path, unsigned int seed,
                 bool resample_opponent_each_turn = false)
      : n_max_(n_max), n_min_(n_min), delta_(delta), adaptive_(true),
        resample_opponent_each_turn_(resample_opponent_each_turn),
        evaluator_(model_path), rng_(seed) {
    if (!evaluator_.loaded())
      throw std::runtime_error("PimcTreePlayer: model failed to load from '" +
                               model_path + "'");
  }

protected:
  void on_deal(const std::array<int, 13> & /*hand*/, int player_num) override {
    player_num_ = player_num;
  }

  Move select_move_impl() override {
    return pimc_select_move_impl(
        game_, player_num_, n_max_, n_min_, adaptive_, delta_, rng_,
        [this](const PartialGame &sim) { return evaluator_.predict(sim); },
        resample_opponent_each_turn_,
        [](Game g, int my_player, const auto &eval_fn, bool redet, std::mt19937 &rng,
           std::vector<int> &legal_scratch) {
          return policy_rollout(g, my_player, eval_fn, redet, rng, legal_scratch);
        },
        legal_work_, candidates_work_);
  }

private:
  int n_max_;
  int n_min_;
  double delta_;
  bool adaptive_;
  bool resample_opponent_each_turn_;
  int player_num_{0};
  TreeEvaluator evaluator_;
  std::mt19937 rng_;
  std::vector<int> legal_work_;
  std::vector<int> candidates_work_;
};

// ---------------------------------------------------------------------------
// PimcLinearPlayer — Ridge linear rollout (after tablebase).
// ---------------------------------------------------------------------------
class PimcLinearPlayer : public Player {
public:
  PimcLinearPlayer(int num_samples, const std::string &model_path,
                   unsigned int seed, bool resample_opponent_each_turn = false)
      : n_max_(num_samples), n_min_(0), delta_(0.05), adaptive_(false),
        resample_opponent_each_turn_(resample_opponent_each_turn),
        evaluator_(model_path), rng_(seed) {
    if (!evaluator_.loaded())
      throw std::runtime_error("PimcLinearPlayer: model failed to load from '" +
                               model_path + "'");
  }

  PimcLinearPlayer(int n_max, int n_min, double delta,
                   const std::string &model_path, unsigned int seed,
                   bool resample_opponent_each_turn = false)
      : n_max_(n_max), n_min_(n_min), delta_(delta), adaptive_(true),
        resample_opponent_each_turn_(resample_opponent_each_turn),
        evaluator_(model_path), rng_(seed) {
    if (!evaluator_.loaded())
      throw std::runtime_error("PimcLinearPlayer: model failed to load from '" +
                               model_path + "'");
  }

protected:
  void on_deal(const std::array<int, 13> & /*hand*/, int player_num) override {
    player_num_ = player_num;
  }

  Move select_move_impl() override {
    return pimc_select_move_impl(
        game_, player_num_, n_max_, n_min_, adaptive_, delta_, rng_,
        [this](const PartialGame &sim) { return evaluator_.predict(sim); },
        resample_opponent_each_turn_,
        [](Game g, int my_player, const auto &eval_fn, bool redet, std::mt19937 &rng,
           std::vector<int> &legal_scratch) {
          return policy_rollout(g, my_player, eval_fn, redet, rng, legal_scratch);
        },
        legal_work_, candidates_work_);
  }

private:
  int n_max_;
  int n_min_;
  double delta_;
  bool adaptive_;
  bool resample_opponent_each_turn_;
  int player_num_{0};
  LinearEvaluator evaluator_;
  std::mt19937 rng_;
  std::vector<int> legal_work_;
  std::vector<int> candidates_work_;
};

// ---------------------------------------------------------------------------
// PimcPassRolloutPlayer — PIMC with pass-aware rollout (after tablebase).
// Loads Ridge pass weights (same format as LinearEvaluator, default data/pass_ridge_w.txt).
// ---------------------------------------------------------------------------
class PimcPassRolloutPlayer : public Player {
public:
  explicit PimcPassRolloutPlayer(int num_samples, const std::string &model_path,
                                 unsigned int seed,
                                 bool resample_opponent_each_turn = false)
      : n_max_(num_samples), n_min_(0), delta_(0.05), adaptive_(false),
        resample_opponent_each_turn_(resample_opponent_each_turn),
        pass_model_(model_path), rng_(seed) {
    if (!pass_model_.loaded())
      throw std::runtime_error("PimcPassRolloutPlayer: model failed to load from '" +
                               model_path + "'");
  }

  PimcPassRolloutPlayer(int n_max, int n_min, double delta,
                        const std::string &model_path, unsigned int seed,
                        bool resample_opponent_each_turn = false)
      : n_max_(n_max), n_min_(n_min), delta_(delta), adaptive_(true),
        resample_opponent_each_turn_(resample_opponent_each_turn),
        pass_model_(model_path), rng_(seed) {
    if (!pass_model_.loaded())
      throw std::runtime_error("PimcPassRolloutPlayer: model failed to load from '" +
                               model_path + "'");
  }

protected:
  void on_deal(const std::array<int, 13> & /*hand*/, int player_num) override {
    player_num_ = player_num;
  }

  Move select_move_impl() override {
    return pimc_select_move_impl(
        game_, player_num_, n_max_, n_min_, adaptive_, delta_, rng_, greedy_hand_eval,
        resample_opponent_each_turn_,
        [this](Game g, int my_player, const auto &eval_fn, bool redet,
               std::mt19937 &rng, std::vector<int> &legal_scratch) {
          return policy_rollout_pass_aware(g, my_player, eval_fn, pass_model_, redet,
                                           rng, legal_scratch);
        },
        legal_work_, candidates_work_);
  }

private:
  int n_max_;
  int n_min_;
  double delta_;
  bool adaptive_;
  bool resample_opponent_each_turn_;
  int player_num_{0};
  LinearEvaluator pass_model_;
  std::mt19937 rng_;
  std::vector<int> legal_work_;
  std::vector<int> candidates_work_;
};

#endif
