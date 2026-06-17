// az_regret: policy-improvement regret probe for az_search.
//
// THE question behind the strength ceiling: when players are equal, the game
// OUTCOME is nearly unpredictable from a position (card luck) — but does the
// MOVE CHOICE matter? Value-flatness != policy-flatness. This probe measures
// the latter directly, using DEEP search as the reference oracle and the net's
// 0-sim PRIOR as the candidate policy.
//
// At every non-definitive decision (definitive = hand-emptying / opp-1 / forced
// win — oracle-handled, zero regret by construction) we run a deep search and
// record, from the root player's POV (Q = P(root wins)):
//   * q_spread  = maxQ - minQ over VISITED legal moves   (does choice matter?)
//   * agreement = (argmax prior == max-visit deep move)   (is 0-sim policy right?)
//   * regret    = maxQ - Q[prior move]                    (value left on table)
// Buckets: deep move type, lead/follow, hand-size phase, and the named decision
// classes (auxiliary selection, lead single-ordering, bomb timing).
//
//   make az_regret
//   ./bin/az_regret --model models/az_seq.pt --games 200 --sims 800 --device cuda
//
// Single net drives BOTH seats (self-play distribution). Batched across games:
// each slot pumps until it needs a leaf eval; evals are flushed per round.

#include "az_search/az_search.h"
#include "az_search/nn_eval.h"
#include "eval_helpers.h"
#include "game.h"
#include "move.h"
#include "util.h"

#include <torch/cuda.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace az_search;

static const char *arg(int argc, char **argv, const char *key, const char *def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return def;
}
static bool has_flag(int argc, char **argv, const char *key) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return true;
  return false;
}

static int hand_count(const std::array<int, 13> &h) {
  int n = 0;
  for (int r = 0; r < 13; ++r) n += h[r];
  return n;
}

static SearchState mover_state(const Game &game, int mover) {
  SearchState s;
  s.our_hand = game.player_hand(mover);
  s.opp_size = game.get_player_hand_size(1 - mover);
  s.discard = game.discard_pile();
  s.last_move = encodeMove(game.last_move());
  s.side = kUs;
  return s;
}

static const char *comb_name(Move::Combination c) {
  using C = Move::Combination;
  switch (c) {
    case C::kPass: return "pass";
    case C::kSingle: return "single";
    case C::kDouble: return "double";
    case C::kTriple: return "triple";
    case C::kFullHouse: return "fullhouse";
    case C::kBomb: return "bomb";
    default: return "straight";  // all straight/double-straight/triple-straight
  }
}

// One running bucket of decision statistics.
struct Bucket {
  long n = 0;
  long agree = 0;
  double sum_regret = 0.0;
  double sum_spread = 0.0;
  long n_regret = 0;  // decisions where prior move's Q was known (child visited)
  void add(bool ag, double spread, bool have_regret, double regret) {
    ++n;
    if (ag) ++agree;
    sum_spread += spread;
    if (have_regret) { sum_regret += regret; ++n_regret; }
  }
};

struct RSlot {
  int gidx = -1;
  Game game;
  std::mt19937 rng;
  std::unique_ptr<Search> tree;
  std::vector<int> history;
  bool prefix_dirty = true;
  int mover = 0;
  int sims_done = 0;
  bool active = false;
  bool searching = false;
  bool waiting = false;
  LeafRequest pending;
  bool result_ready = false;
  NetEval neval;
};

struct SlotEval : Evaluator {
  NNEvaluator &nn;
  int slot;
  SlotEval(NNEvaluator &n, int s) : nn(n), slot(s) {}
  NetEval eval(const EvalFeatures &f) override { return nn.eval_batch({slot}, {f})[0]; }
};

int main(int argc, char **argv) {
  const std::string model = arg(argc, argv, "--model", "models/az_seq.pt");
  const int games = std::atoi(arg(argc, argv, "--games", "200"));
  const int sims = std::atoi(arg(argc, argv, "--sims", "800"));
  const int slots_n = std::atoi(arg(argc, argv, "--slots", "256"));
  const unsigned base_seed = (unsigned)std::strtoul(arg(argc, argv, "--seed", "42"), nullptr, 10);
  std::string device_str = arg(argc, argv, "--device", "cuda");
  const bool use_kv_cache = !has_flag(argc, argv, "--no-kv-cache");

  torch::Device device = torch::kCPU;
  if (device_str == "cuda" || (device_str == "auto" && torch::cuda::is_available()))
    device = torch::kCUDA;

  const int pool = std::min(slots_n, games);
  NNEvaluator nn(model, device, pool, use_kv_cache);

  std::printf("az_regret: model=%s  games=%d  sims=%d  slots=%d  %s\n",
              model.c_str(), games, sims, pool,
              device == torch::kCUDA ? "cuda" : "cpu");

  // Stats.
  std::map<std::string, Bucket> by_type;     // deep move combination
  std::map<std::string, Bucket> by_phase;    // hand-size bin (mover's hand)
  Bucket lead, follow;                        // lead vs follow
  Bucket aux_sel;                             // deep move carries an auxiliary
  Bucket single_order;                        // lead with >=2 singles legal
  Bucket bomb_timing;                         // bomb legal (play-or-not decision)
  Bucket all;
  // spread histogram (does choice matter at all?)
  long sp_lt02 = 0, sp_lt05 = 0, sp_lt10 = 0, sp_ge10 = 0;
  double sum_q_deep_minus_prior = 0.0;  // signed: deep's own Q vs prior's Q

  std::vector<RSlot> slots(pool);
  int next_game = 0;

  auto start_game = [&](RSlot &s) -> bool {
    if (next_game >= games) return false;
    s.gidx = next_game++;
    s.rng.seed(base_seed + (unsigned)s.gidx);
    s.game = Game();
    s.game.shuffle_deal(s.rng);
    s.tree.reset();
    s.history.clear();
    s.prefix_dirty = true;
    s.sims_done = 0;
    s.active = true; s.searching = false; s.waiting = false;
    return true;
  };

  auto commit = [&](RSlot &s, int m) {
    s.game.apply_move(m);
    s.history.push_back(m);
    s.prefix_dirty = true;
    s.tree.reset();  // fresh search per decision (clean prior/Q measurement)
  };

  // Record stats for a finished search at slot s, mover = current player.
  auto record = [&](RSlot &s) {
    const Node *root = s.tree->root_node();
    if (!root || root->terminal) return;
    // Per-move Q from visited children (player node: child->value = P root wins).
    double maxQ = -1.0, minQ = 2.0;
    std::map<int, double> qof;
    for (const Edge &e : root->edges) {
      if (e.child && e.child->N > 0) {
        double q = e.child->value;
        qof[e.move_id] = q;
        if (q > maxQ) maxQ = q;
        if (q < minQ) minQ = q;
      }
    }
    if (qof.empty()) return;  // nothing expanded (shouldn't happen at sims>1)
    double spread = maxQ - minQ;

    // prior argmax + deep (max-visit) move
    int prior_move = -1; long best_pc = -1;
    for (auto &pr : s.tree->root_prior())
      if (pr.second > best_pc) { best_pc = pr.second; prior_move = pr.first; }
    int deep_move = s.tree->best_move();
    bool agree = (prior_move == deep_move);

    bool have_regret = qof.count(prior_move) > 0;
    double regret = have_regret ? (maxQ - qof[prior_move]) : 0.0;
    if (have_regret && qof.count(deep_move))
      sum_q_deep_minus_prior += qof[deep_move] - qof[prior_move];

    // classify
    Move dm(deep_move);
    const bool is_lead = (s.tree->root_node()->st.last_move ==
                          encodeMove(Move(Move::Combination::kPass)));
    const int hs = hand_count(s.tree->root_node()->st.our_hand);

    all.add(agree, spread, have_regret, regret);
    by_type[comb_name(dm.combination)].add(agree, spread, have_regret, regret);
    (is_lead ? lead : follow).add(agree, spread, have_regret, regret);

    std::string ph = hs <= 4 ? "hs<=4" : hs <= 8 ? "hs5-8" : "hs9+";
    by_phase[ph].add(agree, spread, have_regret, regret);

    // auxiliary selection: deep move is a bomb/fullhouse carrying an aux card
    if (dm.auxiliary > 0 &&
        (dm.combination == Move::Combination::kBomb ||
         dm.combination == Move::Combination::kFullHouse))
      aux_sel.add(agree, spread, have_regret, regret);

    // lead single-ordering: leading, >=2 distinct single moves legal
    int n_singles = 0; bool bomb_legal = false;
    for (const Edge &e : root->edges) {
      Move m(e.move_id);
      if (m.combination == Move::Combination::kSingle) ++n_singles;
      if (m.combination == Move::Combination::kBomb) bomb_legal = true;
    }
    if (is_lead && n_singles >= 2)
      single_order.add(agree, spread, have_regret, regret);
    if (bomb_legal)  // bomb available: keep-or-play timing decision
      bomb_timing.add(agree, spread, have_regret, regret);

    if (spread < 0.02) ++sp_lt02;
    else if (spread < 0.05) ++sp_lt05;
    else if (spread < 0.10) ++sp_lt10;
    else ++sp_ge10;
  };

  auto advance = [&](RSlot &s, int idx, std::vector<EvalFeatures> &F,
                     std::vector<int> &S) -> void {
    for (;;) {
      if (!s.active) { if (!start_game(s)) return; continue; }
      if (s.waiting) {
        if (!s.result_ready) return;
        s.tree->apply_eval(s.neval);
        s.result_ready = false; s.waiting = false; ++s.sims_done;
        continue;
      }
      if (s.game.is_over()) { s.active = false; continue; }
      auto legal = s.game.get_legal_moves();
      if (legal.size() == 1) { commit(s, legal[0]); continue; }
      const int mover = s.game.current_player();
      if (!s.searching) {
        SearchState st = mover_state(s.game, mover);
        if (auto d = az_definitive_move(st, /*allow_forced_win=*/true)) {
          commit(s, *d); continue;  // oracle-handled: skip (regret 0)
        }
        unsigned tseed = base_seed ^ (0x9E3779B9u * (unsigned)s.gidx);
        s.tree = std::make_unique<Search>(st, s.history,
                                          SearchConfig{1.5f, sims, tseed, false});
        s.mover = mover; s.sims_done = 0; s.searching = true;
        if (s.prefix_dirty) {
          nn.set_prefixes({idx}, {&s.history});
          s.prefix_dirty = false;
        }
      }
      bool pushed = false;
      while (s.sims_done < sims) {
        LeafRequest req = s.tree->select_leaf();
        if (!req.needs_eval) { ++s.sims_done; continue; }
        s.pending = std::move(req); s.waiting = true;
        F.push_back(std::move(s.pending.feat)); S.push_back(idx);
        pushed = true; break;
      }
      if (pushed) return;
      SlotEval ev(nn, idx);
      s.tree->finalize(&ev);
      record(s);
      commit(s, s.tree->best_move());
      s.searching = false;
    }
  };

  auto t0 = std::chrono::steady_clock::now();
  for (auto &s : slots) start_game(s);
  for (;;) {
    std::vector<EvalFeatures> F; std::vector<int> S;
    bool any_active = false;
    for (int i = 0; i < pool; ++i) {
      if (slots[i].active || next_game < games) any_active = true;
      advance(slots[i], i, F, S);
    }
    if (F.empty()) {
      if (!any_active) break;
      bool still = false;
      for (int i = 0; i < pool; ++i) if (slots[i].active) still = true;
      if (!still && next_game >= games) break;
      continue;
    }
    auto r = nn.eval_batch(S, F);
    for (size_t i = 0; i < S.size(); ++i) {
      slots[S[i]].neval = r[i];
      slots[S[i]].result_ready = true;
    }
  }
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();

  auto row = [](const char *label, const Bucket &b) {
    if (b.n == 0) { std::printf("  %-16s    (none)\n", label); return; }
    double ar = 100.0 * (double)b.agree / (double)b.n;
    double sp = b.sum_spread / (double)b.n;
    double rg = b.n_regret ? b.sum_regret / (double)b.n_regret : 0.0;
    std::printf("  %-16s  n=%-7ld  agree=%5.1f%%  spread=%.4f  regret=%.4f\n",
                label, b.n, ar, sp, rg);
  };

  std::printf("\n=== az_regret  [%s] ===\n", format_elapsed(ms).c_str());
  std::printf("metrics: spread=maxQ-minQ over legal moves (does choice matter?), "
              "agree=prior picks deep move, regret=maxQ-Q[prior]\n\n");
  std::printf("ALL DECISIONS:\n");
  row("all", all);
  std::printf("\nby deep-move type:\n");
  for (auto &kv : by_type) row(kv.first.c_str(), kv.second);
  std::printf("\nby lead/follow:\n");
  row("lead", lead); row("follow", follow);
  std::printf("\nby phase (mover hand):\n");
  for (auto &kv : by_phase) row(kv.first.c_str(), kv.second);
  std::printf("\nnamed decision classes:\n");
  row("aux-select", aux_sel);
  row("lead-single-ord", single_order);
  row("bomb-available", bomb_timing);
  std::printf("\nq_spread histogram (fraction of decisions):\n");
  long tot = all.n ? all.n : 1;
  std::printf("  spread<0.02: %5.1f%%   <0.05: %5.1f%%   <0.10: %5.1f%%   >=0.10: %5.1f%%\n",
              100.0 * sp_lt02 / tot, 100.0 * sp_lt05 / tot,
              100.0 * sp_lt10 / tot, 100.0 * sp_ge10 / tot);
  std::printf("mean Q(deep) - Q(prior) over known: %+.4f\n",
              all.n_regret ? sum_q_deep_minus_prior / all.n_regret : 0.0);
  return 0;
}
