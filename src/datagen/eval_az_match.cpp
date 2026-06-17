// eval_az_match: paired-deal evaluation for az_search, BATCHED.
//
// Player A is always the seq (history-transformer) az_search net (--model-a).
// Player B is one of:
//   * another seq net          (--model-b PATH)
//   * the LEGACY two-net champion (--legacy-player-b PATH --legacy-opp-b PATH)
//     — the pre-memory nets driven through the same new search core, so old
//     vs new is a clean paired comparison
//   * a classic registry policy (--classic NAME [--classic-param P])
//
// Batching: all 2*deals games run concurrently in a single-threaded pool. Each
// round, every slot is pumped until it needs an NN leaf eval (az search) or
// makes a non-NN move (forced move / CPU-opponent move); pending evals are then
// flushed as batched forward passes and distributed. Every applied move is
// appended to the slot's history (the seq net's token prefix); each az decision
// refreshes that slot's KV prefix in its evaluator before its leaf evals.
//
// For each deal we play two games from the same shuffle with seats swapped, so
// card luck cancels. Reports A's overall win rate + Wilson CI, and the PRIMARY
// metric: A's vs B's sweep rate (winning both games of a deal).
//
//   make eval_az_match
//   ./bin/eval_az_match --model-a models/az_seq.pt \
//       --classic greedy --deals 1000 --sims 100 --device cuda

#include "az_pi/pi_nn_eval.h"
#include "az_pimc/det_value.h"  // determinization-in-tree leaf-value override
#include "az_search/az_search.h"
#include "az_search/legacy_nn_eval.h"
#include "az_search/nn_eval.h"
#include "eval_helpers.h"  // wilson_ci, format_elapsed
#include "game.h"
#include "move.h"
#include "player.h"
#include "player_factory_registry.h"
#include "util.h"

#include <torch/cuda.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static SearchState mover_state(const Game &game, int mover) {
  SearchState s;
  s.our_hand = game.player_hand(mover);
  s.opp_size = game.get_player_hand_size(1 - mover);
  s.discard = game.discard_pile();
  s.last_move = encodeMove(game.last_move());
  s.side = kUs;
  return s;
}

// Evaluator view binding a pooled NNEvaluator to one slot (used by finalize's
// forced extension, which evals synchronously against that slot's prefix).
struct SlotEval : Evaluator {
  NNEvaluator &nn;
  int slot;
  SlotEval(NNEvaluator &n, int s) : nn(n), slot(s) {}
  NetEval eval(const EvalFeatures &f) override {
    return nn.eval_batch({slot}, {f})[0];
  }
};

// Like SlotEval but overrides the leaf value with the determinized PI value
// (Design 1). Used on seat A's synchronous forced-extension evals when --det-pi.
struct DetSlotEval : Evaluator {
  NNEvaluator &nn;
  int slot;
  az_pi::PiEvaluator &pi;
  int det_n;
  std::mt19937 &rng;
  DetSlotEval(NNEvaluator &n, int s, az_pi::PiEvaluator &p, int dn,
              std::mt19937 &r)
      : nn(n), slot(s), pi(p), det_n(dn), rng(r) {}
  NetEval eval(const EvalFeatures &f) override {
    NetEval e = nn.eval_batch({slot}, {f})[0];
    e.value = az_pimc::determinized_values({f}, pi, det_n, rng)[0];
    return e;
  }
};

// One concurrent game. A occupies seat `seat_a`; the other seat is B (an az tree
// if B is az, else a CPU Player). Each az seat re-roots its tree from the true
// game state + move history per decision; the CPU seat is driven via the
// Player interface.
struct ESlot {
  int gidx = -1;  // 0..2*deals-1 ; deal = gidx/2, swap = gidx%2, seat_a = swap
  Game game;
  std::mt19937 rng;
  int seat_a = 0;
  std::unique_ptr<Search> tree[2];      // az search per az-controlled seat
  std::unique_ptr<Player> cpu;          // CPU opponent (B), null if B is az
  std::vector<int> history;             // every applied move (token prefix)
  bool prefix_dirty_a = true;           // history changed since last A encode
  bool prefix_dirty_b = true;           // ... last B encode (seq B only)
  int mover = 0;
  int sims_done = 0;
  bool active = false;
  bool searching = false;
  bool waiting = false;   // has an outstanding leaf eval
  LeafRequest pending;
  int pending_seat = 0;
  bool result_ready = false;
  NetEval neval;
  bool done = false;
  int winner = -1;
};

int main(int argc, char **argv) {
  const std::string model_a = arg(argc, argv, "--model-a", "models/az_seq.pt");
  const std::string model_b = arg(argc, argv, "--model-b", "");
  const std::string legacy_player_b = arg(argc, argv, "--legacy-player-b", "");
  const std::string legacy_opp_b = arg(argc, argv, "--legacy-opp-b", "");
  const std::string classic = arg(argc, argv, "--classic", "");
  const double classic_param = std::atof(arg(argc, argv, "--classic-param", "0"));
  const int deals = std::atoi(arg(argc, argv, "--deals", "200"));
  const int sims = std::atoi(arg(argc, argv, "--sims", "100"));
  // B's az search budget (cross-play of different-sims models); defaults to A's.
  const int sims_b = std::atoi(arg(argc, argv, "--sims-b", std::to_string(sims).c_str()));
  const int slots_n = std::atoi(arg(argc, argv, "--slots", "512"));
  const unsigned base_seed = (unsigned)std::strtoul(arg(argc, argv, "--seed", "42"), nullptr, 10);
  std::string device_str = arg(argc, argv, "--device", "cuda");
  const bool use_kv_cache = !has_flag(argc, argv, "--no-kv-cache");
  // Determinization-in-tree (Design 1): override seat A's leaf VALUE with the
  // determinized PI value (uniform belief). --det-pi PATH enables it.
  const std::string det_pi = arg(argc, argv, "--det-pi", "");
  const int det_n = std::atoi(arg(argc, argv, "--det-n", "16"));
  // --det-belief PATH: AR-belief sampling (scripted Big2NetII.sample_opp) instead
  // of uniform; --det-gamma uniform-floor mix.
  const std::string det_belief = arg(argc, argv, "--det-belief", "");
  const float det_gamma = (float)std::atof(arg(argc, argv, "--det-gamma", "0.1"));

  const bool b_is_seq = classic.empty() && !model_b.empty();
  const bool b_is_legacy =
      classic.empty() && !legacy_player_b.empty() && !legacy_opp_b.empty();
  const bool b_is_az = b_is_seq || b_is_legacy;
  if (classic.empty() && !b_is_az) {
    std::fprintf(stderr,
                 "need --classic NAME, --model-b, or --legacy-player-b/--legacy-opp-b\n");
    return 2;
  }
  torch::Device device = torch::kCPU;
  if (device_str == "cuda" || (device_str == "auto" && torch::cuda::is_available()))
    device = torch::kCUDA;

  const int total = 2 * deals;
  const int pool = std::min(slots_n, total);

  NNEvaluator nnA(model_a, device, /*max_slots=*/pool, use_kv_cache);
  std::unique_ptr<az_pi::PiNNEvaluator> det_eval;
  std::unique_ptr<torch::jit::Module> det_belief_net;
  std::mt19937 det_rng(base_seed ^ 0xD37u);
  if (!det_pi.empty()) {
    det_eval = std::make_unique<az_pi::PiNNEvaluator>(det_pi, device,
                                                      /*pts_inputs=*/false);
    if (!det_belief.empty()) {
      det_belief_net =
          std::make_unique<torch::jit::Module>(torch::jit::load(det_belief, device));
      det_belief_net->eval();
    }
    std::printf("[det] seat A leaf value = determinized PI (%s, N=%d, belief=%s)\n",
                det_pi.c_str(), det_n,
                det_belief.empty() ? "uniform" : det_belief.c_str());
  }
  std::unique_ptr<NNEvaluator> nnB;
  std::unique_ptr<LegacyNNEvaluator> legB;
  std::shared_ptr<PlayerFactory> factory_b;
  std::string b_label;
  if (b_is_seq) {
    nnB = std::make_unique<NNEvaluator>(model_b, device, pool, use_kv_cache);
    b_label = "az:" + model_b;
  } else if (b_is_legacy) {
    legB = std::make_unique<LegacyNNEvaluator>(legacy_player_b, legacy_opp_b, device);
    b_label = "az-legacy:" + legacy_player_b;
  } else {
    factory_b = make_player_factory(classic, classic_param, base_seed + 1);
    if (!factory_b) return 1;
    b_label = "classic:" + classic;
  }

  std::printf("eval_az_match (batched): A=az:%s  vs  B=%s  (%d deals x2, sims=%d, slots=%d, %s)\n",
              model_a.c_str(), b_label.c_str(), deals, sims, pool,
              device == torch::kCUDA ? "cuda" : "cpu");

  std::vector<char> a_won(total, 0);  // result per game index
  std::vector<ESlot> slots(pool);
  int next_game = 0;

  auto start_game = [&](ESlot &s) -> bool {
    if (next_game >= total) return false;
    s.gidx = next_game++;
    const int deal = s.gidx / 2, swap = s.gidx % 2;
    s.seat_a = swap;  // A plays seat 0 in swap 0, seat 1 in swap 1
    s.rng.seed(base_seed + (unsigned)deal);  // same shuffle for both swaps -> paired
    s.game = Game();
    s.game.shuffle_deal(s.rng);
    s.tree[0].reset(); s.tree[1].reset();
    s.cpu.reset();
    if (!b_is_az) {
      s.cpu = factory_b->create_player();
      s.cpu->accept_deal(s.game, 1 - s.seat_a);
    }
    s.history.clear();
    s.prefix_dirty_a = s.prefix_dirty_b = true;
    s.mover = 0; s.sims_done = 0;
    s.active = true; s.searching = false; s.waiting = false; s.done = false;
    return true;
  };

  auto seat_is_az = [&](const ESlot &s, int seat) {
    return seat == s.seat_a || b_is_az;  // A always az; B az iff b_is_az
  };

  // Per-flush eval queues. A and seq-B rows carry the ESlot index as the KV
  // slot id; legacy-B rows are slot-free.
  struct Flush {
    std::vector<EvalFeatures> aF; std::vector<int> aS;
    std::vector<EvalFeatures> bF; std::vector<int> bS;
    std::vector<EvalFeatures> lF; std::vector<int> lS;
    bool empty() const { return aF.empty() && bF.empty() && lF.empty(); }
  };

  // Apply a real move everywhere it must land: game, history, CPU mirror.
  auto commit_move = [&](ESlot &s, int m, bool from_cpu) {
    s.game.apply_move(m);
    s.history.push_back(m);
    s.prefix_dirty_a = s.prefix_dirty_b = true;
    if (s.cpu && !from_cpu) s.cpu->accept_opponent_move(Move(m));
  };

  // Pump one slot until it yields (enqueued a leaf eval) or its work is done.
  auto advance = [&](ESlot &s, int idx, Flush &fl) -> void {
    for (;;) {
      if (!s.active) {
        if (!start_game(s)) return;
        continue;
      }
      if (s.waiting) {
        if (!s.result_ready) return;  // parked until the batch flush
        s.tree[s.pending_seat]->apply_eval(s.neval);
        s.result_ready = false; s.waiting = false; ++s.sims_done;
        continue;
      }
      if (s.game.is_over()) {
        s.winner = s.game.get_winner();
        a_won[s.gidx] = (s.winner == s.seat_a) ? 1 : 0;
        s.active = false;
        continue;  // start next game (or finish)
      }
      const int mover = s.game.current_player();
      if (!seat_is_az(s, mover)) {
        // CPU opponent: select_move() handles forced/tablebase moves AND updates
        // the player's own PartialGame, so it must drive every CPU-seat turn.
        Move mv = s.cpu->select_move();
        commit_move(s, encodeMove(mv), /*from_cpu=*/true);
        continue;
      }
      auto legal = s.game.get_legal_moves();
      if (legal.size() == 1) {  // az seat, forced move: skip the search
        commit_move(s, legal[0], /*from_cpu=*/false);
        continue;
      }
      // az decision at `mover`
      const bool useA = (mover == s.seat_a);
      const int seat_sims = useA ? sims : sims_b;
      if (!s.searching) {
        SearchState st = mover_state(s.game, mover);
        // Match the play path's root-skip: definitive (hand-emptying / opp-1
        // tablebase / forced-win) positions are played without search.
        if (auto d = az_definitive_move(st, /*allow_forced_win=*/true)) {
          commit_move(s, *d, /*from_cpu=*/false);
          continue;
        }
        unsigned tseed = base_seed ^ (0x9E3779B9u * (unsigned)(s.gidx * 2 + mover));
        if (!s.tree[mover])
          s.tree[mover] = std::make_unique<Search>(
              st, s.history, SearchConfig{1.5f, seat_sims, tseed, false});
        else
          s.tree[mover]->advance_root(st, s.history);
        s.mover = mover; s.sims_done = 0; s.searching = true;
        // Refresh the mover's KV prefix SYNCHRONOUSLY: finalize's forced
        // extension may eval against it before any batched flush happens
        // (zero-eval decisions on fully reused subtrees). Prefix encodes are
        // one per decision (cheap next to sims evals), so unbatched is fine.
        if (useA) {
          if (s.prefix_dirty_a) {
            nnA.set_prefixes({idx}, {&s.history});
            s.prefix_dirty_a = false;
          }
        } else if (b_is_seq) {
          if (s.prefix_dirty_b) {
            nnB->set_prefixes({idx}, {&s.history});
            s.prefix_dirty_b = false;
          }
        }
      }
      bool pushed = false;
      while (s.sims_done < seat_sims) {
        LeafRequest req = s.tree[mover]->select_leaf();
        if (!req.needs_eval) { ++s.sims_done; continue; }
        s.pending = std::move(req); s.pending_seat = mover;
        s.waiting = true;
        if (useA) { fl.aF.push_back(std::move(s.pending.feat)); fl.aS.push_back(idx); }
        else if (b_is_seq) { fl.bF.push_back(std::move(s.pending.feat)); fl.bS.push_back(idx); }
        else { fl.lF.push_back(std::move(s.pending.feat)); fl.lS.push_back(idx); }
        pushed = true;
        break;
      }
      if (pushed) return;  // parked until the batch flush
      // sims complete -> commit the move. finalize(&ev) runs the NN-valued
      // forced-move extension (matches run()); the few forced leaves evaluate
      // synchronously against this slot's prefix (rare, un-batched is fine).
      if (useA) {
        if (det_eval) {
          DetSlotEval ev(nnA, idx, *det_eval, det_n, det_rng);
          s.tree[mover]->finalize(&ev);
        } else {
          SlotEval ev(nnA, idx);
          s.tree[mover]->finalize(&ev);
        }
      } else if (b_is_seq) {
        SlotEval ev(*nnB, idx);
        s.tree[mover]->finalize(&ev);
      } else {
        s.tree[mover]->finalize(legB.get());
      }
      commit_move(s, s.tree[mover]->best_move(), /*from_cpu=*/false);
      s.searching = false;
    }
  };

  auto t0 = std::chrono::steady_clock::now();
  // Prime slots.
  for (auto &s : slots) start_game(s);

  for (;;) {
    Flush fl;
    bool any_active = false;
    for (int i = 0; i < pool; ++i) {
      ESlot &s = slots[i];
      if (s.active || next_game < total) any_active = true;
      advance(s, i, fl);
    }
    if (fl.empty()) {
      if (!any_active) break;       // all games finished
      bool still = false;
      for (int i = 0; i < pool; ++i) if (slots[i].active) still = true;
      if (!still && next_game >= total) break;
      continue;
    }
    if (!fl.aF.empty()) {
      auto r = nnA.eval_batch(fl.aS, fl.aF);
      if (det_eval) {  // override seat A's leaf values with determinized PI value
        std::vector<float> dv;
        if (det_belief_net) {
          // full history per leaf = that slot's game prefix + in-tree path tokens
          std::vector<std::vector<int>> fh(fl.aF.size());
          for (size_t i = 0; i < fl.aF.size(); ++i) {
            fh[i] = slots[fl.aS[i]].history;
            fh[i].insert(fh[i].end(), fl.aF[i].path_tokens.begin(),
                         fl.aF[i].path_tokens.end());
          }
          dv = az_pimc::determinized_values_ar(fl.aF, fh, *det_belief_net,
                                               *det_eval, det_n, det_gamma,
                                               det_rng, device);
        } else {
          dv = az_pimc::determinized_values(fl.aF, *det_eval, det_n, det_rng);
        }
        for (size_t i = 0; i < dv.size(); ++i) r[i].value = dv[i];
      }
      for (size_t i = 0; i < fl.aS.size(); ++i) {
        slots[fl.aS[i]].neval = r[i];
        slots[fl.aS[i]].result_ready = true;
      }
    }
    if (!fl.bF.empty()) {
      auto r = nnB->eval_batch(fl.bS, fl.bF);
      for (size_t i = 0; i < fl.bS.size(); ++i) {
        slots[fl.bS[i]].neval = r[i];
        slots[fl.bS[i]].result_ready = true;
      }
    }
    if (!fl.lF.empty()) {
      auto r = legB->eval_batch(fl.lF);
      for (size_t i = 0; i < fl.lS.size(); ++i) {
        slots[fl.lS[i]].neval = r[i];
        slots[fl.lS[i]].result_ready = true;
      }
    }
  }
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();

  long a_wins = 0;
  long a_sweep = 0, b_sweep = 0, split = 0;
  for (int d = 0; d < deals; ++d) {
    const int w = a_won[2 * d] + a_won[2 * d + 1];
    a_wins += w;
    if (w == 2) ++a_sweep; else if (w == 0) ++b_sweep; else ++split;
  }
  const long games = total;
  const double p = (double)a_wins / (double)games;
  WilsonCI ci = wilson_ci(p, games);
  std::printf("A wins %ld / %ld = %.3f  (Wilson 95%% CI [%.3f, %.3f])  [%s]\n",
              a_wins, games, p, ci.lo, ci.hi, format_elapsed(ms).c_str());
  if (ci.lo > 0.5) std::printf("  overall: A significantly stronger (CI above 50%%).\n");
  else if (ci.hi < 0.5) std::printf("  overall: B significantly stronger (CI below 50%%).\n");
  else std::printf("  overall: not significant at 95%% (CI straddles 50%%).\n");

  const long decisive = a_sweep + b_sweep;
  std::printf("sweeps: A=%ld  B=%ld  split=%ld  (matches=%d)\n", a_sweep, b_sweep, split, deals);
  if (decisive > 0) {
    const double sp = (double)a_sweep / (double)decisive;
    WilsonCI sci = wilson_ci(sp, decisive);
    std::printf("  A sweep share %ld/%ld = %.3f  (Wilson 95%% CI [%.3f, %.3f])\n",
                a_sweep, decisive, sp, sci.lo, sci.hi);
    if (sci.lo > 0.5) std::printf("  A SWEEPS significantly more than B (95%%).\n");
    else if (sci.hi < 0.5) std::printf("  B sweeps significantly more than A (95%%).\n");
    else std::printf("  sweep ratio not significant at 95%%.\n");
  }
  return 0;
}
