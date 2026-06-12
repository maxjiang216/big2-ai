// eval_az_pi_match: paired-deal evaluation for the perfect-information AZ agent.
//
// Each side (--p0 / --p1) is one of:
//   pi:PATH              perfect-info AZ net + PUCT search (sees both hands)
//   classic:NAME[:PARAM] a registry policy (greedy, typed_search/tree_greedy, …)
//                        driven through the normal Player interface (its
//                        tablebase use is part of its established strength).
//
// For each deal the same shuffle is played twice with seats swapped, so card
// luck cancels. All 2*deals games run concurrently; PI leaf evals are flushed as
// batched forward passes. Reports p0's win rate (Wilson 95% CI) and sweep rate
// (winning both games of a deal). PI seats play deterministically (no Dirichlet
// noise, temperature 0).
//
//   make eval_az_pi_match
//   ./bin/eval_az_pi_match --p0 pi:models/az_pi_gen1.pt --p1 classic:greedy \
//       --deals 1000 --sims 400 --device cuda

#include "az_pi/pi_nn_eval.h"
#include "az_pi/pi_search.h"
#include "eval_helpers.h"  // wilson_ci
#include "game.h"
#include "move.h"
#include "player.h"
#include "player_factory_registry.h"
#include "util.h"

#include <torch/cuda.h>

#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace az_pi;

static const char *arg(int argc, char **argv, const char *key, const char *def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return def;
}

struct Spec {
  bool is_pi = false;
  std::string pi_path;
  std::string classic_name;
  double classic_param = 0.0;
};

static Spec parse_spec(const std::string &s) {
  Spec sp;
  if (s.rfind("pi:", 0) == 0) {
    sp.is_pi = true;
    sp.pi_path = s.substr(3);
  } else if (s.rfind("classic:", 0) == 0) {
    std::string rest = s.substr(8);
    auto colon = rest.find(':');
    if (colon == std::string::npos) {
      sp.classic_name = rest;
    } else {
      sp.classic_name = rest.substr(0, colon);
      sp.classic_param = std::atof(rest.substr(colon + 1).c_str());
    }
  } else {
    std::fprintf(stderr, "bad spec '%s' (use pi:PATH or classic:NAME[:PARAM])\n",
                 s.c_str());
    std::exit(1);
  }
  return sp;
}

struct ESlot {
  Game game;
  int assign[2] = {0, 1};  // seat -> player index (0 = p0, 1 = p1)
  std::unique_ptr<Player> classic[2];
  PiNodePool pool;  // shared node storage for this slot's trees (same thread)
  std::unique_ptr<PiSearch> tree[2];  // per seat: pi-vs-pi must not share a tree
  int sims_done = 0;
  bool waiting = false;
  bool done = false;
  int cur_eval = -1;  // player index of the parked leaf's evaluator
  PiEvalFeatures pending{};
  int winner = -1;
  int game_in_deal = 0;  // 0 or 1 (seat assignment parity)
};

int main(int argc, char **argv) {
  const Spec p0 = parse_spec(arg(argc, argv, "--p0", ""));
  const Spec p1 = parse_spec(arg(argc, argv, "--p1", ""));
  const int deals = std::atoi(arg(argc, argv, "--deals", "1000"));
  const int sims = std::atoi(arg(argc, argv, "--sims", "400"));
  const int sims_b_raw = std::atoi(arg(argc, argv, "--sims-b", "-1"));
  const int sims_b = sims_b_raw < 0 ? sims : sims_b_raw;
  const float cpuct = std::atof(arg(argc, argv, "--cpuct", "1.5"));
  const int solve_cards = std::atoi(arg(argc, argv, "--solve-cards", "12"));
  const long solve_nodes = std::atol(arg(argc, argv, "--solve-nodes", "10000"));
  const unsigned seed = std::atoi(arg(argc, argv, "--seed", "0"));
  const std::string device_s = arg(argc, argv, "--device", "cuda");
  const int threads = std::atoi(arg(argc, argv, "--threads", "0"));
  if (threads > 0) omp_set_num_threads(threads);

  torch::Device device(device_s == "cuda" && torch::cuda::is_available()
                           ? torch::kCUDA
                           : torch::kCPU);

  const Spec specs[2] = {p0, p1};
  const int sims_for[2] = {sims, sims_b};

  // Evaluators (shared when both PI seats use the same model path).
  std::unique_ptr<PiNNEvaluator> owned[2];
  PiNNEvaluator *evals[2] = {nullptr, nullptr};
  for (int i = 0; i < 2; ++i) {
    if (!specs[i].is_pi) continue;
    if (i == 1 && specs[0].is_pi && specs[0].pi_path == specs[1].pi_path) {
      evals[1] = evals[0];
      continue;
    }
    owned[i] = std::make_unique<PiNNEvaluator>(specs[i].pi_path, device);
    evals[i] = owned[i].get();
  }

  std::shared_ptr<PlayerFactory> factories[2];
  for (int i = 0; i < 2; ++i)
    if (!specs[i].is_pi)
      factories[i] = make_player_factory(specs[i].classic_name,
                                         specs[i].classic_param, seed + i);

  auto make_cfg = [&](int pi_idx) {
    PiSearchConfig cfg;
    cfg.c_puct = cpuct;
    cfg.sims = sims_for[pi_idx];
    cfg.root_noise = false;
    cfg.solver.max_total_cards = solve_cards;
    cfg.solver.node_budget = solve_nodes;
    return cfg;
  };

  const int n_slots = 2 * deals;
  std::vector<ESlot> slots(n_slots);
  for (int d = 0; d < deals; ++d) {
    std::mt19937 rng(seed + d);
    Game base;
    base.shuffle_deal(rng);
    for (int g = 0; g < 2; ++g) {
      ESlot &s = slots[2 * d + g];
      s.game = base;  // same shuffle
      s.game_in_deal = g;
      // game 0: seat0=p0, seat1=p1. game 1: seats swapped.
      s.assign[0] = (g == 0) ? 0 : 1;
      s.assign[1] = (g == 0) ? 1 : 0;
      for (int seat = 0; seat < 2; ++seat) {
        const int pidx = s.assign[seat];
        if (!specs[pidx].is_pi) {
          s.classic[seat] = factories[pidx]->create_player();
          s.classic[seat]->accept_deal(s.game, seat);
        }
      }
    }
  }

  // Apply a move: update the non-mover classic player and advance every live
  // PI tree (both seats keep their subtree across turns).
  auto apply = [&](ESlot &s, int mover, int mid) {
    Move m(mid);
    if (s.classic[1 - mover]) s.classic[1 - mover]->accept_opponent_move(m);
    s.game.apply_move(mid);
    for (auto &t : s.tree)
      if (t) t->advance_root(mid);
  };

  // Advance one slot until it parks on a PI leaf eval, or finishes.
  auto advance = [&](ESlot &s) -> bool {
    for (;;) {
      if (s.game.is_over()) {
        s.winner = s.game.get_winner();
        s.done = true;
        // Free this slot's trees and pool immediately: with thousands of
        // concurrent slots the retained high-water memory otherwise grows
        // into the many-GB range and stalls the machine.
        s.tree[0].reset();
        s.tree[1].reset();
        s.pool = PiNodePool();
        return false;
      }
      const int mover = s.game.current_player();
      const int pidx = s.assign[mover];
      if (s.classic[mover]) {  // classic mover
        apply(s, mover, encodeMove(s.classic[mover]->select_move()));
        continue;
      }
      // PI mover
      {
        const auto legal = s.game.get_legal_moves();
        if (legal.size() == 1) {  // forced: no search even with a live tree
          apply(s, mover, legal[0]);
          continue;
        }
      }
      if (!s.tree[mover]) {
        s.tree[mover] =
            std::make_unique<PiSearch>(s.game, make_cfg(pidx), &s.pool);
        s.sims_done = 0;
      }
      while (s.sims_done < sims_for[pidx] && !s.tree[mover]->root_proven()) {
        PiLeafRequest req = s.tree[mover]->select_leaf();
        if (req.needs_eval) {
          s.pending = req.feat;
          s.waiting = true;
          s.cur_eval = pidx;
          return true;
        }
        ++s.sims_done;
      }
      apply(s, mover, s.tree[mover]->best_move());
      s.sims_done = 0;
    }
  };

  auto t0 = std::chrono::steady_clock::now();
  std::vector<PiEvalFeatures> feats[2];
  std::vector<int> fslot[2];

  // Prime: pump every slot to its first parked leaf eval (or completion).
  // Slots are independent games, so the pump parallelises freely.
#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < n_slots; ++i) advance(slots[i]);

  for (;;) {
    feats[0].clear();
    feats[1].clear();
    fslot[0].clear();
    fslot[1].clear();
    for (int i = 0; i < n_slots; ++i) {
      ESlot &s = slots[i];
      if (s.waiting) {
        feats[s.cur_eval].push_back(s.pending);
        fslot[s.cur_eval].push_back(i);
      }
    }
    if (fslot[0].empty() && fslot[1].empty()) break;  // all games finished
    for (int e = 0; e < 2; ++e) {
      if (fslot[e].empty()) continue;
      auto res = evals[e]->eval_batch(feats[e]);
      // Distribute results and re-pump each slot to its next parked eval.
#pragma omp parallel for schedule(dynamic, 4)
      for (std::size_t k = 0; k < fslot[e].size(); ++k) {
        ESlot &s = slots[fslot[e][k]];
        s.tree[s.game.current_player()]->apply_eval(res[k]);
        ++s.sims_done;
        s.waiting = false;
        advance(s);
      }
    }
  }

  // Tally p0 results.
  long p0_wins = 0;
  long sweeps = 0, sweeps1 = 0;
  for (int d = 0; d < deals; ++d) {
    const ESlot &g0 = slots[2 * d];      // seat0 = p0
    const ESlot &g1 = slots[2 * d + 1];  // seat0 = p1, so p0 is seat1
    const bool p0_won_g0 = (g0.winner == 0);
    const bool p0_won_g1 = (g1.winner == 1);
    p0_wins += (p0_won_g0 ? 1 : 0) + (p0_won_g1 ? 1 : 0);
    if (p0_won_g0 && p0_won_g1) ++sweeps;
    if (!p0_won_g0 && !p0_won_g1) ++sweeps1;
  }

  const long total = 2L * deals;
  const double wr = static_cast<double>(p0_wins) / total;
  WilsonCI ci = wilson_ci(wr, total);
  const double sweep_rate = static_cast<double>(sweeps) / deals;
  const double el =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::printf("p0 = %s   p1 = %s\n", arg(argc, argv, "--p0", ""),
              arg(argc, argv, "--p1", ""));
  std::printf("deals=%d  sims=%d/%d  device=%s  %.1fs\n", deals, sims, sims_b,
              device.is_cuda() ? "cuda" : "cpu", el);
  std::printf("p0 win rate: %ld/%ld = %.4f  (Wilson95 [%.4f, %.4f])\n", p0_wins,
              total, wr, ci.lo, ci.hi);
  std::printf("p0 sweep rate (both games of a deal): %ld/%d = %.4f\n", sweeps,
              deals, sweep_rate);
  // Decisive deals (one side wins BOTH seats) carry the skill signal: split
  // deals are decided by the cards. The sweep share is a paired sign test —
  // far more sensitive than the raw rate once models are close.
  const long decisive = sweeps + sweeps1;
  if (decisive > 0) {
    const double share = static_cast<double>(sweeps) / decisive;
    WilsonCI sci = wilson_ci(share, decisive);
    std::printf(
        "decisive deals: %ld (p0 %ld, p1 %ld)  p0 sweep share %.4f  "
        "(Wilson95 [%.4f, %.4f])\n",
        decisive, sweeps, sweeps1, share, sci.lo, sci.hi);
    if (sci.lo > 0.5)
      std::printf("=> p0 significantly stronger (sweep-share CI > 0.5)\n");
    else if (sci.hi < 0.5)
      std::printf("=> p1 significantly stronger (sweep-share CI < 0.5)\n");
    else
      std::printf("=> no significant difference (sweep share)\n");
  } else {
    std::printf("decisive deals: 0 — falling back to raw win rate\n");
    if (ci.lo > 0.5)
      std::printf("=> p0 significantly stronger (95%% CI lower bound > 0.5)\n");
    else if (ci.hi < 0.5)
      std::printf("=> p1 significantly stronger\n");
    else
      std::printf("=> no significant difference\n");
  }
  return 0;
}
