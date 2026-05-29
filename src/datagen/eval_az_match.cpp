// eval_az_match: paired-deal evaluation for az_search, BATCHED.
//
// Player A is always az_search (--player-a / --opp-a). Player B is either
// another az_search model pair (--player-b / --opp-b) or a classic registry
// policy (--classic NAME [--classic-param P]).
//
// Batching: all 2*deals games run concurrently in a single-threaded pool. Each
// round, every slot is pumped until it needs an NN leaf eval (az search) or
// makes a non-NN move (forced move / CPU-opponent move); all pending leaf evals
// are then flushed as batched forward passes (A's nets via nnA, B's az nets via
// nnB) and distributed. This batches the dominant az inference across games
// (~the self-play throughput) while game logic stays exact — CPU opponents use
// the normal Player interface (accept_deal / accept_opponent_move / select_move).
//
// For each deal we play two games from the same shuffle with seats swapped, so
// card luck cancels. Reports A's overall win rate + Wilson CI, and the PRIMARY
// metric: A's vs B's sweep rate (winning both games of a deal).
//
//   make eval_az_match
//   ./bin/eval_az_match --player-a models/az_player.pt --opp-a models/az_opp.pt \
//       --classic greedy --deals 1000 --sims 100 --device cuda

#include "az_search/az_search.h"
#include "az_search/eval_cache.h"  // CachingEvaluator (for finalize's forced extension)
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

static SearchState mover_state(const Game &game, int mover) {
  SearchState s;
  s.our_hand = game.player_hand(mover);
  s.opp_size = game.get_player_hand_size(1 - mover);
  s.discard = game.discard_pile();
  s.last_move = encodeMove(game.last_move());
  s.side = kUs;
  return s;
}

// One concurrent game. A occupies seat `seat_a`; the other seat is B (an az tree
// if B is az, else a CPU Player). Each az seat re-roots its tree from the true
// game state per decision; the CPU seat is driven via the Player interface.
struct ESlot {
  int gidx = -1;  // 0..2*deals-1 ; deal = gidx/2, swap = gidx%2, seat_a = swap
  Game game;
  std::mt19937 rng;
  int seat_a = 0;
  std::unique_ptr<Search> tree[2];      // az search per az-controlled seat
  std::unique_ptr<Player> cpu;          // CPU opponent (B), null if B is az
  int mover = 0;
  int sims_done = 0;
  bool active = false;
  bool searching = false;
  bool waiting = false;   // has an outstanding leaf eval
  LeafRequest pending;
  int pending_seat = 0;
  bool pending_is_player = false;
  bool result_ready = false;
  PlayerEval peval;
  OppEval oeval;
  bool done = false;
  int winner = -1;
};

int main(int argc, char **argv) {
  const std::string player_a = arg(argc, argv, "--player-a", "models/az_player.pt");
  const std::string opp_a = arg(argc, argv, "--opp-a", "models/az_opp.pt");
  const std::string player_b = arg(argc, argv, "--player-b", "");
  const std::string opp_b = arg(argc, argv, "--opp-b", "");
  const std::string classic = arg(argc, argv, "--classic", "");
  const double classic_param = std::atof(arg(argc, argv, "--classic-param", "0"));
  const int deals = std::atoi(arg(argc, argv, "--deals", "200"));
  const int sims = std::atoi(arg(argc, argv, "--sims", "100"));
  const int slots_n = std::atoi(arg(argc, argv, "--slots", "512"));
  const unsigned base_seed = (unsigned)std::strtoul(arg(argc, argv, "--seed", "42"), nullptr, 10);
  std::string device_str = arg(argc, argv, "--device", "cuda");

  const bool b_is_az = classic.empty() && !player_b.empty() && !opp_b.empty();
  if (classic.empty() && !b_is_az) {
    std::fprintf(stderr, "need either --classic NAME or --player-b/--opp-b\n");
    return 2;
  }
  torch::Device device = torch::kCPU;
  if (device_str == "cuda" || (device_str == "auto" && torch::cuda::is_available()))
    device = torch::kCUDA;

  NNEvaluator nnA(player_a, opp_a, device);
  std::unique_ptr<NNEvaluator> nnB;
  std::shared_ptr<PlayerFactory> factory_b;
  std::string b_label;
  if (b_is_az) {
    nnB = std::make_unique<NNEvaluator>(player_b, opp_b, device);
    b_label = "az:" + player_b;
  } else {
    factory_b = make_player_factory(classic, classic_param, base_seed + 1);
    if (!factory_b) return 1;
    b_label = "classic:" + classic;
  }

  const int total = 2 * deals;
  const int pool = std::min(slots_n, total);
  std::printf("eval_az_match (batched): A=az:%s  vs  B=%s  (%d deals x2, sims=%d, slots=%d, %s)\n",
              player_a.c_str(), b_label.c_str(), deals, sims, pool,
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
    s.mover = 0; s.sims_done = 0;
    s.active = true; s.searching = false; s.waiting = false; s.done = false;
    return true;
  };

  auto seat_is_az = [&](const ESlot &s, int seat) {
    return seat == s.seat_a || b_is_az;  // A always az; B az iff b_is_az
  };

  // Pump one slot until it yields (enqueued a leaf eval) or its work is done.
  // Returns true if it enqueued a pending eval (needs a flush before resuming).
  auto advance = [&](ESlot &s, int idx,
                     std::vector<PlayerFeatures> &aP, std::vector<int> &aPs,
                     std::vector<OppFeatures> &aO, std::vector<int> &aOs,
                     std::vector<PlayerFeatures> &bP, std::vector<int> &bPs,
                     std::vector<OppFeatures> &bO, std::vector<int> &bOs) -> void {
    for (;;) {
      if (!s.active) {
        if (!start_game(s)) return;
        continue;
      }
      if (s.waiting) {
        if (!s.result_ready) return;  // shouldn't happen single-threaded, but safe
        if (s.pending_is_player) s.tree[s.pending_seat]->apply_eval(s.peval);
        else                     s.tree[s.pending_seat]->apply_eval(s.oeval);
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
        s.game.apply_move(mv);
        continue;
      }
      auto legal = s.game.get_legal_moves();
      if (legal.size() == 1) {  // az seat, forced move: skip the search
        const int m = legal[0];
        s.game.apply_move(m);
        if (s.cpu) s.cpu->accept_opponent_move(Move(m));
        continue;
      }
      // az decision at `mover`
      if (!s.searching) {
        SearchState st = mover_state(s.game, mover);
        // Match the play path's root-skip: definitive (hand-emptying / opp-1
        // tablebase / forced-win) positions are played without search.
        if (auto d = az_definitive_move(st, /*allow_forced_win=*/true)) {
          s.game.apply_move(*d);
          if (s.cpu) s.cpu->accept_opponent_move(Move(*d));
          continue;
        }
        unsigned tseed = base_seed ^ (0x9E3779B9u * (unsigned)(s.gidx * 2 + mover));
        if (!s.tree[mover])
          s.tree[mover] = std::make_unique<Search>(st, SearchConfig{1.5f, sims, tseed, false});
        else
          s.tree[mover]->advance_root(st);
        s.mover = mover; s.sims_done = 0; s.searching = true;
      }
      bool pushed = false;
      while (s.sims_done < sims) {
        LeafRequest req = s.tree[mover]->select_leaf();
        if (!req.needs_eval) { ++s.sims_done; continue; }
        s.pending = req; s.pending_seat = mover; s.pending_is_player = req.is_player;
        s.waiting = true;
        const bool useA = (mover == s.seat_a);
        if (req.is_player) {
          (useA ? aP : bP).push_back(req.pfeat);
          (useA ? aPs : bPs).push_back(idx);
        } else {
          (useA ? aO : bO).push_back(req.ofeat);
          (useA ? aOs : bOs).push_back(idx);
        }
        pushed = true;
        break;
      }
      if (pushed) return;  // parked until the batch flush
      // sims complete -> commit the move. finalize(&ev) runs the NN-valued
      // forced-move extension (matches run()); a fresh cache evaluates the few
      // forced leaves synchronously (rare, so un-batched is fine).
      {
        NNEvaluator &netf = (mover == s.seat_a) ? nnA : *nnB;
        CachingEvaluator cache(netf);
        s.tree[mover]->finalize(&cache);
      }
      const int m = s.tree[mover]->best_move();
      s.game.apply_move(m);
      if (s.cpu) s.cpu->accept_opponent_move(Move(m));  // az seat -> feed CPU opponent
      s.searching = false;
    }
  };

  auto t0 = std::chrono::steady_clock::now();
  // Prime slots.
  for (auto &s : slots) start_game(s);

  for (;;) {
    std::vector<PlayerFeatures> aP, bP; std::vector<int> aPs, bPs;
    std::vector<OppFeatures> aO, bO; std::vector<int> aOs, bOs;
    bool any_active = false;
    for (int i = 0; i < pool; ++i) {
      ESlot &s = slots[i];
      if (s.active || next_game < total) any_active = true;
      advance(s, i, aP, aPs, aO, aOs, bP, bPs, bO, bOs);
    }
    if (aP.empty() && aO.empty() && bP.empty() && bO.empty()) {
      if (!any_active) break;       // all games finished
      // No pending evals but games remain: everything resolved without NN this
      // round (e.g. all CPU/forced). Loop again (slots advanced in-place).
      bool still = false;
      for (int i = 0; i < pool; ++i) if (slots[i].active) still = true;
      if (!still && next_game >= total) break;
      continue;
    }
    if (!aP.empty()) { auto r = nnA.eval_players(aP); for (size_t i = 0; i < aPs.size(); ++i) { slots[aPs[i]].peval = r[i]; slots[aPs[i]].result_ready = true; } }
    if (!aO.empty()) { auto r = nnA.eval_opps(aO);    for (size_t i = 0; i < aOs.size(); ++i) { slots[aOs[i]].oeval = r[i]; slots[aOs[i]].result_ready = true; } }
    if (!bP.empty()) { auto r = nnB->eval_players(bP);for (size_t i = 0; i < bPs.size(); ++i) { slots[bPs[i]].peval = r[i]; slots[bPs[i]].result_ready = true; } }
    if (!bO.empty()) { auto r = nnB->eval_opps(bO);   for (size_t i = 0; i < bOs.size(); ++i) { slots[bOs[i]].oeval = r[i]; slots[bOs[i]].result_ready = true; } }
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
