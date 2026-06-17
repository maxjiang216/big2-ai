// eval_az_series: full-SERIES evaluation for az_search (batched).
//
// The series objective's true metric: play complete series (first to 50 points,
// winner leads the next game, 3-of-spades opens at 0-0) and report A's series
// win rate. Each slot in the batched pool runs ONE ongoing series: when a game
// ends we apply scoring (loser's remaining cards -> points, see series.h), and
// either deal the next game (series continues) or record the series result and
// start a fresh series.
//
// Player A is the seq az_search net (--model-a). Player B is another seq net
// (--model-b) or a classic registry policy (--classic NAME). Both az seats use
// the loaded series V-table (--series-v) so leaf / terminal values are series
// win probabilities and the net conditions on the match state.
//
// Variance reduction (mirrored seeds): series come in pairs. Series 2j and 2j+1
// share the same per-game deal RNG stream (game k of each draws the same deck
// and 3-of-spades opener) but swap which seat A controls. Pairing is exact for
// game 0 and decays as series lengths diverge — still a real, free reduction.
//
//   make eval_az_series
//   ./bin/eval_az_series --model-a models/az_seq_series.pt \
//       --model-b models/az_seq_series.pt --series-v data/series_v_gen3.csv \
//       --pairs 500 --sims 100 --device cuda [--dump-outcomes data/eval_outcomes.csv]

#include "az_search/az_search.h"
#include "az_search/nn_eval.h"
#include "eval_helpers.h"  // wilson_ci, format_elapsed
#include "game.h"
#include "move.h"
#include "player.h"
#include "player_factory_registry.h"
#include "series.h"
#include "util.h"

#include <torch/cuda.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

static SearchState mover_state(const Game &game, int mover, const int pts[2]) {
  SearchState s;
  s.our_hand = game.player_hand(mover);
  s.opp_size = game.get_player_hand_size(1 - mover);
  s.discard = game.discard_pile();
  s.last_move = encodeMove(game.last_move());
  s.side = kUs;
  s.my_pts = pts[mover];
  s.opp_pts = pts[1 - mover];
  return s;
}

struct SlotEval : Evaluator {
  NNEvaluator &nn;
  int slot;
  SlotEval(NNEvaluator &n, int s) : nn(n), slot(s) {}
  NetEval eval(const EvalFeatures &f) override { return nn.eval_batch({slot}, {f})[0]; }
};

// One per-game outcome record (for --dump-outcomes / Markov-chain re-use).
struct Outcome {
  int a, b, first_player, winner, loser_cards, model_seat;
};

// One concurrent SERIES.
struct SSlot {
  int sidx = -1;       // series index 0..2*pairs-1
  int pair = 0, swap = 0, seat_a = 0;
  int pts[2] = {0, 0}; // seat-indexed series points
  int leader = 0;      // seat that opens the current game
  int game_no = 0;     // game number within the series
  int games_in_series = 0;

  Game game;
  std::mt19937 rng;
  std::unique_ptr<Search> tree[2];
  std::unique_ptr<Player> cpu;
  std::vector<int> history;
  bool prefix_dirty_a = true, prefix_dirty_b = true;
  int mover = 0, sims_done = 0;
  bool active = false, searching = false, waiting = false;
  LeafRequest pending;
  int pending_seat = 0;
  bool result_ready = false;
  NetEval neval;
};

int main(int argc, char **argv) {
  const std::string model_a = arg(argc, argv, "--model-a", "models/az_seq.pt");
  const std::string model_b = arg(argc, argv, "--model-b", "");
  const std::string classic = arg(argc, argv, "--classic", "");
  const double classic_param = std::atof(arg(argc, argv, "--classic-param", "0"));
  const std::string series_v = arg(argc, argv, "--series-v", "");
  const std::string dump_path = arg(argc, argv, "--dump-outcomes", "");
  const int pairs = std::atoi(arg(argc, argv, "--pairs", "200"));
  const int sims = std::atoi(arg(argc, argv, "--sims", "100"));
  const int sims_b = std::atoi(arg(argc, argv, "--sims-b", std::to_string(sims).c_str()));
  const int slots_n = std::atoi(arg(argc, argv, "--slots", "512"));
  const unsigned base_seed = (unsigned)std::strtoul(arg(argc, argv, "--seed", "42"), nullptr, 10);
  std::string device_str = arg(argc, argv, "--device", "cuda");
  const bool use_kv_cache = !has_flag(argc, argv, "--no-kv-cache");

  const bool b_is_seq = classic.empty() && !model_b.empty();
  if (classic.empty() && !b_is_seq) {
    std::fprintf(stderr, "need --classic NAME or --model-b PATH\n");
    return 2;
  }
  SeriesTable table;
  if (series_v.empty() || !load_series_table(series_v, table)) {
    std::fprintf(stderr, "--series-v <csv> is required and must load\n");
    return 2;
  }
  torch::Device device = torch::kCPU;
  if (device_str == "cuda" || (device_str == "auto" && torch::cuda::is_available()))
    device = torch::kCUDA;

  const int total_series = 2 * pairs;
  const int pool = std::min(slots_n, total_series);

  NNEvaluator nnA(model_a, device, /*max_slots=*/pool, use_kv_cache);
  std::unique_ptr<NNEvaluator> nnB;
  std::shared_ptr<PlayerFactory> factory_b;
  std::string b_label;
  if (b_is_seq) {
    nnB = std::make_unique<NNEvaluator>(model_b, device, pool, use_kv_cache);
    b_label = "az:" + model_b;
  } else {
    factory_b = make_player_factory(classic, classic_param, base_seed + 1);
    if (!factory_b) return 1;
    b_label = "classic:" + classic;
  }

  std::printf("eval_az_series (batched): A=az:%s  vs  B=%s  (%d pairs = %d series, "
              "sims=%d, slots=%d, %s)\n",
              model_a.c_str(), b_label.c_str(), pairs, total_series, sims, pool,
              device == torch::kCUDA ? "cuda" : "cpu");

  std::vector<char> a_won_series(total_series, 0);
  std::vector<int> series_len(total_series, 0);
  std::vector<SSlot> slots(pool);
  int next_series = 0;

  // Aggregate per-game stats (single-threaded pool, no locking needed).
  long total_games = 0, a_game_wins = 0;
  double a_pts_for = 0.0, a_pts_against = 0.0;
  std::vector<Outcome> outcomes;

  // Per-game deal seed: identical for both swaps of a pair at the same game_no.
  auto deal_seed = [&](int pair, int gno) -> unsigned {
    return base_seed ^ (0x9E3779B9u * (unsigned)(pair * 131 + gno + 1));
  };

  // Deal the next game in a series: shuffle, set the opener (3-spades at game 0,
  // else the previous winner), reset per-game state.
  auto deal_game = [&](SSlot &s) {
    s.rng.seed(deal_seed(s.pair, s.game_no));
    s.game = Game();
    s.game.shuffle_deal(s.rng);
    if (s.game_no == 0)
      s.leader = sample_first_player_3s(s.game.player_hand(0),
                                        s.game.player_hand(1), s.rng);
    s.game.set_first_player(s.leader);
    s.tree[0].reset(); s.tree[1].reset();
    s.cpu.reset();
    if (!b_is_seq) {
      s.cpu = factory_b->create_player();
      s.cpu->accept_deal(s.game, 1 - s.seat_a);
    }
    s.history.clear();
    s.prefix_dirty_a = s.prefix_dirty_b = true;
    s.mover = 0; s.sims_done = 0;
    s.searching = false; s.waiting = false;
  };

  auto start_series = [&](SSlot &s) -> bool {
    if (next_series >= total_series) return false;
    s.sidx = next_series++;
    s.pair = s.sidx / 2;
    s.swap = s.sidx % 2;
    s.seat_a = s.swap;  // A controls seat `swap`
    s.pts[0] = s.pts[1] = 0;
    s.leader = 0;
    s.game_no = 0;
    s.games_in_series = 0;
    s.active = true;
    deal_game(s);
    return true;
  };

  auto seat_is_az = [&](const SSlot &s, int seat) {
    return seat == s.seat_a || b_is_seq;
  };

  struct Flush {
    std::vector<EvalFeatures> aF; std::vector<int> aS;
    std::vector<EvalFeatures> bF; std::vector<int> bS;
    bool empty() const { return aF.empty() && bF.empty(); }
  };

  auto commit_move = [&](SSlot &s, int m, bool from_cpu) {
    s.game.apply_move(m);
    s.history.push_back(m);
    s.prefix_dirty_a = s.prefix_dirty_b = true;
    if (s.cpu && !from_cpu) s.cpu->accept_opponent_move(Move(m));
  };

  // Game over: score it, update the series, then either deal the next game or
  // finish the series.
  auto on_game_over = [&](SSlot &s) {
    const int w = s.game.get_winner();
    const int loser = 1 - w;
    const int lc = s.game.get_player_hand_size(loser);
    const int p = points_for_cards(lc);
    ++total_games;
    ++s.games_in_series;
    if (w == s.seat_a) { ++a_game_wins; a_pts_for += p; }
    else a_pts_against += p;
    if (!dump_path.empty()) {
      Outcome o;
      o.a = s.pts[s.leader];
      o.b = s.pts[1 - s.leader];
      o.first_player = s.leader;
      o.winner = w;
      o.loser_cards = lc;
      o.model_seat = s.seat_a;
      outcomes.push_back(o);
    }
    s.pts[w] += p;
    if (s.pts[w] >= kSeriesTarget) {
      a_won_series[s.sidx] = (w == s.seat_a) ? 1 : 0;
      series_len[s.sidx] = s.games_in_series;
      s.active = false;  // a fresh series starts on the next advance()
      return;
    }
    s.leader = w;       // winner opens the next game
    ++s.game_no;
    deal_game(s);
  };

  auto advance = [&](SSlot &s, int idx, Flush &fl) -> void {
    for (;;) {
      if (!s.active) {
        if (!start_series(s)) return;
        continue;
      }
      if (s.waiting) {
        if (!s.result_ready) return;
        s.tree[s.pending_seat]->apply_eval(s.neval);
        s.result_ready = false; s.waiting = false; ++s.sims_done;
        continue;
      }
      if (s.game.is_over()) { on_game_over(s); continue; }
      const int mover = s.game.current_player();
      if (!seat_is_az(s, mover)) {
        Move mv = s.cpu->select_move();
        commit_move(s, encodeMove(mv), /*from_cpu=*/true);
        continue;
      }
      auto legal = s.game.get_legal_moves();
      if (legal.size() == 1) { commit_move(s, legal[0], false); continue; }
      const bool useA = (mover == s.seat_a);
      const int seat_sims = useA ? sims : sims_b;
      if (!s.searching) {
        SearchState st = mover_state(s.game, mover, s.pts);
        if (auto d = az_definitive_move(st, /*allow_forced_win=*/true)) {
          commit_move(s, *d, false);
          continue;
        }
        unsigned tseed = base_seed ^ (0x9E3779B9u * (unsigned)(s.sidx * 64 + s.game_no * 2 + mover));
        SearchConfig sc{1.5f, seat_sims, tseed, false};
        sc.series = &table;
        if (!s.tree[mover])
          s.tree[mover] = std::make_unique<Search>(st, s.history, sc);
        else
          s.tree[mover]->advance_root(st, s.history);
        s.mover = mover; s.sims_done = 0; s.searching = true;
        if (useA) {
          if (s.prefix_dirty_a) { nnA.set_prefixes({idx}, {&s.history}); s.prefix_dirty_a = false; }
        } else if (b_is_seq) {
          if (s.prefix_dirty_b) { nnB->set_prefixes({idx}, {&s.history}); s.prefix_dirty_b = false; }
        }
      }
      bool pushed = false;
      while (s.sims_done < seat_sims) {
        LeafRequest req = s.tree[mover]->select_leaf();
        if (!req.needs_eval) { ++s.sims_done; continue; }
        s.pending = std::move(req); s.pending_seat = mover;
        s.waiting = true;
        if (useA) { fl.aF.push_back(std::move(s.pending.feat)); fl.aS.push_back(idx); }
        else { fl.bF.push_back(std::move(s.pending.feat)); fl.bS.push_back(idx); }
        pushed = true;
        break;
      }
      if (pushed) return;
      if (useA) { SlotEval ev(nnA, idx); s.tree[mover]->finalize(&ev); }
      else { SlotEval ev(*nnB, idx); s.tree[mover]->finalize(&ev); }
      commit_move(s, s.tree[mover]->best_move(), false);
      s.searching = false;
    }
  };

  auto t0 = std::chrono::steady_clock::now();
  for (auto &s : slots) start_series(s);

  for (;;) {
    Flush fl;
    bool any_active = false;
    for (int i = 0; i < pool; ++i) {
      if (slots[i].active || next_series < total_series) any_active = true;
      advance(slots[i], i, fl);
    }
    if (fl.empty()) {
      if (!any_active) break;
      bool still = false;
      for (int i = 0; i < pool; ++i) if (slots[i].active) still = true;
      if (!still && next_series >= total_series) break;
      continue;
    }
    if (!fl.aF.empty()) {
      auto r = nnA.eval_batch(fl.aS, fl.aF);
      for (size_t i = 0; i < fl.aS.size(); ++i) { slots[fl.aS[i]].neval = r[i]; slots[fl.aS[i]].result_ready = true; }
    }
    if (!fl.bF.empty()) {
      auto r = nnB->eval_batch(fl.bS, fl.bF);
      for (size_t i = 0; i < fl.bS.size(); ++i) { slots[fl.bS[i]].neval = r[i]; slots[fl.bS[i]].result_ready = true; }
    }
  }
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();

  // --- primary: series win rate ---
  long a_series = 0;
  for (char c : a_won_series) a_series += c;
  const double sp = (double)a_series / (double)total_series;
  WilsonCI ci = wilson_ci(sp, total_series);
  std::printf("A wins %ld / %d series = %.3f  (Wilson 95%% CI [%.3f, %.3f])  [%s]\n",
              a_series, total_series, sp, ci.lo, ci.hi, format_elapsed(ms).c_str());
  if (ci.lo > 0.5) std::printf("  series: A significantly stronger (CI above 50%%).\n");
  else if (ci.hi < 0.5) std::printf("  series: B significantly stronger (CI below 50%%).\n");
  else std::printf("  series: not significant at 95%% (CI straddles 50%%).\n");

  // mirrored-pair sweeps (A wins both series of a pair / B does / split).
  long a_sweep = 0, b_sweep = 0, split = 0;
  for (int j = 0; j < pairs; ++j) {
    int w = a_won_series[2 * j] + a_won_series[2 * j + 1];
    if (w == 2) ++a_sweep; else if (w == 0) ++b_sweep; else ++split;
  }
  std::printf("pair sweeps: A=%ld  B=%ld  split=%ld  (pairs=%d)\n",
              a_sweep, b_sweep, split, pairs);

  // --- secondary: per-game stats ---
  const double gp = (double)a_game_wins / (double)std::max<long>(total_games, 1);
  double mean_len = (double)total_games / (double)std::max(total_series, 1);
  std::printf("games: A win rate %.3f (%ld/%ld)  mean series length %.2f\n",
              gp, a_game_wins, total_games, mean_len);
  std::printf("points: A scored/game(won) %.2f  conceded/game(lost) %.2f\n",
              a_pts_for / std::max<long>(a_game_wins, 1),
              a_pts_against / std::max<long>(total_games - a_game_wins, 1));

  if (!dump_path.empty()) {
    std::ofstream out(dump_path);
    out << "a,b,first_player,winner,loser_cards,model_seat\n";
    for (const auto &o : outcomes)
      out << o.a << ',' << o.b << ',' << o.first_player << ',' << o.winner << ','
          << o.loser_cards << ',' << o.model_seat << '\n';
    std::printf("dumped %zu game outcomes -> %s\n", outcomes.size(), dump_path.c_str());
  }
  return 0;
}
