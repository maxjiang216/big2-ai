// az_selfplay: self-play data generation for the az_search nets.
//
// Modes:
//   * gen0 bootstrap (no --player-model): self-play with policy/behavior targets
//     = the one-hot played move. Either uniform-random (default) or, with
//     --teacher NAME (e.g. typed_search, the strongest non-az player), driven by
//     a registry policy so the NN imitates a strong teacher from the start.
//   * gen N>=1 (--player-model / --opp-model): NN-guided MCTS self-play. NN leaf
//     evaluations are BATCHED ACROSS MANY CONCURRENT GAMES (the throughput win):
//     every game advances its current search by simulations, leaf requests are
//     pooled into player-net / opp-net batches, flushed, and distributed back.
//
// Samples are recorded per head only where the search would actually query that
// head's NN (classify_position / PosType) — we don't train a head on positions
// it resolves trivially:
//   * PLAYER policy (visit distribution): real decisions only.
//   * PLAYER value: real decisions AND tablebase (opp-1) positions (the move is
//     pinned but the leaf VALUE is still used); recorded as a value-only player
//     sample (empty visit list -> no policy gradient). Excluded at insta-win /
//     forced-win (value trivially 1) and forced-pass (node fused).
//   * OPP behavior + value: every position except the mover's forced-pass or
//     hand-emptying insta-win (the observer can't tell forced-win / tablebase
//     positions from public info, so it must model them).
// Both value targets are backfilled from the game winner. At real decisions the
// move played is sampled proportional to the root visit counts (exploration).
//
// Output: two Parquet files matching the nn/dataset.py schemas
// (az_player_genN.parquet, az_opp_genN.parquet).

#include "az_search/az_search.h"
#include "az_search/features.h"
#include "az_search/nn_eval.h"

#include "game.h"
#include "game_record.h"
#include "game_simulator.h"
#include "move.h"
#include "player_factory_registry.h"
#include "tablebase_opp1.h"
#include "util.h"

#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <parquet/arrow/writer.h>
#include <torch/cuda.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace az_search;

// ---------------------------------------------------------------------------
// Tablebase / forced detection (mirror of eval_helpers / nn_game_runner).
// ---------------------------------------------------------------------------
static int tablebase_move_id(const Game &game, int cp,
                             const std::vector<int> &legal) {
  auto hand = game.player_hand(cp);
  int hand_size = game.get_player_hand_size(cp);
  int opp_size = game.get_player_hand_size(1 - cp);
  for (int mid : legal)
    if (mid != kPASS && MOVE_TO_CARDS[mid][13] == hand_size) return mid;
  if (game.last_move().combination != Move::Combination::kPass) return -1;
  auto discard = game.discard_pile();
  if (opp_size == 1) {
    int best_rank = -1;
    bool all_singles = true;
    for (int mid : legal) {
      if (mid == kPASS) continue;
      Move m(mid);
      if (m.combination != Move::Combination::kSingle) { all_singles = false; break; }
      if (m.rank > best_rank) best_rank = m.rank;
    }
    if (all_singles && best_rank != -1)
      for (int mid : legal)
        if (mid != kPASS && Move(mid).rank == best_rank) return mid;
    Opp1Result o = lookup_opp1(hand);
    if (o.first_move_id != 0)
      for (int mid : legal)
        if (mid == o.first_move_id) return mid;
    if (auto def = opp1_default_strategy_move(hand)) return *def;
  }
  if (auto seq = find_forced_win(hand, discard, opp_size)) return (*seq)[0];
  return -1;
}

// Position classification for the mover, used to decide which heads (if any) get
// a training sample — the rule is "will the search ever query that head's NN on
// this kind of position?" (don't train heads on positions the search resolves
// trivially). See advance_to_decision below for the per-type recording.
enum PosType {
  POS_REAL,        // >=2 genuine choices: player policy + value + opp recorded
  POS_TABLEBASE,   // opp-1 line: move pinned but VALUE used -> value + opp (no policy)
  POS_FORCED_WIN,  // proven win: player value=1 trivial -> opp only
  POS_INSTA_WIN,   // hand-emptying move: terminal before NN -> nothing
  POS_FORCED_PASS, // must pass: node fused in search -> nothing
};

// Classify the mover's position; for non-real positions also yield the forced
// move the mover will play. Mirrors expand_player's short-circuits.
static PosType classify_position(const Game &game, const std::vector<int> &legal,
                                 int &forced_move) {
  const int mover = game.current_player();
  const int our_size = game.get_player_hand_size(mover);
  for (int m : legal)
    if (m != kPASS && MOVE_TO_CARDS[m][13] == our_size) {
      forced_move = m;
      return POS_INSTA_WIN;
    }
  if (legal.size() == 1 && legal[0] == kPASS) {
    forced_move = kPASS;
    return POS_FORCED_PASS;
  }
  const bool lead = (game.last_move().combination == Move::Combination::kPass);
  if (lead) {
    auto hand = game.player_hand(mover);
    auto discard = game.discard_pile();
    const int opp_size = game.get_player_hand_size(1 - mover);
    if (auto seq = find_forced_win(hand, discard, opp_size)) {
      forced_move = (*seq)[0];
      return POS_FORCED_WIN;
    }
    if (opp_size == 1) {
      int tb = tablebase_move_id(game, mover, legal);  // opp-1 line (move known)
      if (tb >= 0) { forced_move = tb; return POS_TABLEBASE; }
    }
  }
  // A size-1 legal set is always either {pass} (forced pass) or, on the lead
  // with 1 card, the hand-emptying single (insta-win) — both caught above; any
  // genuine ">=1 move" position with real cards has >=2 single plays.
  return POS_REAL;
}

// ---------------------------------------------------------------------------
// Samples (values backfilled at game end).
// ---------------------------------------------------------------------------
struct PlayerSample {
  int game_id, turn_idx;
  std::array<int, 13> hand, opp_max, trick;
  int opp_size, our_size;
  std::vector<int> legal;
  std::vector<int> visit_moves, visit_counts;
  int mover;
  float value = 0.0f;
};

struct OppSample {
  int game_id, turn_idx;
  std::array<int, 13> hand, opp_max, trick;  // hand = observer's exact hand
  int opp_size, our_size;
  std::vector<int> legal;
  int move_id;
  int observer;
  float value = 0.0f;
};

static SearchState mover_state(const Game &game, int mover) {
  SearchState s;
  s.our_hand = game.player_hand(mover);
  s.opp_size = game.get_player_hand_size(1 - mover);
  s.discard = game.discard_pile();
  s.last_move = encodeMove(game.last_move());
  s.side = kUs;
  return s;
}

// Player-net sample (the mover's view). `visits` is the MCTS visit distribution
// (the policy target); pass an EMPTY list for a value-only sample (tablebase
// positions) — the zero policy target contributes no policy gradient, so the
// value head trains while the policy head is untouched.
static void record_player_sample(const Game &game, int mover,
                                 const std::vector<std::pair<int, long>> &visits,
                                 int game_id, int turn_idx,
                                 std::vector<PlayerSample> &psamples) {
  const int obs = 1 - mover;
  auto hand = game.player_hand(mover);
  auto discard = game.discard_pile();
  const int last = encodeMove(game.last_move());
  PlayerSample ps;
  ps.game_id = game_id; ps.turn_idx = turn_idx;
  ps.hand = hand;
  ps.opp_max = opp_max_counts(hand, discard);
  ps.trick = trick_counts(last);
  ps.opp_size = game.get_player_hand_size(obs);
  ps.our_size = game.get_player_hand_size(mover);
  ps.legal = compute_legal_moves(hand, Move(last));
  for (auto &[m, n] : visits) { ps.visit_moves.push_back(m); ps.visit_counts.push_back((int)n); }
  ps.mover = mover;
  psamples.push_back(std::move(ps));
}

// Opponent-net sample (the observer's public view of the mover's actual move).
static void record_opp_sample(const Game &game, int mover, int chosen_move,
                              int game_id, int turn_idx,
                              std::vector<OppSample> &osamples) {
  const int obs = 1 - mover;
  auto ohand = game.player_hand(obs);
  auto discard = game.discard_pile();
  const int last = encodeMove(game.last_move());
  OppSample os;
  os.game_id = game_id; os.turn_idx = turn_idx;
  os.hand = ohand;                               // observer's exact hand (NN input)
  os.opp_max = opp_max_counts(ohand, discard);  // observer's upper bound on the mover
  os.trick = trick_counts(last);
  os.opp_size = game.get_player_hand_size(mover);   // mover hand size
  os.our_size = game.get_player_hand_size(obs);     // observer hand size
  os.legal = compute_possible_moves(ohand, discard, os.opp_size, Move(last),
                                    /*exclude_bombs=*/false);
  os.move_id = chosen_move;
  os.observer = obs;
  osamples.push_back(std::move(os));
}

// Real-decision recording: player policy+value sample (with visits) + opp sample.
static void record_decision(Game &game, int mover, int chosen_move,
                            const std::vector<std::pair<int, long>> &visits,
                            int game_id, int turn_idx,
                            std::vector<PlayerSample> &psamples,
                            std::vector<OppSample> &osamples) {
  record_player_sample(game, mover, visits, game_id, turn_idx, psamples);
  record_opp_sample(game, mover, chosen_move, game_id, turn_idx, osamples);
}

static int sample_from_visits(const std::vector<std::pair<int, long>> &visits,
                              std::mt19937 &rng) {
  long total = 0;
  for (auto &[m, n] : visits) total += n;
  if (total <= 0) return visits.empty() ? kPASS : visits.front().first;
  long r = std::uniform_int_distribution<long>(1, total)(rng);
  long acc = 0;
  for (auto &[m, n] : visits) { acc += n; if (r <= acc) return m; }
  return visits.back().first;
}

static void backfill_values(int winner, std::size_t p_from, std::size_t o_from,
                            std::vector<PlayerSample> &ps, std::vector<OppSample> &os) {
  for (std::size_t i = p_from; i < ps.size(); ++i)
    ps[i].value = (ps[i].mover == winner) ? 1.0f : 0.0f;
  for (std::size_t i = o_from; i < os.size(); ++i)
    os[i].value = (os[i].observer == winner) ? 1.0f : 0.0f;
}

// Advance through forced positions to the next real decision, recording exactly
// the heads the search would query at each forced position (classify_position):
// value-only player + opp at tablebase, opp only at forced-win, nothing at
// insta-win / forced-pass. Returns the legal moves at the real
// decision, or {} if the game ended first.
static std::vector<int> advance_to_decision(Game &game, int game_id, int &turn,
                                            std::vector<PlayerSample> &ps,
                                            std::vector<OppSample> &os) {
  while (!game.is_over()) {
    auto legal = game.get_legal_moves();
    int fm = -1;
    const PosType t = classify_position(game, legal, fm);
    if (t == POS_REAL) return legal;
    const int mover = game.current_player();
    if (t == POS_TABLEBASE) {
      record_player_sample(game, mover, {}, game_id, turn, ps);  // value-only
      record_opp_sample(game, mover, fm, game_id, turn, os);
      ++turn;
    } else if (t == POS_FORCED_WIN) {
      record_opp_sample(game, mover, fm, game_id, turn, os);  // observer learns the loss
      ++turn;
    }
    // POS_INSTA_WIN / POS_FORCED_PASS: the search never queries an NN head here.
    game.apply_move(fm);
  }
  return {};
}

// ---------------------------------------------------------------------------
// gen0: uniform-random self-play (no search).
// ---------------------------------------------------------------------------
static void run_random_selfplay(int total_games, unsigned seed,
                                std::vector<PlayerSample> &ps,
                                std::vector<OppSample> &os) {
  std::mt19937 rng(seed);
  for (int g = 0; g < total_games; ++g) {
    Game game;
    game.shuffle_deal(rng);
    const std::size_t pf = ps.size(), of = os.size();
    int turn = 0;
    while (true) {
      auto legal = advance_to_decision(game, g, turn, ps, os);
      if (legal.empty()) break;  // game ended
      const int mover = game.current_player();
      const int m = legal[std::uniform_int_distribution<int>(0, (int)legal.size() - 1)(rng)];
      record_decision(game, mover, m, {{m, 1}}, g, turn++, ps, os);
      game.apply_move(m);
    }
    if (game.is_over()) backfill_values(game.get_winner(), pf, of, ps, os);
  }
}

// ---------------------------------------------------------------------------
// gen0/gen1 teacher bootstrap: self-play by a registry policy (e.g. typed_search,
// the strongest non-az player). Each game is played by two teacher instances via
// GameSimulator; az samples are then extracted from the GameRecord turns under
// the same per-head filter as the search path, with the policy / behavior target
// being the move the teacher actually played (one-hot imitation). Games start
// from full deals (no start-state mixing). The teacher's own tablebase moves are
// recorded honestly per classify_position (value-only player at tablebase, etc.).
// ---------------------------------------------------------------------------
static void run_classic_selfplay(const std::string &policy, double param,
                                 int total_games, unsigned seed,
                                 std::vector<PlayerSample> &ps,
                                 std::vector<OppSample> &os) {
  auto factory = make_player_factory(policy, param, seed);
  std::mt19937 rng(seed);
  for (int g = 0; g < total_games; ++g) {
    GameSimulator sim(factory->create_player(), factory->create_player(), rng);
    GameRecord rec = sim.run();
    const std::size_t pf = ps.size(), of = os.size();
    int turn = 0;
    for (const TurnRecord &tr : rec.turns()) {
      const Game &game = tr.game;        // full pre-move state at this turn
      const int mover = tr.current_player;
      auto legal = game.get_legal_moves();
      int fm = -1;
      const PosType t = classify_position(game, legal, fm);
      const int played = encodeMove(tr.move);  // what the teacher actually played
      if (t == POS_REAL) {
        record_player_sample(game, mover, {{played, 1}}, g, turn, ps);  // one-hot
        record_opp_sample(game, mover, played, g, turn, os);
        ++turn;
      } else if (t == POS_TABLEBASE) {
        record_player_sample(game, mover, {}, g, turn, ps);  // value-only
        record_opp_sample(game, mover, played, g, turn, os);
        ++turn;
      } else if (t == POS_FORCED_WIN) {
        record_opp_sample(game, mover, played, g, turn, os);
        ++turn;
      }
      // POS_INSTA_WIN / POS_FORCED_PASS: no NN head queried -> no sample.
    }
    backfill_values(rec.game().get_winner(), pf, of, ps, os);
  }
}

// ---------------------------------------------------------------------------
// gen N>=1: batched NN-guided MCTS self-play, async producer/consumer.
//
// W CPU worker threads each own a DISJOINT block of slots and run all the search
// (select_leaf / apply_eval / advance_root) for them — per-slot state stays
// single-threaded. A single inference thread is the SOLE caller of nn.eval_*
// (so the shared TorchScript module is touched by exactly one thread), batching
// leaf requests across every worker's slots. A worker pumps each of its slots
// until it needs an NN eval, enqueues the request, and parks that slot; when all
// its slots are parked it sleeps until results arrive. The inference thread
// flushes when the queue reaches batch_target OR every active slot is parked
// (tail drain — avoids deadlock). The search DAG is GC'd on re-root (see
// az_search), so memory stays bounded. Each slot keeps ONE persistent tree per
// seat, re-rooted across turns.
//
// Determinism: each game's RNG and per-(game,seat) tree seeds derive only from
// (seed, game_id), but content is NOT bitwise-reproducible across multi-slot
// runs — batched NN inference is batch-composition-sensitive (different GEMM
// kernels/accumulation order per batch size), and async scheduling varies the
// composition, which can flip a tie-broken move. --slots 1 (always batch-of-1)
// is fully reproducible. The variation is benign for data generation.
// ---------------------------------------------------------------------------
struct Slot {
  int game_id = -1;
  Game game;
  std::mt19937 rng;                 // per-game RNG (reseeded at game start)
  std::unique_ptr<Search> tree[2];  // persistent per-seat search trees
  int mover = 0;
  int turn = 0;
  int sims_done = 0;
  std::size_t p_from = 0, o_from = 0;  // sample range start for this game
  bool active = false;
  bool searching = false;  // mid-search (true) vs needs a new decision setup
  bool waiting = false;    // has an outstanding NN eval request
  LeafRequest pending;     // the pumped leaf awaiting eval
  bool pending_is_player = false;
  std::atomic<bool> result_ready{false};  // inference -> worker handoff flag
  PlayerEval peval;        // inference writes (player leaf)
  OppEval oeval;           // inference writes (opp leaf)
};

// One queued leaf-eval request (features copied by value; the inference thread
// never touches a slot's Search tree).
struct EvalReq {
  int slot;
  bool is_player;
  PlayerFeatures pf;
  OppFeatures of;
};

static void run_search_selfplay(NNEvaluator &nn, int total_games, int slots_n,
                                int sims, unsigned seed,
                                std::vector<PlayerSample> &ps,
                                std::vector<OppSample> &os) {
  const int hw = std::max(1, (int)std::thread::hardware_concurrency() - 1);
  const int n_workers = std::max(1, std::min(hw, slots_n));
  const int per = (slots_n + n_workers - 1) / n_workers;
  const int batch_target = std::max(1, std::min(slots_n, 512));

  std::vector<std::pair<int, int>> ranges;  // disjoint slot ranges per worker
  for (int w = 0; w < n_workers; ++w) {
    int b = w * per, e = std::min(slots_n, (w + 1) * per);
    if (b < e) ranges.emplace_back(b, e);
  }
  const int n_threads = (int)ranges.size();

  std::vector<Slot> slots(slots_n);
  std::vector<std::vector<PlayerSample>> wps(n_threads);  // per-worker samples
  std::vector<std::vector<OppSample>> wos(n_threads);

  std::mutex mtx;  // guards queue_; co-checked with active_slots / active_workers
  std::condition_variable cv_infer;
  std::deque<EvalReq> queue_;
  std::atomic<int> active_slots{0};
  std::atomic<int> active_workers{n_threads};
  std::atomic<long> started{0};

  std::mutex rmtx;  // worker wakeup
  std::condition_variable rcv;

  auto worker_fn = [&](int wid, int rb, int re) {
    std::vector<PlayerSample> &lps = wps[wid];
    std::vector<OppSample> &los = wos[wid];

    auto start_game = [&](Slot &s) -> bool {
      long gid = started.fetch_add(1, std::memory_order_relaxed);
      if (gid >= total_games) return false;
      s.game_id = (int)gid;
      s.turn = 0;
      s.rng.seed((unsigned)(seed ^ (0x9E3779B9u * (unsigned)(gid + 1))));
      s.game.shuffle_deal(s.rng);
      s.p_from = lps.size();
      s.o_from = los.size();
      s.active = true;
      s.searching = false;
      s.waiting = false;
      active_slots.fetch_add(1, std::memory_order_relaxed);
      return true;
    };

    auto setup_decision = [&](Slot &s) -> bool {
      auto legal = advance_to_decision(s.game, s.game_id, s.turn, lps, los);
      if (legal.empty()) return false;
      s.mover = s.game.current_player();
      SearchState st = mover_state(s.game, s.mover);
      auto &tr = s.tree[s.mover];
      unsigned tseed =
          (unsigned)(seed ^ (0x9E3779B9u * (unsigned)(s.game_id * 2 + s.mover)));
      if (!tr)
        tr = std::make_unique<Search>(st, SearchConfig{1.5f, sims, tseed, true});
      else
        tr->advance_root(st);
      s.sims_done = 0;
      s.searching = true;
      return true;
    };

    auto complete_decision = [&](Slot &s) {
      Search &tr = *s.tree[s.mover];
      tr.finalize();
      auto visits = tr.root_visits();
      int m = sample_from_visits(visits, s.rng);
      record_decision(s.game, s.mover, m, visits, s.game_id, s.turn++, lps, los);
      s.game.apply_move(m);  // trees persist; next setup_decision re-roots
      s.searching = false;
    };

    auto finalize_game = [&](Slot &s) {
      if (s.game.is_over())
        backfill_values(s.game.get_winner(), s.p_from, s.o_from, lps, los);
      s.active = false;
      s.searching = false;
      s.tree[0].reset();
      s.tree[1].reset();
      { std::lock_guard<std::mutex> lk(mtx); active_slots.fetch_sub(1, std::memory_order_relaxed); }
      cv_infer.notify_one();  // active_slots dropped -> may enable a tail drain
    };

    auto enqueue = [&](Slot &s, int slot_idx) {
      {
        std::lock_guard<std::mutex> lk(mtx);
        EvalReq r;
        r.slot = slot_idx;
        r.is_player = s.pending.is_player;
        if (r.is_player) r.pf = s.pending.pfeat; else r.of = s.pending.ofeat;
        queue_.push_back(std::move(r));
      }
      cv_infer.notify_one();
    };

    // Advance one slot as far as it can go; returns true while it is still
    // active (parked on an eval), false once its work is fully done.
    auto advance = [&](Slot &s, int slot_idx) -> bool {
      for (;;) {
        if (!s.active) {
          if (!start_game(s)) return false;  // no games left -> done
          continue;
        }
        if (s.waiting) {
          if (!s.result_ready.load(std::memory_order_acquire)) return true;
          if (s.pending_is_player) s.tree[s.mover]->apply_eval(s.peval);
          else                     s.tree[s.mover]->apply_eval(s.oeval);
          s.result_ready.store(false, std::memory_order_relaxed);
          s.waiting = false;
          ++s.sims_done;
          continue;
        }
        if (!s.searching) {
          if (!setup_decision(s)) { finalize_game(s); continue; }
        }
        bool pushed = false;
        while (s.sims_done < sims) {
          LeafRequest req = s.tree[s.mover]->select_leaf();
          if (!req.needs_eval) { ++s.sims_done; continue; }
          s.pending = req;
          s.pending_is_player = req.is_player;
          s.waiting = true;
          enqueue(s, slot_idx);
          pushed = true;
          break;
        }
        if (pushed) continue;  // now parked; next iteration returns true
        complete_decision(s);
        if (s.game.is_over()) finalize_game(s);
        continue;
      }
    };

    for (;;) {
      bool any_waiting = false;
      for (int i = rb; i < re; ++i)
        if (advance(slots[i], i)) any_waiting = true;
      if (!any_waiting) break;  // all of this worker's slots are done
      std::unique_lock<std::mutex> lk(rmtx);
      rcv.wait(lk, [&] {
        for (int i = rb; i < re; ++i) {
          Slot &s = slots[i];
          if (s.waiting && s.result_ready.load(std::memory_order_acquire)) return true;
        }
        return false;
      });
    }

    { std::lock_guard<std::mutex> lk(mtx); active_workers.fetch_sub(1, std::memory_order_relaxed); }
    cv_infer.notify_one();  // let the inference thread exit once all workers done
  };

  auto infer_fn = [&]() {
    std::vector<PlayerFeatures> pf; std::vector<int> pslot;
    std::vector<OppFeatures> of; std::vector<int> oslot;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mtx);
        cv_infer.wait(lk, [&] {
          if (active_workers.load() == 0) return true;
          if (queue_.empty()) return false;
          return (int)queue_.size() >= batch_target ||
                 (long)queue_.size() >= (long)active_slots.load();
        });
        if (queue_.empty()) {
          if (active_workers.load() == 0) break;
          continue;
        }
        for (auto &r : queue_) {
          if (r.is_player) { pf.push_back(r.pf); pslot.push_back(r.slot); }
          else { of.push_back(r.of); oslot.push_back(r.slot); }
        }
        queue_.clear();
      }
      if (!pf.empty()) {
        auto res = nn.eval_players(pf);
        for (std::size_t i = 0; i < pslot.size(); ++i) {
          Slot &s = slots[pslot[i]];
          s.peval = res[i];
          s.result_ready.store(true, std::memory_order_release);
        }
      }
      if (!of.empty()) {
        auto res = nn.eval_opps(of);
        for (std::size_t i = 0; i < oslot.size(); ++i) {
          Slot &s = slots[oslot[i]];
          s.oeval = res[i];
          s.result_ready.store(true, std::memory_order_release);
        }
      }
      pf.clear(); pslot.clear(); of.clear(); oslot.clear();
      { std::lock_guard<std::mutex> lk(rmtx); }  // serialize with worker waits
      rcv.notify_all();
    }
  };

  std::thread infer(infer_fn);
  std::vector<std::thread> workers;
  for (int w = 0; w < n_threads; ++w)
    workers.emplace_back(worker_fn, w, ranges[w].first, ranges[w].second);
  for (auto &t : workers) t.join();
  infer.join();

  for (int w = 0; w < n_threads; ++w) {
    ps.insert(ps.end(), wps[w].begin(), wps[w].end());
    os.insert(os.end(), wos[w].begin(), wos[w].end());
  }
}

// ---------------------------------------------------------------------------
// Parquet writers (schemas mirror nn/dataset.py).
// ---------------------------------------------------------------------------
static std::shared_ptr<arrow::Array> finish_i32(arrow::Int32Builder &b) {
  std::shared_ptr<arrow::Array> a; b.Finish(&a); return a;
}
static std::shared_ptr<arrow::Array> finish_f32(arrow::FloatBuilder &b) {
  std::shared_ptr<arrow::Array> a; b.Finish(&a); return a;
}

static void write_table(const std::string &path,
                        std::vector<std::shared_ptr<arrow::Field>> fields,
                        std::vector<std::shared_ptr<arrow::Array>> arrays) {
  auto table = arrow::Table::Make(arrow::schema(std::move(fields)), std::move(arrays));
  auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  parquet::WriterProperties::Builder pb;
  pb.compression(parquet::Compression::SNAPPY);
  parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 1 << 20, pb.build());
}

static void write_player_parquet(const std::string &path, const std::vector<PlayerSample> &v) {
  auto pool = arrow::default_memory_pool();
  arrow::Int32Builder game_id, turn_idx, opp_size, our_size;
  arrow::Int32Builder hand[13], opp_max[13], trick[13];
  arrow::FloatBuilder value;
  arrow::ListBuilder legal(pool, std::make_shared<arrow::Int32Builder>(pool));
  arrow::ListBuilder vmoves(pool, std::make_shared<arrow::Int32Builder>(pool));
  arrow::ListBuilder vcounts(pool, std::make_shared<arrow::Int32Builder>(pool));
  auto *legal_v = static_cast<arrow::Int32Builder *>(legal.value_builder());
  auto *vmoves_v = static_cast<arrow::Int32Builder *>(vmoves.value_builder());
  auto *vcounts_v = static_cast<arrow::Int32Builder *>(vcounts.value_builder());

  for (const auto &s : v) {
    game_id.Append(s.game_id); turn_idx.Append(s.turn_idx);
    opp_size.Append(s.opp_size); our_size.Append(s.our_size);
    value.Append(s.value);
    for (int r = 0; r < 13; ++r) { hand[r].Append(s.hand[r]); opp_max[r].Append(s.opp_max[r]); trick[r].Append(s.trick[r]); }
    legal.Append(); legal_v->AppendValues(s.legal.data(), (int64_t)s.legal.size());
    vmoves.Append(); vmoves_v->AppendValues(s.visit_moves.data(), (int64_t)s.visit_moves.size());
    vcounts.Append(); vcounts_v->AppendValues(s.visit_counts.data(), (int64_t)s.visit_counts.size());
  }

  std::vector<std::shared_ptr<arrow::Field>> f;
  std::vector<std::shared_ptr<arrow::Array>> a;
  auto push = [&](std::shared_ptr<arrow::Field> fld, std::shared_ptr<arrow::Array> arr) {
    f.push_back(std::move(fld)); a.push_back(std::move(arr));
  };
  push(arrow::field("game_id", arrow::int32()), finish_i32(game_id));
  push(arrow::field("turn_idx", arrow::int32()), finish_i32(turn_idx));
  for (int r = 0; r < 13; ++r) push(arrow::field("hand_" + std::to_string(r), arrow::int32()), finish_i32(hand[r]));
  for (int r = 0; r < 13; ++r) push(arrow::field("opp_max_" + std::to_string(r), arrow::int32()), finish_i32(opp_max[r]));
  for (int r = 0; r < 13; ++r) push(arrow::field("trick_" + std::to_string(r), arrow::int32()), finish_i32(trick[r]));
  push(arrow::field("opp_size", arrow::int32()), finish_i32(opp_size));
  push(arrow::field("our_size", arrow::int32()), finish_i32(our_size));
  push(arrow::field("value", arrow::float32()), finish_f32(value));
  std::shared_ptr<arrow::Array> la, vma, vca;
  legal.Finish(&la); vmoves.Finish(&vma); vcounts.Finish(&vca);
  push(arrow::field("legal_moves", arrow::list(arrow::int32())), la);
  push(arrow::field("visit_moves", arrow::list(arrow::int32())), vma);
  push(arrow::field("visit_counts", arrow::list(arrow::int32())), vca);
  write_table(path, std::move(f), std::move(a));
}

static void write_opp_parquet(const std::string &path, const std::vector<OppSample> &v) {
  auto pool = arrow::default_memory_pool();
  arrow::Int32Builder game_id, turn_idx, opp_size, our_size, move_id;
  arrow::Int32Builder hand[13], opp_max[13], trick[13];
  arrow::FloatBuilder value;
  arrow::ListBuilder legal(pool, std::make_shared<arrow::Int32Builder>(pool));
  auto *legal_v = static_cast<arrow::Int32Builder *>(legal.value_builder());

  for (const auto &s : v) {
    game_id.Append(s.game_id); turn_idx.Append(s.turn_idx);
    opp_size.Append(s.opp_size); our_size.Append(s.our_size);
    move_id.Append(s.move_id); value.Append(s.value);
    for (int r = 0; r < 13; ++r) { hand[r].Append(s.hand[r]); opp_max[r].Append(s.opp_max[r]); trick[r].Append(s.trick[r]); }
    legal.Append(); legal_v->AppendValues(s.legal.data(), (int64_t)s.legal.size());
  }

  std::vector<std::shared_ptr<arrow::Field>> f;
  std::vector<std::shared_ptr<arrow::Array>> a;
  auto push = [&](std::shared_ptr<arrow::Field> fld, std::shared_ptr<arrow::Array> arr) {
    f.push_back(std::move(fld)); a.push_back(std::move(arr));
  };
  push(arrow::field("game_id", arrow::int32()), finish_i32(game_id));
  push(arrow::field("turn_idx", arrow::int32()), finish_i32(turn_idx));
  for (int r = 0; r < 13; ++r) push(arrow::field("hand_" + std::to_string(r), arrow::int32()), finish_i32(hand[r]));
  for (int r = 0; r < 13; ++r) push(arrow::field("opp_max_" + std::to_string(r), arrow::int32()), finish_i32(opp_max[r]));
  for (int r = 0; r < 13; ++r) push(arrow::field("trick_" + std::to_string(r), arrow::int32()), finish_i32(trick[r]));
  push(arrow::field("opp_size", arrow::int32()), finish_i32(opp_size));
  push(arrow::field("our_size", arrow::int32()), finish_i32(our_size));
  push(arrow::field("value", arrow::float32()), finish_f32(value));
  std::shared_ptr<arrow::Array> la;
  legal.Finish(&la);
  push(arrow::field("legal_moves", arrow::list(arrow::int32())), la);
  push(arrow::field("move_id", arrow::int32()), finish_i32(move_id));
  write_table(path, std::move(f), std::move(a));
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
static const char *arg(int argc, char **argv, const char *key, const char *def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return def;
}

int main(int argc, char **argv) {
  const std::string player_out = arg(argc, argv, "--player-out", "data/az_player.parquet");
  const std::string opp_out = arg(argc, argv, "--opp-out", "data/az_opp.parquet");
  const std::string player_model = arg(argc, argv, "--player-model", "");
  const std::string opp_model = arg(argc, argv, "--opp-model", "");
  const int games = std::atoi(arg(argc, argv, "--games", "1000"));
  const int sims = std::atoi(arg(argc, argv, "--sims", "100"));
  const int slots = std::atoi(arg(argc, argv, "--slots", "256"));
  // gen0 bootstrap teacher: a registry policy (e.g. typed_search) plays self-play
  // and the NN imitates it. Empty -> uniform-random gen0 (only relevant when no
  // --player-model/--opp-model is given).
  const std::string teacher = arg(argc, argv, "--teacher", "");
  const double teacher_param = std::atof(arg(argc, argv, "--teacher-param", "0"));
  const unsigned seed = (unsigned)std::strtoul(arg(argc, argv, "--seed", "0"), nullptr, 10);
  // Inference device: the NN forward dominates self-play wall time, so CUDA is a
  // big win (the batched-across-slots forward is exactly the GPU's strength).
  std::string device_str = arg(argc, argv, "--device", "cpu");
  if (device_str == "cuda" && !torch::cuda::is_available()) {
    std::printf("[az_selfplay] CUDA requested but unavailable; falling back to CPU\n");
    device_str = "cpu";
  }
  torch::Device device(device_str == "cuda" ? torch::kCUDA : torch::kCPU);

  std::vector<PlayerSample> ps;
  std::vector<OppSample> os;

  const bool gen0 = player_model.empty() || opp_model.empty();
  if (gen0 && !teacher.empty()) {
    std::printf("[az_selfplay] gen0 teacher self-play (%s, param=%.3g): %d games\n",
                teacher.c_str(), teacher_param, games);
    run_classic_selfplay(teacher, teacher_param, games, seed, ps, os);
  } else if (gen0) {
    std::printf("[az_selfplay] gen0 random self-play: %d games\n", games);
    run_random_selfplay(games, seed, ps, os);
  } else {
    std::printf("[az_selfplay] NN self-play: %d games, sims=%d, slots=%d, device=%s\n",
                games, sims, slots, device_str.c_str());
    NNEvaluator nn(player_model, opp_model, device);
    run_search_selfplay(nn, games, slots, sims, seed, ps, os);
  }

  std::printf("[az_selfplay] samples: player=%zu opp=%zu -> %s, %s\n",
              ps.size(), os.size(), player_out.c_str(), opp_out.c_str());
  write_player_parquet(player_out, ps);
  write_opp_parquet(opp_out, os);
  return 0;
}
