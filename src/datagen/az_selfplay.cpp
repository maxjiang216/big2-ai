// az_selfplay: self-play data generation for the az_search nets.
//
// Modes:
//   * gen0 bootstrap (no --model): self-play with policy/behavior targets
//     = the one-hot played move. Either uniform-random (default) or, with
//     --teacher NAME (e.g. typed_search, the strongest non-az player), driven by
//     a registry policy so the NN imitates a strong teacher from the start.
//   * gen N>=1 (--model): NN-guided MCTS self-play. NN leaf
//     evaluations are BATCHED ACROSS MANY CONCURRENT GAMES (the throughput win):
//     every game advances its current search by simulations, leaf requests are
//     pooled into player-net / opp-net batches, flushed, and distributed back.
//
// Samples are recorded per head only where the search would actually query that
// head's NN (classify_position / PosType) — we don't train a head on positions
// it resolves trivially:
//   * PLAYER policy (visit distribution): real decisions only.
//   * PLAYER value: real decisions AND opp-1 oracle positions (the move is
//     pinned but the leaf VALUE is still used); recorded as a value-only player
//     sample (empty visit list -> no policy gradient). Excluded at insta-win /
//     forced-win (value resolved without an NN value query) and forced-pass.
//   * OPP behavior + value: every position except the mover's forced-pass or
//     hand-emptying insta-win (the observer can't tell forced-win / opp-1
//     positions from public info, so it must model them).
// Both value targets are backfilled from the game outcome: under the series
// objective the target is V(resulting series state) (series_value_after_win);
// without a series table it is the legacy 1/0 game win/loss. At real decisions
// the move played is sampled proportional to the root visit counts.
//
// Output: three Parquet files matching the nn/dataset_seq.py schemas
// (az_player_genN.parquet, az_opp_genN.parquet, az_games_genN.parquet). The
// games file holds each game's full move-id sequence (the transformer tokens);
// sample rows carry (game_id, hist_idx) so training slices the prefix. The
// hist_idx convention (see nn/model_az_seq.py) counts ALL applied moves
// including forced passes / insta-wins — it is NOT the sample turn_idx.

#include "az_search/az_search.h"
#include "az_search/considered_moves.h"
#include "az_search/features.h"
#include "az_search/nn_eval.h"
#include "nn_encode.h"  // encode_exact / encode_thermo (same as inference -> no drift)

#include "game.h"
#include "game_record.h"
#include "game_simulator.h"
#include "move.h"
#include "player_factory_registry.h"
#include "series.h"
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
// Forced / endgame detection (mirror of az_search's expand_player oracles).
// ---------------------------------------------------------------------------

// Position classification for the mover, used to decide which heads (if any) get
// a training sample — the rule is "will the search ever query that head's NN on
// this kind of position?" (don't train heads on positions the search resolves
// trivially). See advance_to_decision below for the per-type recording.
enum PosType {
  POS_REAL,        // >=2 genuine choices: player policy + value + opp recorded
  POS_OPP1,        // opp-1 series oracle: move pinned but VALUE used -> value + opp
  POS_FORCED_WIN,  // proven win: player value (V of resulting state) -> opp only
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
    // opp-has-1-card series oracle: only when the hand has no straight lead
    // (mirrors az_search::opp1_move). Straight-holding hands fall to POS_REAL.
    if (opp_size == 1 && !hand_has_straight_lead(hand)) {
      forced_move = opp1_series_move(hand);
      return POS_OPP1;
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
  int hist_idx;  // moves applied before this decision (sequence readout index)
  std::array<int, 13> hand, opp_max, trick;
  int opp_size, our_size;
  std::vector<int> legal;
  std::vector<int> visit_moves, visit_counts;
  int mover;
  float value = 0.0f;
};

struct OppSample {
  int game_id, turn_idx;
  int hist_idx;  // moves applied before this decision (sequence readout index)
  std::array<int, 13> hand, opp_max, trick;  // hand = observer's exact hand
  int opp_size, our_size;
  std::vector<int> legal;
  int move_id;
  int observer;
  float value = 0.0f;
};

// One row per finished game: the full applied-move sequence (transformer
// tokens) plus winner / first player and the series context. pts0/pts1 are the
// per-seat series points at game START; loser_cards is the loser's remaining
// count (drives the points scored). Training derives owner-relative series-state
// inputs and joins natural-frequency weights from these.
struct GameRow {
  int game_id;
  int first_player;
  int winner;
  int pts0;
  int pts1;
  int loser_cards;
  std::vector<int> moves;
};

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

// Player-net sample (the mover's view). `visits` is the MCTS visit distribution
// (the policy target); pass an EMPTY list for a value-only sample (tablebase
// positions) — the zero policy target contributes no policy gradient, so the
// value head trains while the policy head is untouched.
static void record_player_sample(const Game &game, int mover,
                                 const std::vector<std::pair<int, long>> &visits,
                                 int game_id, int turn_idx, int hist_idx,
                                 std::vector<PlayerSample> &psamples) {
  const int obs = 1 - mover;
  auto hand = game.player_hand(mover);
  auto discard = game.discard_pile();
  const int last = encodeMove(game.last_move());
  PlayerSample ps;
  ps.game_id = game_id; ps.turn_idx = turn_idx; ps.hist_idx = hist_idx;
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
                              int game_id, int turn_idx, int hist_idx,
                              std::vector<OppSample> &osamples) {
  const int obs = 1 - mover;
  auto ohand = game.player_hand(obs);
  auto discard = game.discard_pile();
  const int last = encodeMove(game.last_move());
  OppSample os;
  os.game_id = game_id; os.turn_idx = turn_idx; os.hist_idx = hist_idx;
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
                            int game_id, int turn_idx, int hist_idx,
                            std::vector<PlayerSample> &psamples,
                            std::vector<OppSample> &osamples) {
  record_player_sample(game, mover, visits, game_id, turn_idx, hist_idx, psamples);
  record_opp_sample(game, mover, chosen_move, game_id, turn_idx, hist_idx, osamples);
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

// Backfill value targets from the game outcome. With a series table the target
// is V(resulting series state) from the WINNER's POV (series_value_after_win);
// without one it is the legacy 1/0 game win/loss. pts is seat-indexed start
// points. Returns the loser's remaining card count (for the games row).
static int backfill_values(const Game &game, const int pts[2],
                           const SeriesTable *series, std::size_t p_from,
                           std::size_t o_from, std::vector<PlayerSample> &ps,
                           std::vector<OppSample> &os) {
  const int winner = game.get_winner();
  const int loser = 1 - winner;
  const int loser_cards = game.get_player_hand_size(loser);
  const float tw =  // P(this game's winner wins the series)
      series ? series_value_after_win(*series, pts[winner], pts[loser], loser_cards)
             : 1.0f;
  for (std::size_t i = p_from; i < ps.size(); ++i)
    ps[i].value = (ps[i].mover == winner) ? tw : 1.0f - tw;
  for (std::size_t i = o_from; i < os.size(); ++i)
    os[i].value = (os[i].observer == winner) ? tw : 1.0f - tw;
  return loser_cards;
}

// Advance through forced positions to the next real decision, recording exactly
// the heads the search would query at each forced position (classify_position):
// value-only player + opp at tablebase, opp only at forced-win, nothing at
// insta-win / forced-pass. Returns the legal moves at the real
// decision, or {} if the game ended first.
static std::vector<int> advance_to_decision(Game &game, int game_id, int &turn,
                                            std::vector<int> &history,
                                            std::vector<PlayerSample> &ps,
                                            std::vector<OppSample> &os) {
  while (!game.is_over()) {
    auto legal = game.get_legal_moves();
    int fm = -1;
    const PosType t = classify_position(game, legal, fm);
    if (t == POS_REAL) return legal;
    const int mover = game.current_player();
    const int hidx = (int)history.size();
    if (t == POS_OPP1) {
      record_player_sample(game, mover, {}, game_id, turn, hidx, ps);  // value-only
      record_opp_sample(game, mover, fm, game_id, turn, hidx, os);
      ++turn;
    } else if (t == POS_FORCED_WIN) {
      record_opp_sample(game, mover, fm, game_id, turn, hidx, os);  // observer learns the loss
      ++turn;
    }
    // POS_INSTA_WIN / POS_FORCED_PASS: the search never queries an NN head here.
    game.apply_move(fm);
    history.push_back(fm);
  }
  return {};
}

// ---------------------------------------------------------------------------
// gen0: uniform-random self-play (no search).
// ---------------------------------------------------------------------------
static void run_random_selfplay(int total_games, unsigned seed,
                                const SeriesTable *series,
                                std::vector<PlayerSample> &ps,
                                std::vector<OppSample> &os,
                                std::vector<GameRow> &gs) {
  std::mt19937 rng(seed);
  const int pts[2] = {0, 0};  // bootstrap: games always start a fresh series
  for (int g = 0; g < total_games; ++g) {
    Game game;
    game.shuffle_deal(rng);
    const std::size_t pf = ps.size(), of = os.size();
    int turn = 0;
    std::vector<int> history;
    const int first_player = game.current_player();
    while (true) {
      auto legal = advance_to_decision(game, g, turn, history, ps, os);
      if (legal.empty()) break;  // game ended
      const int mover = game.current_player();
      const int m = legal[std::uniform_int_distribution<int>(0, (int)legal.size() - 1)(rng)];
      record_decision(game, mover, m, {{m, 1}}, g, turn++, (int)history.size(), ps, os);
      game.apply_move(m);
      history.push_back(m);
    }
    if (game.is_over()) {
      const int lc = backfill_values(game, pts, series, pf, of, ps, os);
      gs.push_back({g, first_player, game.get_winner(), pts[0], pts[1], lc,
                    std::move(history)});
    }
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
                                 const SeriesTable *series,
                                 std::vector<PlayerSample> &ps,
                                 std::vector<OppSample> &os,
                                 std::vector<GameRow> &gs) {
  auto factory = make_player_factory(policy, param, seed);
  const int pts[2] = {0, 0};  // teacher bootstrap: fresh series each game
  std::mt19937 rng(seed);
  for (int g = 0; g < total_games; ++g) {
    GameSimulator sim(factory->create_player(), factory->create_player(), rng);
    GameRecord rec = sim.run();
    const std::size_t pf = ps.size(), of = os.size();
    int turn = 0;
    // rec.turns() records EVERY applied move (forced ones included), so the
    // turn list index IS hist_idx and its move stream IS the token sequence.
    std::vector<int> history;
    history.reserve(rec.turns().size());
    for (const TurnRecord &tr : rec.turns()) {
      const Game &game = tr.game;        // full pre-move state at this turn
      const int mover = tr.current_player;
      auto legal = game.get_legal_moves();
      int fm = -1;
      const PosType t = classify_position(game, legal, fm);
      const int played = encodeMove(tr.move);  // what the teacher actually played
      const int hidx = (int)history.size();
      if (t == POS_REAL) {
        record_player_sample(game, mover, {{played, 1}}, g, turn, hidx, ps);  // one-hot
        record_opp_sample(game, mover, played, g, turn, hidx, os);
        ++turn;
      } else if (t == POS_OPP1) {
        record_player_sample(game, mover, {}, g, turn, hidx, ps);  // value-only
        record_opp_sample(game, mover, played, g, turn, hidx, os);
        ++turn;
      } else if (t == POS_FORCED_WIN) {
        record_opp_sample(game, mover, played, g, turn, hidx, os);
        ++turn;
      }
      // POS_INSTA_WIN / POS_FORCED_PASS: no NN head queried -> no sample.
      history.push_back(played);
    }
    const int lc = backfill_values(rec.game(), pts, series, pf, of, ps, os);
    const int first_player = rec.turns().empty() ? 0 : rec.turns().front().current_player;
    gs.push_back({g, first_player, rec.game().get_winner(), pts[0], pts[1], lc,
                  std::move(history)});
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
// Per-game series start-state sampler. With probability natural_frac a start
// state is drawn from the chain's natural visit frequencies (matches the
// deployment distribution); otherwise uniformly over reachable states (broad
// coverage, incl. rare late-series states). Reachable = {(0,0)} ∪ {1<=a<=49,
// 0<=b<=49} in leader perspective (a = leader points). Inactive (always (0,0))
// when no series table is loaded — the legacy single-game objective.
struct StartStateSampler {
  std::vector<std::pair<int, int>> states;  // reachable (a, b), leader-perspective
  std::vector<double> nat_cdf;              // cumulative natural_freq over `states`
  double nat_total = 0.0;
  float natural_frac = 0.5f;
  bool active = false;

  void build(const SeriesTable *series, float frac) {
    natural_frac = frac;
    active = (series != nullptr);
    if (!active) return;
    states.emplace_back(0, 0);
    for (int a = 1; a < kSeriesTarget; ++a)
      for (int b = 0; b < kSeriesTarget; ++b)
        states.emplace_back(a, b);
    nat_cdf.reserve(states.size());
    for (auto [a, b] : states) {
      nat_total += series->natural[a][b];
      nat_cdf.push_back(nat_total);
    }
  }

  // Returns leader-perspective (a, b). (0,0) when inactive.
  std::pair<int, int> sample(std::mt19937 &rng) const {
    if (!active) return {0, 0};
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (nat_total > 0.0 && u01(rng) < natural_frac) {
      double r = std::uniform_real_distribution<double>(0.0, nat_total)(rng);
      auto it = std::lower_bound(nat_cdf.begin(), nat_cdf.end(), r);
      std::size_t idx = std::min<std::size_t>(it - nat_cdf.begin(), states.size() - 1);
      return states[idx];
    }
    std::size_t idx =
        std::uniform_int_distribution<std::size_t>(0, states.size() - 1)(rng);
    return states[idx];
  }
};

struct Slot {
  int game_id = -1;
  Game game;
  int pts[2] = {0, 0};              // seat-indexed series points at game start
  std::mt19937 rng;                 // per-game RNG (reseeded at game start)
  std::unique_ptr<Search> tree[2];  // persistent per-seat search trees
  std::vector<int> history;         // every applied move (token sequence)
  int first_player = 0;
  int mover = 0;
  int turn = 0;
  int sims_done = 0;
  std::size_t p_from = 0, o_from = 0;  // sample range start for this game
  bool active = false;
  bool searching = false;  // mid-search (true) vs needs a new decision setup
  bool waiting = false;    // has an outstanding NN eval request
  bool prefix_dirty = true;  // history changed since the last KV-prefix encode
  LeafRequest pending;     // the pumped leaf awaiting eval
  std::atomic<bool> result_ready{false};  // inference -> worker handoff flag
  NetEval neval;           // inference writes the unified eval here
};

// One queued leaf-eval request (features copied by value; the inference thread
// never touches a slot's Search tree). The FIRST request of a decision carries
// a snapshot of the slot's game history so the inference thread can refresh
// that slot's KV prefix before evaluating (no cross-thread history reads).
struct EvalReq {
  int slot;
  EvalFeatures feat;
  bool refresh_prefix = false;
  std::vector<int> prefix_snapshot;
};

static void run_search_selfplay(NNEvaluator &nn, int total_games, int slots_n,
                                int sims, unsigned seed,
                                const SeriesTable *series, float natural_frac,
                                std::vector<PlayerSample> &ps,
                                std::vector<OppSample> &os,
                                std::vector<GameRow> &gs) {
  StartStateSampler sampler;
  sampler.build(series, natural_frac);
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
  std::vector<std::vector<GameRow>> wgs(n_threads);  // per-worker game rows

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
    std::vector<GameRow> &lgs = wgs[wid];

    auto start_game = [&](Slot &s) -> bool {
      long gid = started.fetch_add(1, std::memory_order_relaxed);
      if (gid >= total_games) return false;
      s.game_id = (int)gid;
      s.turn = 0;
      s.rng.seed((unsigned)(seed ^ (0x9E3779B9u * (unsigned)(gid + 1))));
      s.game.shuffle_deal(s.rng);
      // Sample the series start state. Leader (a points) is seat 0 for any
      // state != (0,0); at (0,0) the 3-of-spades rule picks the opener.
      auto [a, b] = sampler.sample(s.rng);
      if (a == 0 && b == 0) {
        int fp = sampler.active
                     ? sample_first_player_3s(s.game.player_hand(0),
                                              s.game.player_hand(1), s.rng)
                     : 0;
        s.game.set_first_player(fp);
        s.pts[0] = 0;
        s.pts[1] = 0;
      } else {
        s.game.set_first_player(0);  // seat 0 leads at a non-initial state
        s.pts[0] = a;
        s.pts[1] = b;
      }
      s.history.clear();
      s.first_player = s.game.current_player();
      s.prefix_dirty = true;
      s.p_from = lps.size();
      s.o_from = los.size();
      s.active = true;
      s.searching = false;
      s.waiting = false;
      active_slots.fetch_add(1, std::memory_order_relaxed);
      return true;
    };

    auto setup_decision = [&](Slot &s) -> bool {
      auto legal = advance_to_decision(s.game, s.game_id, s.turn, s.history, lps, los);
      if (legal.empty()) return false;
      s.mover = s.game.current_player();
      SearchState st = mover_state(s.game, s.mover, s.pts);
      auto &tr = s.tree[s.mover];
      unsigned tseed =
          (unsigned)(seed ^ (0x9E3779B9u * (unsigned)(s.game_id * 2 + s.mover)));
      if (!tr) {
        SearchConfig sc{1.5f, sims, tseed, true};
        sc.series = series;
        tr = std::make_unique<Search>(st, s.history, sc);
      } else {
        tr->advance_root(st, s.history);
      }
      s.sims_done = 0;
      s.searching = true;
      s.prefix_dirty = true;  // history grew since the last decision
      return true;
    };

    auto complete_decision = [&](Slot &s) {
      Search &tr = *s.tree[s.mover];
      tr.finalize();
      // The policy target / sampling distribution is the MCTS visit counts, but
      // when those are degenerate (<2 visited moves — notably sims=1, where the
      // lone eval only expands the root and no child is visited) fall back to the
      // net's root PRIOR. Without this the move sampled is always kPASS and the
      // policy target is empty -> the player net collapses.
      auto visits = tr.root_visits();
      std::vector<std::pair<int, long>> prior;
      const std::vector<std::pair<int, long>> *dist = &visits;
      if (visits.size() < 2) { prior = tr.root_prior(); dist = &prior; }
      int m = sample_from_visits(*dist, s.rng);
      record_decision(s.game, s.mover, m, *dist, s.game_id, s.turn++,
                      (int)s.history.size(), lps, los);
      s.game.apply_move(m);  // trees persist; next setup_decision re-roots
      s.history.push_back(m);
      s.searching = false;
    };

    auto finalize_game = [&](Slot &s) {
      if (s.game.is_over()) {
        const int lc =
            backfill_values(s.game, s.pts, series, s.p_from, s.o_from, lps, los);
        lgs.push_back({s.game_id, s.first_player, s.game.get_winner(), s.pts[0],
                       s.pts[1], lc, s.history});
      }
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
        r.feat = std::move(s.pending.feat);
        if (s.prefix_dirty) {
          r.refresh_prefix = true;
          r.prefix_snapshot = s.history;
          s.prefix_dirty = false;
        }
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
          s.tree[s.mover]->apply_eval(s.neval);
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
          s.pending = std::move(req);
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
    std::vector<EvalFeatures> feats; std::vector<int> fslot;
    std::vector<int> rslot; std::vector<std::vector<int>> rhist;
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
          if (r.refresh_prefix) {
            rslot.push_back(r.slot);
            rhist.push_back(std::move(r.prefix_snapshot));
          }
          feats.push_back(std::move(r.feat));
          fslot.push_back(r.slot);
        }
        queue_.clear();
      }
      // KV prefixes FIRST (one padded batch), then all leaf evals against them.
      if (!rslot.empty()) {
        std::vector<const std::vector<int> *> hp;
        hp.reserve(rhist.size());
        for (auto &h : rhist) hp.push_back(&h);
        nn.set_prefixes(rslot, hp);
      }
      if (!feats.empty()) {
        auto res = nn.eval_batch(fslot, feats);
        for (std::size_t i = 0; i < fslot.size(); ++i) {
          Slot &s = slots[fslot[i]];
          s.neval = res[i];
          s.result_ready.store(true, std::memory_order_release);
        }
      }
      feats.clear(); fslot.clear(); rslot.clear(); rhist.clear();
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
    gs.insert(gs.end(), wgs[w].begin(), wgs[w].end());
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

// Raw 13-int rank-count columns (hand_0..hand_12); the 48-dim encodings are
// built in Python (training) / nn_eval (inference) from these + the history.
static void append_hand(std::vector<std::unique_ptr<arrow::Int32Builder>> &b,
                        const std::array<int, 13> &hand) {
  for (int r = 0; r < 13; ++r) b[r]->Append(hand[r]);
}

static void write_player_parquet(const std::string &path, const std::vector<PlayerSample> &v) {
  auto pool = arrow::default_memory_pool();
  arrow::Int32Builder game_id, turn_idx, hist_idx, owner_seat, opp_size, our_size;
  arrow::FloatBuilder value;
  std::vector<std::unique_ptr<arrow::Int32Builder>> hand;
  for (int r = 0; r < 13; ++r) hand.push_back(std::make_unique<arrow::Int32Builder>());
  arrow::ListBuilder legal(pool, std::make_shared<arrow::Int32Builder>(pool));  // concrete ids
  arrow::ListBuilder vmoves(pool, std::make_shared<arrow::Int32Builder>(pool));
  arrow::ListBuilder vcounts(pool, std::make_shared<arrow::Int32Builder>(pool));
  auto *legal_v = static_cast<arrow::Int32Builder *>(legal.value_builder());
  auto *vmoves_v = static_cast<arrow::Int32Builder *>(vmoves.value_builder());
  auto *vcounts_v = static_cast<arrow::Int32Builder *>(vcounts.value_builder());

  for (const auto &s : v) {
    game_id.Append(s.game_id); turn_idx.Append(s.turn_idx);
    hist_idx.Append(s.hist_idx); owner_seat.Append(s.mover);
    opp_size.Append(s.opp_size); our_size.Append(s.our_size);
    value.Append(s.value);
    append_hand(hand, s.hand);
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
  push(arrow::field("hist_idx", arrow::int32()), finish_i32(hist_idx));
  push(arrow::field("owner_seat", arrow::int32()), finish_i32(owner_seat));
  push(arrow::field("opp_size", arrow::int32()), finish_i32(opp_size));
  push(arrow::field("our_size", arrow::int32()), finish_i32(our_size));
  push(arrow::field("value", arrow::float32()), finish_f32(value));
  for (int r = 0; r < 13; ++r)
    push(arrow::field("hand_" + std::to_string(r), arrow::int32()), finish_i32(*hand[r]));
  std::shared_ptr<arrow::Array> la, vma, vca;
  legal.Finish(&la); vmoves.Finish(&vma); vcounts.Finish(&vca);
  push(arrow::field("legal", arrow::list(arrow::int32())), la);          // concrete move ids
  push(arrow::field("visit_moves", arrow::list(arrow::int32())), vma);
  push(arrow::field("visit_counts", arrow::list(arrow::int32())), vca);
  write_table(path, std::move(f), std::move(a));
}

static void write_opp_parquet(const std::string &path, const std::vector<OppSample> &v) {
  auto pool = arrow::default_memory_pool();
  arrow::Int32Builder game_id, turn_idx, hist_idx, owner_seat, opp_size, our_size, target_idx;
  arrow::FloatBuilder value;
  std::vector<std::unique_ptr<arrow::Int32Builder>> hand;
  for (int r = 0; r < 13; ++r) hand.push_back(std::make_unique<arrow::Int32Builder>());
  arrow::ListBuilder legal(pool, std::make_shared<arrow::Int32Builder>(pool));  // head slots
  auto *legal_v = static_cast<arrow::Int32Builder *>(legal.value_builder());

  std::vector<int> hidx;
  for (const auto &s : v) {
    game_id.Append(s.game_id); turn_idx.Append(s.turn_idx);
    hist_idx.Append(s.hist_idx); owner_seat.Append(s.observer);
    opp_size.Append(s.opp_size); our_size.Append(s.our_size);
    target_idx.Append(az_opp_head_index(s.move_id));  // head-slot target (no Python map)
    value.Append(s.value);
    append_hand(hand, s.hand);
    hidx.clear(); for (int m : s.legal) hidx.push_back(az_opp_head_index(m));  // map once in C++
    legal.Append(); legal_v->AppendValues(hidx.data(), (int64_t)hidx.size());
  }

  std::vector<std::shared_ptr<arrow::Field>> f;
  std::vector<std::shared_ptr<arrow::Array>> a;
  auto push = [&](std::shared_ptr<arrow::Field> fld, std::shared_ptr<arrow::Array> arr) {
    f.push_back(std::move(fld)); a.push_back(std::move(arr));
  };
  push(arrow::field("game_id", arrow::int32()), finish_i32(game_id));
  push(arrow::field("turn_idx", arrow::int32()), finish_i32(turn_idx));
  push(arrow::field("hist_idx", arrow::int32()), finish_i32(hist_idx));
  push(arrow::field("owner_seat", arrow::int32()), finish_i32(owner_seat));
  push(arrow::field("opp_size", arrow::int32()), finish_i32(opp_size));
  push(arrow::field("our_size", arrow::int32()), finish_i32(our_size));
  push(arrow::field("value", arrow::float32()), finish_f32(value));
  for (int r = 0; r < 13; ++r)
    push(arrow::field("hand_" + std::to_string(r), arrow::int32()), finish_i32(*hand[r]));
  std::shared_ptr<arrow::Array> la;
  legal.Finish(&la);
  push(arrow::field("legal", arrow::list(arrow::int32())), la);     // opp head-slot indices
  push(arrow::field("target_idx", arrow::int32()), finish_i32(target_idx));
  write_table(path, std::move(f), std::move(a));
}

static void write_games_parquet(const std::string &path, const std::vector<GameRow> &v) {
  auto pool = arrow::default_memory_pool();
  arrow::Int32Builder game_id, first_player, winner, pts0, pts1, loser_cards;
  arrow::ListBuilder moves(pool, std::make_shared<arrow::Int32Builder>(pool));
  auto *moves_v = static_cast<arrow::Int32Builder *>(moves.value_builder());
  for (const auto &g : v) {
    game_id.Append(g.game_id); first_player.Append(g.first_player);
    winner.Append(g.winner);
    pts0.Append(g.pts0); pts1.Append(g.pts1); loser_cards.Append(g.loser_cards);
    moves.Append(); moves_v->AppendValues(g.moves.data(), (int64_t)g.moves.size());
  }
  std::vector<std::shared_ptr<arrow::Field>> f;
  std::vector<std::shared_ptr<arrow::Array>> a;
  f.push_back(arrow::field("game_id", arrow::int32())); a.push_back(finish_i32(game_id));
  f.push_back(arrow::field("first_player", arrow::int32())); a.push_back(finish_i32(first_player));
  f.push_back(arrow::field("winner", arrow::int32())); a.push_back(finish_i32(winner));
  f.push_back(arrow::field("pts0", arrow::int32())); a.push_back(finish_i32(pts0));
  f.push_back(arrow::field("pts1", arrow::int32())); a.push_back(finish_i32(pts1));
  f.push_back(arrow::field("loser_cards", arrow::int32())); a.push_back(finish_i32(loser_cards));
  std::shared_ptr<arrow::Array> ma;
  moves.Finish(&ma);
  f.push_back(arrow::field("moves", arrow::list(arrow::int32()))); a.push_back(ma);
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
  const std::string games_out = arg(argc, argv, "--games-out", "data/az_games.parquet");
  const std::string model = arg(argc, argv, "--model", "");
  bool use_kv_cache = true;  // --no-kv-cache: full-recompute debug/correctness mode
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--no-kv-cache") == 0) use_kv_cache = false;
  const int games = std::atoi(arg(argc, argv, "--games", "1000"));
  const int sims = std::atoi(arg(argc, argv, "--sims", "100"));
  const int slots = std::atoi(arg(argc, argv, "--slots", "256"));
  // gen0 bootstrap teacher: a registry policy (e.g. typed_search) plays self-play
  // and the NN imitates it. Empty -> uniform-random gen0 (only relevant when no
  // --model is given).
  const std::string teacher = arg(argc, argv, "--teacher", "");
  const double teacher_param = std::atof(arg(argc, argv, "--teacher-param", "0"));
  const unsigned seed = (unsigned)std::strtoul(arg(argc, argv, "--seed", "0"), nullptr, 10);
  // Series objective: a V/natural-frequency table makes value targets the series
  // win probability (and lets NN selfplay sample start states). Absent => legacy
  // single-game win/loss objective at the (0,0) state only.
  const std::string series_v = arg(argc, argv, "--series-v", "");
  const float natural_frac = (float)std::atof(arg(argc, argv, "--natural-frac", "0.5"));
  SeriesTable series_table;
  const SeriesTable *series = nullptr;
  if (!series_v.empty()) {
    if (!load_series_table(series_v, series_table)) {
      std::fprintf(stderr, "[az_selfplay] failed to load series table %s\n",
                   series_v.c_str());
      return 1;
    }
    series = &series_table;
    std::printf("[az_selfplay] series objective: V table %s, natural-frac=%.2f\n",
                series_v.c_str(), natural_frac);
  }
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
  std::vector<GameRow> gs;

  const bool gen0 = model.empty();
  if (gen0 && !teacher.empty()) {
    std::printf("[az_selfplay] gen0 teacher self-play (%s, param=%.3g): %d games\n",
                teacher.c_str(), teacher_param, games);
    run_classic_selfplay(teacher, teacher_param, games, seed, series, ps, os, gs);
  } else if (gen0) {
    std::printf("[az_selfplay] gen0 random self-play: %d games\n", games);
    run_random_selfplay(games, seed, series, ps, os, gs);
  } else {
    std::printf("[az_selfplay] NN self-play: %d games, sims=%d, slots=%d, device=%s, kv=%s\n",
                games, sims, slots, device_str.c_str(), use_kv_cache ? "on" : "off");
    NNEvaluator nn(model, device, /*max_slots=*/slots, use_kv_cache);
    run_search_selfplay(nn, games, slots, sims, seed, series, natural_frac, ps, os, gs);
  }

  std::printf("[az_selfplay] samples: player=%zu opp=%zu games=%zu -> %s, %s, %s\n",
              ps.size(), os.size(), gs.size(), player_out.c_str(), opp_out.c_str(),
              games_out.c_str());
  write_player_parquet(player_out, ps);
  write_opp_parquet(opp_out, os);
  write_games_parquet(games_out, gs);
  return 0;
}
