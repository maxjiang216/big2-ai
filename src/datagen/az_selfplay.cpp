// az_selfplay: self-play data generation for the az_search nets.
//
// Two modes:
//   * gen0 bootstrap (no --player-model): uniform-random self-play. Policy /
//     behavior targets are the one-hot played move; this mainly bootstraps the
//     value heads from realized outcomes.
//   * gen N>=1 (--player-model / --opp-model): NN-guided MCTS self-play. NN leaf
//     evaluations are BATCHED ACROSS MANY CONCURRENT GAMES (the throughput win):
//     every game advances its current search by simulations, leaf requests are
//     pooled into player-net / opp-net batches, flushed, and distributed back.
//
// Each real decision (>=2 legal moves, not a tablebase/forced position — those
// are excluded) emits two samples:
//   * a PLAYER sample (the mover's view): MCTS visit distribution policy target;
//   * an OPP sample (the observer's public view): the move actually played, the
//     behavior (imitation) target.
// Both value targets are backfilled from the game winner. The move played is
// sampled proportional to the root visit counts (exploration).
//
// Output: two Parquet files matching the nn/dataset.py schemas
// (az_player_genN.parquet, az_opp_genN.parquet).

#include "az_search/az_search.h"
#include "az_search/features.h"
#include "az_search/nn_eval.h"

#include "game.h"
#include "move.h"
#include "tablebase_opp1.h"
#include "util.h"

#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <parquet/arrow/writer.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
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

// Advance the game through tablebase / single-legal moves (excluded from
// training) until a real decision point. Returns the legal moves there, or an
// empty vector if the game ended first.
static std::vector<int> advance_to_decision(Game &game) {
  while (!game.is_over()) {
    auto legal = game.get_legal_moves();
    int tb = tablebase_move_id(game, game.current_player(), legal);
    if (tb >= 0) { game.apply_move(tb); continue; }
    if (legal.size() == 1) { game.apply_move(legal[0]); continue; }
    return legal;
  }
  return {};
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
  std::array<int, 13> opp_max, trick;
  int opp_size, our_size;
  std::vector<int> legal;
  int move_id;
  int observer;
  float value = 0.0f;
};

// ---------------------------------------------------------------------------
// Start-state sampler: most games start from a full deal; a fraction start from
// a mid-game state, obtained by playing a random number of uniform-random legal
// moves forward from a fresh deal. This guarantees deck + last-trick consistency
// by construction (the state is reachable) while broadening coverage of smaller
// hands / arbitrary current tricks.
// ---------------------------------------------------------------------------
static void setup_game(Game &game, std::mt19937 &rng, float start_frac) {
  game.shuffle_deal(rng);
  if (std::uniform_real_distribution<float>(0, 1)(rng) >= start_frac) return;
  // Mid-game: play a random prefix of random legal moves.
  int steps = std::uniform_int_distribution<int>(1, 12)(rng);
  for (int i = 0; i < steps && !game.is_over(); ++i) {
    auto legal = game.get_legal_moves();
    if (legal.empty()) break;
    game.apply_move(legal[std::uniform_int_distribution<int>(0, (int)legal.size() - 1)(rng)]);
  }
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

// Record the player + opp samples for a decision, given the chosen move and the
// (possibly empty) MCTS visit distribution. For gen0 random play the visit list
// is a single (move, 1) entry.
static void record_decision(Game &game, int mover, int chosen_move,
                            const std::vector<std::pair<int, long>> &visits,
                            int game_id, int turn_idx,
                            std::vector<PlayerSample> &psamples,
                            std::vector<OppSample> &osamples) {
  const int obs = 1 - mover;
  auto hand = game.player_hand(mover);
  auto ohand = game.player_hand(obs);
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

  OppSample os;
  os.game_id = game_id; os.turn_idx = turn_idx;
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

// ---------------------------------------------------------------------------
// gen0: uniform-random self-play (no search).
// ---------------------------------------------------------------------------
static void run_random_selfplay(int total_games, float start_frac, unsigned seed,
                                std::vector<PlayerSample> &ps,
                                std::vector<OppSample> &os) {
  std::mt19937 rng(seed);
  for (int g = 0; g < total_games; ++g) {
    Game game;
    setup_game(game, rng, start_frac);
    const std::size_t pf = ps.size(), of = os.size();
    int turn = 0;
    while (true) {
      auto legal = advance_to_decision(game);
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
// gen N>=1: batched NN-guided MCTS self-play across concurrent game slots.
//
// Each slot keeps ONE persistent search tree per seat for the whole game. When
// a seat moves again, advance_root re-roots onto the current state, reusing the
// subtree (and its accumulated visits) that the seat already explored on earlier
// turns — more effective search per fixed sim budget. Both seats use the same
// logic, so the recorded targets stay symmetric/unbiased. (Memory grows with
// slots x sims x game length; tune via --slots.)
// ---------------------------------------------------------------------------
struct Slot {
  int game_id = -1;
  Game game;
  std::unique_ptr<Search> tree[2];  // persistent per-seat search trees
  int mover = 0;
  int turn = 0;
  int sims_done = 0;
  std::size_t p_from = 0, o_from = 0;  // sample range start for this game
  bool active = false;
  bool searching = false;  // mid-search (true) vs needs a new decision setup
  bool has_pending = false;
  LeafRequest pending;
};

static void run_search_selfplay(NNEvaluator &nn, int total_games, int slots_n,
                                int sims, float start_frac, unsigned seed,
                                std::vector<PlayerSample> &ps,
                                std::vector<OppSample> &os) {
  std::mt19937 rng(seed);
  std::vector<Slot> slots(slots_n);
  int started = 0;

  auto finalize_game = [&](Slot &s) {
    if (s.game.is_over()) backfill_values(s.game.get_winner(), s.p_from, s.o_from, ps, os);
    s.active = false;
    s.searching = false;
    s.tree[0].reset();
    s.tree[1].reset();  // free both persistent trees at game end
  };

  // Set up the next decision in slot s; returns false if the game ended. Re-roots
  // the mover's persistent tree onto the current state (creating it on the mover's
  // first turn), reusing any subtree the mover explored on earlier turns.
  auto setup_decision = [&](Slot &s) -> bool {
    auto legal = advance_to_decision(s.game);
    if (legal.empty()) return false;
    s.mover = s.game.current_player();
    SearchState st = mover_state(s.game, s.mover);
    auto &tr = s.tree[s.mover];
    if (!tr)
      tr = std::make_unique<Search>(
          st, SearchConfig{1.5f, sims,
                           (unsigned)(seed ^ (0x9E3779B9u * (s.game_id * 2 + s.mover))),
                           /*training=*/true});
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
    int m = sample_from_visits(visits, rng);
    record_decision(s.game, s.mover, m, visits, s.game_id, s.turn++, ps, os);
    s.game.apply_move(m);  // trees persist; next setup_decision re-roots via the memo
    s.searching = false;
    if (s.game.is_over()) finalize_game(s);
  };

  // Pump a slot's (mover's) search until it queues an NN eval or exhausts its
  // budget. Returns true if an eval was queued.
  auto pump = [&](Slot &s) -> bool {
    Search &tr = *s.tree[s.mover];
    while (s.sims_done < sims) {
      LeafRequest req = tr.select_leaf();
      if (!req.needs_eval) { ++s.sims_done; continue; }
      s.pending = req;
      s.has_pending = true;
      return true;
    }
    return false;
  };

  std::vector<PlayerFeatures> pbatch;
  std::vector<OppFeatures> obatch;
  std::vector<Slot *> pwait, owait;

  for (;;) {
    pbatch.clear(); obatch.clear(); pwait.clear(); owait.clear();
    bool any_active = false;

    for (auto &s : slots) {
      if (!s.active) {
        if (started >= total_games) continue;
        s.game_id = started++;
        s.turn = 0;
        s.p_from = ps.size();
        s.o_from = os.size();
        setup_game(s.game, rng, start_frac);
        s.active = true;
        s.searching = false;
      }
      any_active = true;
      if (!s.searching && !setup_decision(s)) { finalize_game(s); continue; }
      if (pump(s)) {
        if (s.pending.is_player) { pbatch.push_back(s.pending.pfeat); pwait.push_back(&s); }
        else { obatch.push_back(s.pending.ofeat); owait.push_back(&s); }
      } else {
        complete_decision(s);
      }
    }

    if (!pbatch.empty()) {
      auto res = nn.eval_players(pbatch);
      for (std::size_t i = 0; i < pwait.size(); ++i) {
        Slot *s = pwait[i];
        s->tree[s->mover]->apply_eval(res[i].value, res[i].logits.data());
        ++s->sims_done;
        s->has_pending = false;
      }
    }
    if (!obatch.empty()) {
      auto res = nn.eval_opps(obatch);
      for (std::size_t i = 0; i < owait.size(); ++i) {
        Slot *s = owait[i];
        s->tree[s->mover]->apply_eval(res[i].value, res[i].logits.data());
        ++s->sims_done;
        s->has_pending = false;
      }
    }

    if (!any_active && started >= total_games) break;
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
  arrow::Int32Builder opp_max[13], trick[13];
  arrow::FloatBuilder value;
  arrow::ListBuilder legal(pool, std::make_shared<arrow::Int32Builder>(pool));
  auto *legal_v = static_cast<arrow::Int32Builder *>(legal.value_builder());

  for (const auto &s : v) {
    game_id.Append(s.game_id); turn_idx.Append(s.turn_idx);
    opp_size.Append(s.opp_size); our_size.Append(s.our_size);
    move_id.Append(s.move_id); value.Append(s.value);
    for (int r = 0; r < 13; ++r) { opp_max[r].Append(s.opp_max[r]); trick[r].Append(s.trick[r]); }
    legal.Append(); legal_v->AppendValues(s.legal.data(), (int64_t)s.legal.size());
  }

  std::vector<std::shared_ptr<arrow::Field>> f;
  std::vector<std::shared_ptr<arrow::Array>> a;
  auto push = [&](std::shared_ptr<arrow::Field> fld, std::shared_ptr<arrow::Array> arr) {
    f.push_back(std::move(fld)); a.push_back(std::move(arr));
  };
  push(arrow::field("game_id", arrow::int32()), finish_i32(game_id));
  push(arrow::field("turn_idx", arrow::int32()), finish_i32(turn_idx));
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
  const float start_frac = std::atof(arg(argc, argv, "--start-frac", "0.2"));
  const unsigned seed = (unsigned)std::strtoul(arg(argc, argv, "--seed", "0"), nullptr, 10);

  std::vector<PlayerSample> ps;
  std::vector<OppSample> os;

  const bool gen0 = player_model.empty() || opp_model.empty();
  if (gen0) {
    std::printf("[az_selfplay] gen0 random self-play: %d games (start_frac=%.2f)\n", games, start_frac);
    run_random_selfplay(games, start_frac, seed, ps, os);
  } else {
    std::printf("[az_selfplay] NN self-play: %d games, sims=%d, slots=%d\n", games, sims, slots);
    NNEvaluator nn(player_model, opp_model, torch::Device(torch::kCPU));
    run_search_selfplay(nn, games, slots, sims, start_frac, seed, ps, os);
  }

  std::printf("[az_selfplay] samples: player=%zu opp=%zu -> %s, %s\n",
              ps.size(), os.size(), player_out.c_str(), opp_out.c_str());
  write_player_parquet(player_out, ps);
  write_opp_parquet(opp_out, os);
  return 0;
}
