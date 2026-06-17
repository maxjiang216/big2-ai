// az_seq_record.h — shared sample extraction + parquet writers for the
// history-transformer ("seq") nets. Used by BOTH the az_search self-play
// generator (src/datagen/az_selfplay.cpp) and the perfect-information teacher
// distillation generator (src/datagen/az_pi_selfplay.cpp), so PI-distilled data
// uses the exact same recording filter (classify_position) the az_ii / az_search
// search will query, and the same dataset_seq.py parquet schema.
//
// Policy / behavior targets are the ONE-HOT played move (imitation / cloning).
// Both value targets are backfilled from the game outcome (series value with a
// table, else 1/0). opp_hand_* columns are the EXACT opponent hand (the belief
// target for the AR opp-hand head). opp_max thermo is NOT stored — dataset_seq.py
// recomputes it from the move list + hand.
//
// All functions are static/inline (internal linkage): each translation unit gets
// its own copy, no ODR concern.

#pragma once

#include "az_search/considered_moves.h"  // az_opp_head_index
#include "az_search/features.h"  // opp_max_counts / trick_counts / forced-win helpers
#include "nn_encode.h"

#include "game.h"
#include "move.h"
#include "series.h"
#include "util.h"

#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <parquet/arrow/writer.h>

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace az_seq_record {

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
  std::array<int, 13> opp_hand{};  // opponent's EXACT current hand (belief target)
  int opp_size, our_size;
  std::vector<int> legal;
  std::vector<int> visit_moves, visit_counts;
  int mover;
  float value = 0.0f;
  int margin = 0;  // signed final loser_cards from owner POV (won:+ lost:-)
};

struct OppSample {
  int game_id, turn_idx;
  int hist_idx;  // moves applied before this decision (sequence readout index)
  std::array<int, 13> hand, opp_max, trick;  // hand = observer's exact hand
  std::array<int, 13> opp_hand{};  // mover's EXACT current hand (belief target)
  int opp_size, our_size;
  std::vector<int> legal;
  int move_id;
  int observer;
  float value = 0.0f;
  int margin = 0;  // signed final loser_cards from owner (observer) POV
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
  ps.opp_hand = game.player_hand(obs);  // opponent's exact hand (belief target)
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
  os.opp_hand = game.player_hand(mover);         // mover's exact hand (belief target)
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
  for (std::size_t i = p_from; i < ps.size(); ++i) {
    ps[i].value = (ps[i].mover == winner) ? tw : 1.0f - tw;
    ps[i].margin = (ps[i].mover == winner) ? loser_cards : -loser_cards;
  }
  for (std::size_t i = o_from; i < os.size(); ++i) {
    os[i].value = (os[i].observer == winner) ? tw : 1.0f - tw;
    os[i].margin = (os[i].observer == winner) ? loser_cards : -loser_cards;
  }
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
// Parquet writers (schemas mirror nn/dataset_seq.py).
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
  arrow::Int32Builder game_id, turn_idx, hist_idx, owner_seat, opp_size, our_size, margin;
  arrow::FloatBuilder value;
  std::vector<std::unique_ptr<arrow::Int32Builder>> hand, opp_hand;
  for (int r = 0; r < 13; ++r) hand.push_back(std::make_unique<arrow::Int32Builder>());
  for (int r = 0; r < 13; ++r) opp_hand.push_back(std::make_unique<arrow::Int32Builder>());
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
    value.Append(s.value); margin.Append(s.margin);
    append_hand(hand, s.hand);
    append_hand(opp_hand, s.opp_hand);
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
  push(arrow::field("margin", arrow::int32()), finish_i32(margin));
  for (int r = 0; r < 13; ++r)
    push(arrow::field("hand_" + std::to_string(r), arrow::int32()), finish_i32(*hand[r]));
  for (int r = 0; r < 13; ++r)
    push(arrow::field("opp_hand_" + std::to_string(r), arrow::int32()), finish_i32(*opp_hand[r]));
  std::shared_ptr<arrow::Array> la, vma, vca;
  legal.Finish(&la); vmoves.Finish(&vma); vcounts.Finish(&vca);
  push(arrow::field("legal", arrow::list(arrow::int32())), la);          // concrete move ids
  push(arrow::field("visit_moves", arrow::list(arrow::int32())), vma);
  push(arrow::field("visit_counts", arrow::list(arrow::int32())), vca);
  write_table(path, std::move(f), std::move(a));
}

static void write_opp_parquet(const std::string &path, const std::vector<OppSample> &v) {
  auto pool = arrow::default_memory_pool();
  arrow::Int32Builder game_id, turn_idx, hist_idx, owner_seat, opp_size, our_size, target_idx, margin;
  arrow::FloatBuilder value;
  std::vector<std::unique_ptr<arrow::Int32Builder>> hand, opp_hand;
  for (int r = 0; r < 13; ++r) hand.push_back(std::make_unique<arrow::Int32Builder>());
  for (int r = 0; r < 13; ++r) opp_hand.push_back(std::make_unique<arrow::Int32Builder>());
  arrow::ListBuilder legal(pool, std::make_shared<arrow::Int32Builder>(pool));  // head slots
  auto *legal_v = static_cast<arrow::Int32Builder *>(legal.value_builder());

  std::vector<int> hidx;
  for (const auto &s : v) {
    game_id.Append(s.game_id); turn_idx.Append(s.turn_idx);
    hist_idx.Append(s.hist_idx); owner_seat.Append(s.observer);
    opp_size.Append(s.opp_size); our_size.Append(s.our_size);
    target_idx.Append(az_opp_head_index(s.move_id));  // head-slot target (no Python map)
    value.Append(s.value); margin.Append(s.margin);
    append_hand(hand, s.hand);
    append_hand(opp_hand, s.opp_hand);
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
  push(arrow::field("margin", arrow::int32()), finish_i32(margin));
  for (int r = 0; r < 13; ++r)
    push(arrow::field("hand_" + std::to_string(r), arrow::int32()), finish_i32(*hand[r]));
  for (int r = 0; r < 13; ++r)
    push(arrow::field("opp_hand_" + std::to_string(r), arrow::int32()), finish_i32(*opp_hand[r]));
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

}  // namespace az_seq_record
