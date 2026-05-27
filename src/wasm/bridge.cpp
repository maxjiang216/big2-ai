// WebAssembly bridge for the typed-Shannon search AI.
//
// Exposes a tiny C ABI to JavaScript:
//   ts_load_tables()  — load the six v5 table files from MEMFS /tables/*.bin.
//   ts_select_move(hand,discard,opp_count,trick,out) — choose the CPU's move.
//
// All hand/discard/trick vectors are 13-element rank-count arrays in the
// engine's rank order (index 0 = '3' … 10 = 'K', 11 = 'A', 12 = '2'). The
// caller passes the trick to beat as the cards on the table (empty = we lead);
// the bridge resolves it to a move id internally. The chosen move's rank-count
// vector is written to `out` so the UI can pull the matching cards from its own
// (suited) representation.

#include "move.h"
#include "partial_game.h"
#include "tablebase_peek.h"
#include "util.h"

#include "typed_search/eval_table.h"
#include "typed_search/move_prob_disc_table.h"
#include "typed_search/move_prob_table.h"
#include "typed_search/typed_search.h"
#include "typed_search/typed_search_player.h"  // TypedSearchTables bundle

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <emscripten/emscripten.h>

using typed_search::TypedSearch;
using typed_search::TypedSearchTables;

namespace {

std::shared_ptr<TypedSearchTables> g_tables;
std::uint64_t g_move_counter = 0;

// Find the unique non-pass move id whose card requirement matches the given
// rank-count vector. The mapping is one-to-one for the cards that make up a
// single combination (including bomb kickers and full-house pairs), so a
// linear scan over MOVE_TO_CARDS resolves it unambiguously. Returns -1 if no
// move matches (treated as "fresh trick").
int find_move_id_by_counts(const int *counts, int total) {
  for (int mid = 1; mid < LEGAL_MOVES_SIZE; ++mid) {
    const auto &mc = MOVE_TO_CARDS[mid];
    if (mc[13] != total) continue;
    bool ok = true;
    for (int r = 0; r < 13; ++r) {
      if (mc[r] != counts[r]) {
        ok = false;
        break;
      }
    }
    if (ok) return mid;
  }
  return -1;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
int ts_load_tables() {
  g_tables = std::make_shared<TypedSearchTables>();
  g_tables->eval_extended.load("/tables/eval_extended.bin");
  g_tables->eval_main.load("/tables/eval_main.bin");
  g_tables->eval_fallback.load("/tables/eval_fallback.bin");
  g_tables->mp_disc.load("/tables/mp_disc.bin");
  g_tables->mp_main.load("/tables/mp_main.bin");
  g_tables->mp_fallback.load("/tables/mp_fallback.bin");
  // Sanity signal: non-zero if the largest table loaded any entries.
  return static_cast<int>(g_tables->eval_extended.size());
}

// hand, discard, trick: pointers to 13 int32 rank counts in WASM heap.
// out: pointer to 13 int32; receives the chosen move's rank counts.
// Returns the engine move id (0 = pass).
EMSCRIPTEN_KEEPALIVE
int ts_select_move(const int *hand_p, const int *discard_p, int opp_count,
                   const int *trick_p, int *out) {
  std::array<int, 13> hand{};
  std::array<int, 13> discard{};
  int trick_total = 0;
  for (int r = 0; r < 13; ++r) {
    hand[r] = hand_p[r];
    discard[r] = discard_p[r];
    trick_total += trick_p[r];
  }

  Move last(Move::Combination::kPass);
  if (trick_total > 0) {
    int tmid = find_move_id_by_counts(trick_p, trick_total);
    if (tmid > 0) last = Move(tmid);
  }

  // turn 0 = the CPU (this player) is to move.
  PartialGame pg(hand, discard, opp_count, last, 0);

  int move_id = kPASS;
  TablebasePeekResult tb = peek_tablebase_move(pg);
  if (tb.move) {
    move_id = encodeMove(*tb.move);
  } else if (g_tables) {
    const std::uint64_t seed =
        0xD1B54A32D192ED03ull ^ (0x9E3779B97F4A7C15ull * (++g_move_counter));
    TypedSearch search(g_tables->eval_extended, g_tables->eval_main,
                       g_tables->eval_fallback, g_tables->mp_disc,
                       g_tables->mp_main, seed);
    TypedSearch::Result res = search.run(hand, discard, opp_count, last);
    if (res.move_id < 0) {
      auto legal = compute_legal_moves(hand, last);
      move_id = legal.empty() ? kPASS : legal.front();
    } else {
      move_id = res.move_id;
    }
  }

  const auto &mc = MOVE_TO_CARDS[move_id];
  for (int r = 0; r < 13; ++r) out[r] = mc[r];
  return move_id;
}

// Number of legal non-pass moves `hand` has against the trick on the table.
// 0 means the holder is forced to pass. Used by the UI to flag forced passes.
EMSCRIPTEN_KEEPALIVE
int ts_legal_move_count(const int *hand_p, const int *trick_p) {
  std::array<int, 13> hand{};
  int trick_total = 0;
  for (int r = 0; r < 13; ++r) {
    hand[r] = hand_p[r];
    trick_total += trick_p[r];
  }
  Move last(Move::Combination::kPass);
  if (trick_total > 0) {
    int tmid = find_move_id_by_counts(trick_p, trick_total);
    if (tmid > 0) last = Move(tmid);
  }
  int n = 0;
  for (int mid : compute_legal_moves(hand, last)) {
    if (mid != kPASS) ++n;
  }
  return n;
}

}  // extern "C"
