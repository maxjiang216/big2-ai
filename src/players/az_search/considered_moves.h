#ifndef AZ_SEARCH_CONSIDERED_MOVES_H
#define AZ_SEARCH_CONSIDERED_MOVES_H

// ---------------------------------------------------------------------------
// Move-set audit result (az_search build step 1).
//
// The engine exposes LEGAL_MOVES_SIZE = 468 encoded move IDs (see util.h /
// move.cpp). For the two AlphaZero networks we do not give each of the 468 a
// distinct head logit; two groups of long combinations are special-cased, with
// the rationale captured below and verified in test/test_az_search.cpp:
//
//   * Triple-straight-5 (TS5, ids [kTRIPLESTRAIGHT5_START, kDOUBLESTRAIGHT8_START))
//     — 5 consecutive triples = 15 cards. Two distinct TS5 would need 6
//     consecutive ranks of triples = 18 cards > 16, so a hand holds AT MOST ONE
//     TS5 ever. Its rank therefore never offers a real choice. Collapse all 7
//     ids to ONE shared head entry in BOTH nets (masked to whichever single
//     variant is legal). TS5 leaves 1 card, so it is a normal continuation, not
//     a win — it must stay a selectable head entry.
//
//   * Double-straight-8 (DS8, ids [kDOUBLESTRAIGHT8_START, LEGAL_MOVES_SIZE))
//     — 8 consecutive pairs = 16 cards = the ENTIRE hand. Whenever a DS8 is
//     legal for us we empty our hand and win outright, so we always play it; it
//     is handled as a forced/terminal win, never a policy decision. The PLAYER
//     head DROPS all 5 DS8 ids. For the OPPONENT, a DS8 means the opponent wins
//     (value 0 for us) regardless of rank, so the OPPONENT head COLLAPSES all 5
//     DS8 ids into ONE entry (included only when some DS8 is possibly legal).
//
// Every other id (incl. straights up to S13, sisters up to DS7, full houses,
// bombs) is kept as a distinct head entry — the audit's at-most-one test does
// not hold for those (e.g. a 16-card hand can hold two overlapping DS7), so
// their rank is a genuine choice.
//
// Index layout (both heads keep ids [0, kTRIPLESTRAIGHT5_START) at identity):
//   player head:  idx 0..kTRIPLESTRAIGHT5_START-1 = id; idx kTRIPLESTRAIGHT5_START = TS5 slot.
//                 DS8 ids have no player index. dim = kTRIPLESTRAIGHT5_START + 1.
//   opponent head: same, plus idx kTRIPLESTRAIGHT5_START+1 = DS8 slot.
//                 dim = kTRIPLESTRAIGHT5_START + 2.
//
// This file is the single source of truth for the head dimensions; nn/model_az.py
// mirrors the identical rule (see az_player_head_index / az_opp_head_index).
// ---------------------------------------------------------------------------

#include "util.h"  // kTRIPLESTRAIGHT5_START, kDOUBLESTRAIGHT8_START, LEGAL_MOVES_SIZE

namespace az_search {

// TS5 occupies [kTRIPLESTRAIGHT5_START, kDOUBLESTRAIGHT8_START); DS8 occupies
// [kDOUBLESTRAIGHT8_START, LEGAL_MOVES_SIZE). Everything below kTRIPLESTRAIGHT5_START
// is kept verbatim.
constexpr int kTS5RangeBegin = kTRIPLESTRAIGHT5_START;
constexpr int kTS5RangeEnd = kDOUBLESTRAIGHT8_START;   // == DS8 begin
constexpr int kDS8RangeBegin = kDOUBLESTRAIGHT8_START;
constexpr int kDS8RangeEnd = LEGAL_MOVES_SIZE;

// Shared head slots.
constexpr int kTS5Slot = kTRIPLESTRAIGHT5_START;        // both heads
constexpr int kOppDS8Slot = kTRIPLESTRAIGHT5_START + 1; // opponent head only

// Head output dimensions.
constexpr int AZ_PLAYER_HEAD_DIM = kTRIPLESTRAIGHT5_START + 1;  // + TS5 slot
constexpr int AZ_OPP_HEAD_DIM = kTRIPLESTRAIGHT5_START + 2;     // + TS5 + DS8 slots

// Canonical (lowest-id) representative of each collapsed group.
constexpr int kTS5CanonicalMove = kTRIPLESTRAIGHT5_START;
constexpr int kDS8CanonicalMove = kDOUBLESTRAIGHT8_START;

// move id -> player policy head index, or -1 if the id is not a player head
// entry (the dropped DS8 ids).
inline constexpr int az_player_head_index(int move_id) {
  if (move_id < kTS5RangeBegin) return move_id;
  if (move_id < kTS5RangeEnd) return kTS5Slot;
  return -1;  // DS8: never a player policy choice (auto-win, handled elsewhere)
}

// move id -> opponent behavior head index (every id maps; never -1).
inline constexpr int az_opp_head_index(int move_id) {
  if (move_id < kTS5RangeBegin) return move_id;
  if (move_id < kTS5RangeEnd) return kTS5Slot;
  return kOppDS8Slot;  // DS8
}

// player head index -> a representative engine move id. For the TS5 slot the
// caller must resolve the single legal TS5 against the hand; this returns the
// canonical id as a fallback.
inline constexpr int az_player_move_for_index(int idx) {
  if (idx < kTS5Slot) return idx;
  return kTS5CanonicalMove;
}

// opponent head index -> a representative engine move id (canonical for slots).
inline constexpr int az_opp_move_for_index(int idx) {
  if (idx < kTS5Slot) return idx;
  if (idx == kTS5Slot) return kTS5CanonicalMove;
  return kDS8CanonicalMove;
}

}  // namespace az_search

#endif  // AZ_SEARCH_CONSIDERED_MOVES_H
