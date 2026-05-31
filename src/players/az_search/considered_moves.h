#ifndef AZ_SEARCH_CONSIDERED_MOVES_H
#define AZ_SEARCH_CONSIDERED_MOVES_H

// ---------------------------------------------------------------------------
// az_search head layouts.
//
// OPPONENT behavior + per-move-value head (UNCHANGED, flat):
//   one logit per engine move id, with two long-combination collapses —
//   * Triple-straight-5 (TS5): a hand holds at most one, so all 7 ids collapse
//     to ONE shared slot (masked to the single legal variant).
//   * Double-straight-8 (DS8): 16-card whole-hand auto-win; all 5 ids collapse
//     to ONE opponent slot (the opponent winning outright, value 0 for us).
//   Index layout: ids [0, kTRIPLESTRAIGHT5_START) kept verbatim, then the TS5
//   slot, then (opp only) the DS8 slot.  dim = kTRIPLESTRAIGHT5_START + 2.
//
// PLAYER policy head (FACTORED / hierarchical, dim 138). The flat 457-wide head
// was dominated by full houses (132) + bombs (156) + straights (132), whose rare
// members got near-untrained logits.  We factor each family into a few shared
// logits and compose a concrete move's logit as the SUM of its path components;
// the masked softmax over concrete legal moves is then mathematically a nested
// softmax (family -> rank/high -> aux/low) with log-sum-exp entries.  Layout:
//
//   pass                1   [0]
//   single  (rank)      13  [1 .. 13]    rank 3..2   -> idx rank-3
//   double  (rank)      12  [14 .. 25]   rank 3..A   -> idx rank-3
//   triple  (rank)      11  [26 .. 36]   rank 3..K   -> idx rank-3
//   full-house rank     12  [37 .. 48]   triple 3..A -> idx rank-3
//   full-house aux pair 12  [49 .. 60]   pair 3..A   -> idx aux-3  (shared/rank)
//   bomb entry          1   [61]         overall bomb-vs-rest mass (rank uniform)
//   bomb bare           1   [62]         no-kicker option
//   bomb kicker (rank)  13  [63 .. 75]   kicker 3..2 -> idx aux-3  (shared/rank)
//   single-straight hi  10  [76 .. 85]   hi 6..2(eng 6..15) -> idx hi-6
//   single-straight lo  10  [86 .. 95]   lo 2..J(eng 2..11) -> idx lo-2
//   double-straight hi  11  [96 .. 106]  hi 4..A(eng 4..14) -> idx hi-4
//   double-straight lo  11  [107 .. 117] lo 3..K(eng 3..13) -> idx lo-3
//   triple-straight hi  10  [118 .. 127] hi 4..K(eng 4..13) -> idx hi-4
//   triple-straight lo  10  [128 .. 137] lo 3..Q(eng 3..12) -> idx lo-3
//
// A straight is (type, hi, lo) with rank-span = hi - lo + 1 (engine "rank" is a
// linear straight scale where 2 may be low=2 in 2-3-4-5-6 or high=15 in JQKA2),
// so length is implied, never a logit.  DS8 / TS5 need no special handling: the
// insta-win check in expand_player catches DS8 (whole-hand auto-win) before the
// policy is consulted, and TS5 is just one concrete triple-straight a hand holds
// at most once.  See nn/model_az.py for the mirrored Python head + composition
// matrix (built from this file via az_compose_gen).
// ---------------------------------------------------------------------------

#include "move.h"
#include "util.h"  // kTRIPLESTRAIGHT5_START, kDOUBLESTRAIGHT8_START, all_moves()

namespace az_search {

// =====================  OPPONENT head (unchanged, flat)  ====================

constexpr int kTS5RangeBegin = kTRIPLESTRAIGHT5_START;
constexpr int kTS5RangeEnd = kDOUBLESTRAIGHT8_START;   // == DS8 begin
constexpr int kDS8RangeBegin = kDOUBLESTRAIGHT8_START;
constexpr int kDS8RangeEnd = LEGAL_MOVES_SIZE;

constexpr int kTS5Slot = kTRIPLESTRAIGHT5_START;        // opp head
constexpr int kOppDS8Slot = kTRIPLESTRAIGHT5_START + 1; // opp head only

constexpr int AZ_OPP_HEAD_DIM = kTRIPLESTRAIGHT5_START + 2;  // + TS5 + DS8 slots

constexpr int kTS5CanonicalMove = kTRIPLESTRAIGHT5_START;
constexpr int kDS8CanonicalMove = kDOUBLESTRAIGHT8_START;

// move id -> opponent behavior/value head index (every id maps; never -1).
inline constexpr int az_opp_head_index(int move_id) {
  if (move_id < kTS5RangeBegin) return move_id;
  if (move_id < kTS5RangeEnd) return kTS5Slot;
  return kOppDS8Slot;  // DS8
}

// opponent head index -> a representative engine move id (canonical for slots).
inline constexpr int az_opp_move_for_index(int idx) {
  if (idx < kTS5Slot) return idx;
  if (idx == kTS5Slot) return kTS5CanonicalMove;
  return kDS8CanonicalMove;
}

// =====================  PLAYER head (factored, dim 138)  ====================

// Region bases (head-index space).
constexpr int kPHpass = 0;
constexpr int kPHsingle = 1;        // 13
constexpr int kPHdouble = 14;       // 12
constexpr int kPHtriple = 26;       // 11
constexpr int kPHfhRank = 37;       // 12
constexpr int kPHfhAux = 49;        // 12
constexpr int kPHbombEntry = 61;    // 1
constexpr int kPHbombBare = 62;     // 1
constexpr int kPHbombKicker = 63;   // 13
constexpr int kPHsglHigh = 76;      // 10
constexpr int kPHsglLow = 86;       // 10
constexpr int kPHdblHigh = 96;      // 11
constexpr int kPHdblLow = 107;      // 11
constexpr int kPHtplHigh = 118;     // 10
constexpr int kPHtplLow = 128;      // 10

constexpr int AZ_PLAYER_HEAD_DIM = 138;

// Level-1 families (match the search's player-node grouping).
enum PlayerFamily {
  kFamPass = 0,
  kFamSingle = 1,
  kFamDouble = 2,
  kFamTriple = 3,
  kFamFullHouse = 4,
  kFamBomb = 5,
  kFamSglStraight = 6,
  kFamDblStraight = 7,
  kFamTplStraight = 8,
};

// Up to two head-logit indices a concrete move's logit sums (its path through
// the family -> rank/high -> aux/low tree).
struct PathLogits {
  int idx[2];
  int n;
};

inline int player_family_id(int move_id) {
  using C = Move::Combination;
  switch (all_moves()[move_id].combination) {
    case C::kPass: return kFamPass;
    case C::kSingle: return kFamSingle;
    case C::kDouble: return kFamDouble;
    case C::kTriple: return kFamTriple;
    case C::kFullHouse: return kFamFullHouse;
    case C::kBomb: return kFamBomb;
    case C::kTripleStraight2:
    case C::kTripleStraight3:
    case C::kTripleStraight4:
    case C::kTripleStraight5: return kFamTplStraight;
    case C::kDoubleStraight2:
    case C::kDoubleStraight3:
    case C::kDoubleStraight4:
    case C::kDoubleStraight5:
    case C::kDoubleStraight6:
    case C::kDoubleStraight7:
    case C::kDoubleStraight8: return kFamDblStraight;
    default: return kFamSglStraight;  // kStraight5..kStraight13
  }
}

// Level-2 sub-key within a family (the primary axis): rank for
// single/double/triple/full-house, high card for straights, kicker for bombs.
inline int player_subgroup_key(int move_id) {
  const Move m = all_moves()[move_id];
  if (m.combination == Move::Combination::kBomb) return m.auxiliary;
  return m.rank;  // rank == straight high card for straight families
}

inline PathLogits player_path_logits(int move_id) {
  using C = Move::Combination;
  const Move m = all_moves()[move_id];
  switch (m.combination) {
    case C::kPass: return {{kPHpass, 0}, 1};
    case C::kSingle: return {{kPHsingle + (m.rank - 3), 0}, 1};
    case C::kDouble: return {{kPHdouble + (m.rank - 3), 0}, 1};
    case C::kTriple: return {{kPHtriple + (m.rank - 3), 0}, 1};
    case C::kFullHouse:
      return {{kPHfhRank + (m.rank - 3), kPHfhAux + (m.auxiliary - 3)}, 2};
    case C::kBomb:
      if (m.auxiliary == 0) return {{kPHbombEntry, kPHbombBare}, 2};
      return {{kPHbombEntry, kPHbombKicker + (m.auxiliary - 3)}, 2};
    default: break;
  }
  // Straights: span from the combination, hi = engine straight rank, lo derived.
  const int c = static_cast<int>(m.combination);
  if (c >= static_cast<int>(C::kStraight5) &&
      c <= static_cast<int>(C::kStraight13)) {
    const int span = 5 + (c - static_cast<int>(C::kStraight5));
    const int hi = m.rank, lo = hi - span + 1;
    return {{kPHsglHigh + (hi - 6), kPHsglLow + (lo - 2)}, 2};
  }
  if (c >= static_cast<int>(C::kDoubleStraight2) &&
      c <= static_cast<int>(C::kDoubleStraight8)) {
    const int span = 2 + (c - static_cast<int>(C::kDoubleStraight2));
    const int hi = m.rank, lo = hi - span + 1;
    return {{kPHdblHigh + (hi - 4), kPHdblLow + (lo - 3)}, 2};
  }
  // kTripleStraight2..5
  const int span = 2 + (c - static_cast<int>(C::kTripleStraight2));
  const int hi = m.rank, lo = hi - span + 1;
  return {{kPHtplHigh + (hi - 4), kPHtplLow + (lo - 3)}, 2};
}

// Composed logit for a concrete move = sum of its path-component logits.
inline float player_composed_logit(int move_id, const float *logits) {
  const PathLogits p = player_path_logits(move_id);
  float s = 0.0f;
  for (int i = 0; i < p.n; ++i) s += logits[p.idx[i]];
  return s;
}

}  // namespace az_search

#endif  // AZ_SEARCH_CONSIDERED_MOVES_H
