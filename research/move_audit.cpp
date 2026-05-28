// Move-set audit for az_search (build step 1).
//
// Enumerates all LEGAL_MOVES_SIZE engine move ids, prints the breakdown by
// combination type, reports the two AlphaZero head dimensions, and re-verifies
// the structural claims behind the TS5/DS8 special-casing (also asserted in
// test/test_az_search.cpp). Exits non-zero if any claim fails.
//
//   make move_audit && ./bin/move_audit

#include "az_search/considered_moves.h"
#include "move.h"
#include "util.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>

using az_search::AZ_OPP_HEAD_DIM;
using az_search::AZ_PLAYER_HEAD_DIM;

static const char *combo_name(Move::Combination c) {
  using C = Move::Combination;
  switch (c) {
  case C::kPass: return "Pass";
  case C::kSingle: return "Single";
  case C::kDouble: return "Double";
  case C::kTriple: return "Triple";
  case C::kFullHouse: return "FullHouse";
  case C::kBomb: return "Bomb";
  case C::kStraight5: return "Straight5";
  case C::kStraight6: return "Straight6";
  case C::kStraight7: return "Straight7";
  case C::kStraight8: return "Straight8";
  case C::kStraight9: return "Straight9";
  case C::kStraight10: return "Straight10";
  case C::kStraight11: return "Straight11";
  case C::kStraight12: return "Straight12";
  case C::kStraight13: return "Straight13";
  case C::kDoubleStraight2: return "Sisters2";
  case C::kDoubleStraight3: return "Sisters3";
  case C::kDoubleStraight4: return "Sisters4";
  case C::kDoubleStraight5: return "Sisters5";
  case C::kDoubleStraight6: return "Sisters6";
  case C::kDoubleStraight7: return "Sisters7";
  case C::kDoubleStraight8: return "Sisters8(DS8)";
  case C::kTripleStraight2: return "TripleStraight2";
  case C::kTripleStraight3: return "TripleStraight3";
  case C::kTripleStraight4: return "TripleStraight4";
  case C::kTripleStraight5: return "TripleStraight5(TS5)";
  }
  return "?";
}

int main() {
  const auto &moves = all_moves();

  // Breakdown by combination, in enum order of first appearance.
  std::printf("=== Move-set breakdown (%d engine ids) ===\n", LEGAL_MOVES_SIZE);
  std::printf("%-22s %6s\n", "Combination", "Count");
  Move::Combination cur = moves[0].combination;
  int run = 0, total = 0;
  auto flush = [&](Move::Combination c, int n) {
    std::printf("%-22s %6d\n", combo_name(c), n);
  };
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    if (moves[id].combination != cur) {
      flush(cur, run);
      cur = moves[id].combination;
      run = 0;
    }
    ++run;
    ++total;
  }
  flush(cur, run);
  std::printf("%-22s %6d\n", "TOTAL", total);

  std::printf("\n=== AlphaZero head dimensions ===\n");
  std::printf("player policy head : %d  (ids [0,%d) verbatim + 1 TS5 slot; DS8 dropped)\n",
              AZ_PLAYER_HEAD_DIM, az_search::kTS5RangeBegin);
  std::printf("opponent behavior  : %d  (+ 1 DS8 slot)\n", AZ_OPP_HEAD_DIM);

  std::printf("\n=== Collapse / drop decisions ===\n");
  std::printf("TS5 ids [%d,%d): %d ids -> 1 shared slot (both heads); each uses %d cards\n",
              az_search::kTS5RangeBegin, az_search::kTS5RangeEnd,
              az_search::kTS5RangeEnd - az_search::kTS5RangeBegin,
              MOVE_TO_CARDS[az_search::kTS5RangeBegin][13]);
  std::printf("DS8 ids [%d,%d): %d ids -> dropped (player) / 1 slot (opp); each uses %d cards (full hand)\n",
              az_search::kDS8RangeBegin, az_search::kDS8RangeEnd,
              az_search::kDS8RangeEnd - az_search::kDS8RangeBegin,
              MOVE_TO_CARDS[az_search::kDS8RangeBegin][13]);

  // Re-verify structural claims (mirrors test_az_search.cpp).
  int failures = 0;
  auto check = [&](bool ok, const char *what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
  };
  for (int id = az_search::kDS8RangeBegin; id < az_search::kDS8RangeEnd; ++id)
    check(MOVE_TO_CARDS[id][13] == 16, "DS8 uses full 16-card hand (auto-win)");
  for (int id = az_search::kTS5RangeBegin; id < az_search::kTS5RangeEnd; ++id)
    check(MOVE_TO_CARDS[id][13] == 15, "TS5 uses 15 cards");

  auto min_hand_for_pair = [&](int a, int b) {
    int t = 0;
    for (int r = 0; r < 13; ++r)
      t += std::max(MOVE_TO_CARDS[a][r], MOVE_TO_CARDS[b][r]);
    return t;
  };
  for (int a = az_search::kTS5RangeBegin; a < az_search::kTS5RangeEnd; ++a)
    for (int b = a + 1; b < az_search::kTS5RangeEnd; ++b)
      check(min_hand_for_pair(a, b) > 16, "at most one TS5 fits in 16 cards");
  for (int a = az_search::kDS8RangeBegin; a < az_search::kDS8RangeEnd; ++a)
    for (int b = a + 1; b < az_search::kDS8RangeEnd; ++b)
      check(min_hand_for_pair(a, b) > 16, "at most one DS8 fits in 16 cards");

  // Opponent head: flat index in range. Player head: factored — every id maps
  // to a valid family and 1-2 in-range path-logit indices.
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    int oi = az_search::az_opp_head_index(id);
    check(oi >= 0 && oi < AZ_OPP_HEAD_DIM, "opp index in range");
    int fam = az_search::player_family_id(id);
    check(fam >= az_search::kFamPass && fam <= az_search::kFamTplStraight,
          "player family in range");
    az_search::PathLogits p = az_search::player_path_logits(id);
    check(p.n == 1 || p.n == 2, "player path has 1-2 logits");
    for (int k = 0; k < p.n; ++k)
      check(p.idx[k] >= 0 && p.idx[k] < AZ_PLAYER_HEAD_DIM,
            "player path logit in range");
  }

  std::printf("\n%s\n", failures == 0 ? "Audit OK." : "AUDIT FAILED.");
  return failures == 0 ? 0 : 1;
}
