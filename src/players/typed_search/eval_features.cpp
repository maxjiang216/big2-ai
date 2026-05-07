#include "eval_features.h"

#include "move.h"
#include "util.h"

#include <algorithm>

namespace typed_search {

namespace {

// Hand-size bucket: [1-4][5-7][8-11][12-16] -> 0/1/2/3.
inline int size_bucket(int n) {
  if (n <= 4) return 0;
  if (n <= 7) return 1;
  if (n <= 11) return 2;
  return 3;
}

inline int hand_total(const std::array<int, 13> &h) {
  int s = 0;
  for (int c : h) s += c;
  return s;
}

inline bool has_bomb(const std::array<int, 13> &h) {
  if (h[11] == 3) return true;  // triple-A is a bomb
  for (int r = 0; r < 11; ++r) {
    if (h[r] == 4) return true;
  }
  return false;
}

// True if rank r is a bomb in our hand.
inline bool is_bomb_rank(const std::array<int, 13> &h, int r) {
  if (r == 11) return h[11] == 3;
  if (r >= 0 && r < 11) return h[r] == 4;
  return false;
}

// Build (at1, at2, at3) bitmasks from a count array, with bomb-rank bits
// cleared (to match the user's spec that bomb ranks are excluded from straight
// consideration).
inline void straight_input_bits(const std::array<int, 13> &h, uint16_t &at1,
                                 uint16_t &at2, uint16_t &at3) {
  at1 = at2 = at3 = 0;
  for (int r = 0; r < 13; ++r) {
    if (is_bomb_rank(h, r)) continue;
    if (h[r] >= 1) at1 |= static_cast<uint16_t>(1u << r);
    if (h[r] >= 2) at2 |= static_cast<uint16_t>(1u << r);
    if (h[r] >= 3) at3 |= static_cast<uint16_t>(1u << r);
  }
}

// Bucket: 0 = no straights, 1 = weak only (highest rank < 10),
//         2 = at least one strong (highest rank >= 10).
// Bomb-rank cards are excluded from the hand before checking.
int straight_tier(const std::array<int, 13> &h) {
  uint16_t a1, a2, a3;
  straight_input_bits(h, a1, a2, a3);
  thread_local std::vector<int> moves;
  straight_moves_for_pass_masks_into(a1, a2, a3, moves);
  bool weak = false;
  const auto &all = all_moves();
  for (int mid : moves) {
    const Move &m = all[mid];
    auto c = m.combination;
    if (c >= Move::Combination::kStraight5 && c <= Move::Combination::kStraight13) {
      // m.rank is the highest card's face value (3..K=13, A=14, 2=15).
      if (m.rank >= 10) {
        return 2;  // strong dominates; early exit
      }
      weak = true;
    }
  }
  return weak ? 1 : 0;
}

// Bucket: 0 = none, 1 = weak only (length-2 doubles, rank<10),
//         2 = strong present (anything else; longer DS or any TS).
int double_triple_straight_tier(const std::array<int, 13> &h) {
  uint16_t a1, a2, a3;
  straight_input_bits(h, a1, a2, a3);
  thread_local std::vector<int> moves;
  straight_moves_for_pass_masks_into(a1, a2, a3, moves);
  bool weak = false;
  const auto &all = all_moves();
  for (int mid : moves) {
    const Move &m = all[mid];
    auto c = m.combination;
    if (c >= Move::Combination::kTripleStraight2 &&
        c <= Move::Combination::kTripleStraight5) {
      return 2;
    }
    if (c == Move::Combination::kDoubleStraight2) {
      if (m.rank >= 10) return 2;
      weak = true;
    } else if (c >= Move::Combination::kDoubleStraight3 &&
               c <= Move::Combination::kDoubleStraight8) {
      return 2;
    }
  }
  return weak ? 1 : 0;
}

inline int bucket_count_3(int count) {
  if (count <= 0) return 0;
  if (count == 1) return 1;
  return 2;
}
inline int bucket_count_2(int count) { return count > 0 ? 1 : 0; }

}  // namespace

GuaranteedLargestThresholds compute_r_star(const std::array<int, 13> &hand,
                                            const std::array<int, 13> &discard,
                                            int opp_count) {
  GuaranteedLargestThresholds out;
  out.r_star[0] = -1;
  out.r_star[1] = -1;
  out.r_star[2] = -1;
  out.r_star[3] = -1;
  std::array<int, 13> opp_max{};
  for (int r = 0; r < 13; ++r) {
    int mx = max_cards_in_deck_for_rank(r) - hand[r] - discard[r];
    if (mx < 0) mx = 0;
    if (mx > opp_count) mx = opp_count;
    opp_max[r] = mx;
  }
  // For k=1: highest rank with opp_max>=1
  // For k=2: requires opp_count>=2 (otherwise no doubles possible)
  // For k=3: requires opp_count>=3
  for (int k = 1; k <= 3; ++k) {
    if (opp_count < k) {
      out.r_star[k] = -1;
      continue;
    }
    int best = -1;
    for (int r = 0; r < 13; ++r) {
      if (opp_max[r] >= k) best = r;
    }
    out.r_star[k] = best;
  }
  return out;
}

ImputedValue impute_terminal(const LeafContext &ctx) {
  const int our_total = hand_total(ctx.player_hand);
  if (our_total == 0) return {true, 1.0f};
  if (ctx.opp_count <= 0) return {true, 0.0f};
  // We have initiative AND can play all cards in one move.
  if (ctx.initiative == 0) {
    Move pass(Move::Combination::kPass);
    auto legal = compute_legal_moves(ctx.player_hand, pass);
    for (int mid : legal) {
      if (mid == kPASS) continue;
      Move m(mid);
      // m.numCards() = total cards used by this move (ignoring auxiliary kicker).
      // For full house: numCards = 5 (3+2). For aux bomb: 5 (4+1). All counted in MOVE_TO_CARDS.
      const auto &cost = MOVE_TO_CARDS[mid];
      if (cost[13] == our_total) return {true, 1.0f};
    }
  }
  return {false, 0.0f};
}

uint32_t main_state_id(const LeafContext &ctx) {
  const auto &h = ctx.player_hand;
  const int our_total = hand_total(h);

  const int init = ctx.initiative ? 1 : 0;
  const int opp_b = size_bucket(ctx.opp_count);
  const int our_b = size_bucket(our_total);
  const int hb = has_bomb(h) ? 1 : 0;
  const int st = straight_tier(h);
  const int ds = double_triple_straight_tier(h);

  GuaranteedLargestThresholds gl =
      compute_r_star(h, ctx.discard_pile, ctx.opp_count);

  // Singles bucketing: count by region after promoting "guaranteed largest" to top.
  int singles_small = 0;   // ranks 0..4 (3-7), excluding promoted
  int singles_medium = 0;  // ranks 5..7 (8-10)
  int singles_large = 0;   // ranks 8..10 (J-K)
  int singles_top = 0;     // ranks 11..12 (A-2) ∪ promoted
  for (int r = 0; r < 13; ++r) {
    if (h[r] != 1) continue;
    const bool promote = (r >= 11) || (r > gl.r_star[1]);
    if (promote) {
      ++singles_top;
    } else if (r <= 4) {
      ++singles_small;
    } else if (r <= 7) {
      ++singles_medium;
    } else {  // r in 8..10 (J-K), not promoted
      ++singles_large;
    }
  }

  // Doubles bucketing.
  // small=3-7 (0..4), medium=8-10 (5..7), large=J-A (8..11) ∪ promoted.
  int doubles_small = 0, doubles_medium = 0, doubles_large = 0;
  for (int r = 0; r < 12; ++r) {  // rank 12 (2) max 1, never a double
    if (h[r] != 2) continue;
    const bool promote = (r >= 8) || (r > gl.r_star[2]);
    if (promote) {
      ++doubles_large;
    } else if (r <= 4) {
      ++doubles_small;
    } else {  // r in 5..7
      ++doubles_medium;
    }
  }

  // Triples bucketing.
  // small=3-8 (0..5), large=9-K (6..10) ∪ promoted. A excluded (3-A is a bomb).
  int triples_small = 0, triples_large = 0;
  for (int r = 0; r < 11; ++r) {  // exclude A (11) and 2 (12)
    if (h[r] != 3) continue;
    const bool promote = (r >= 6) || (r > gl.r_star[3]);
    if (promote) {
      ++triples_large;
    } else {
      ++triples_small;
    }
  }

  // Mixed-radix pack. Boolean buckets clamp to {0,1}; small/medium etc. clamp to {0,1,2}.
  uint32_t s = 0;
  s = s * 2 + init;
  s = s * 4 + opp_b;
  s = s * 4 + our_b;
  s = s * 2 + hb;
  s = s * 3 + st;
  s = s * 3 + ds;
  s = s * 3 + bucket_count_3(singles_small);
  s = s * 2 + bucket_count_2(singles_medium);
  s = s * 2 + bucket_count_2(singles_large);
  s = s * 2 + bucket_count_2(singles_top);
  s = s * 2 + bucket_count_2(doubles_small);
  s = s * 2 + bucket_count_2(doubles_medium);
  s = s * 2 + bucket_count_2(doubles_large);
  s = s * 2 + bucket_count_2(triples_small);
  s = s * 2 + bucket_count_2(triples_large);
  return s;
}

uint32_t extended_state_id(const LeafContext &ctx) {
  const auto &h = ctx.player_hand;
  const auto &d = ctx.discard_pile;
  const int our_total = hand_total(h);

  const int init = ctx.initiative ? 1 : 0;
  const int opp_b = size_bucket(ctx.opp_count);
  const int our_b = size_bucket(our_total);
  const int hb = has_bomb(h) ? 1 : 0;

  GuaranteedLargestThresholds gl =
      compute_r_star(h, ctx.discard_pile, ctx.opp_count);

  // Singles: per-rank indicators for 3 (idx 0) and 4 (idx 1); regional buckets
  // for 5-7 (idx 2..4) / 8-10 (idx 5..7) / J-K (idx 8..10); plus a "top"
  // bucket = A-2 (idx 11..12) ∪ guaranteed-largest singles.
  int s_3 = 0, s_4 = 0, s_57 = 0, s_med = 0, s_lg = 0, s_top = 0;
  for (int r = 0; r < 13; ++r) {
    if (h[r] != 1) continue;
    const bool promote = (r >= 11) || (r > gl.r_star[1]);
    if (promote) ++s_top;
    else if (r == 0) ++s_3;
    else if (r == 1) ++s_4;
    else if (r <= 4) ++s_57;
    else if (r <= 7) ++s_med;
    else ++s_lg;
  }

  // Doubles: a single 0/1/2+ count over all rank indices that have count 2.
  int doubles = 0;
  for (int r = 0; r < 12; ++r) {  // rank 12 (2) max 1, never a double
    if (h[r] == 2) ++doubles;
  }

  // Triples: 2 regions (small=3-8, large=9-K) at radix 3 (0/1/2+). A excluded
  // (triple-A is a bomb), 2 can't form a triple.
  int trip_sm = 0, trip_lg = 0;
  for (int r = 0; r < 11; ++r) {
    if (h[r] != 3) continue;
    if (r <= 5) ++trip_sm;       // 3-8
    else ++trip_lg;              // 9-K (A and 2 excluded by loop bound)
  }

  // Opponent-aware features. opp_max[r] = max cards opp could still hold of
  // rank r, given our hand and the discard pile, capped by opp_count.
  auto opp_max_for = [&](int r) {
    int mx = max_cards_in_deck_for_rank(r) - h[r] - d[r];
    if (mx < 0) mx = 0;
    if (mx > ctx.opp_count) mx = ctx.opp_count;
    return mx;
  };
  // The 2 (rank 12) only has 1 copy in deck; opp_max is 0 or 1.
  const int opp_2_bit = (opp_max_for(12) > 0) ? 1 : 0;
  const int opp_A_max = opp_max_for(11);
  const int opp_Q_max = opp_max_for(9);
  const int opp_K_max = opp_max_for(10);
  const int opp_high_max = std::max(opp_Q_max, opp_K_max);

  auto b2 = [](int n) { return n > 0 ? 1 : 0; };
  auto b3 = [](int n) { return n <= 0 ? 0 : (n == 1 ? 1 : 2); };
  auto b4 = [](int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    if (n == 2) return 2;
    return 3;
  };

  uint32_t s = 0;
  s = s * 2 + init;
  s = s * 4 + opp_b;
  s = s * 4 + our_b;
  s = s * 2 + hb;
  s = s * 2 + b2(s_3);
  s = s * 2 + b2(s_4);
  s = s * 4 + b4(s_57);
  s = s * 3 + b3(s_med);
  s = s * 3 + b3(s_lg);
  s = s * 3 + b3(s_top);
  s = s * 3 + b3(doubles);
  s = s * 3 + b3(trip_sm);
  s = s * 3 + b3(trip_lg);
  s = s * 2 + opp_2_bit;
  s = s * 3 + b3(opp_A_max);
  s = s * 3 + b3(opp_high_max);
  return s;
}

uint32_t fallback_state_id(const LeafContext &ctx) {
  const auto &h = ctx.player_hand;
  const int our_total = hand_total(h);
  const int init = ctx.initiative ? 1 : 0;
  const int opp_b = size_bucket(ctx.opp_count);
  const int our_b = size_bucket(our_total);
  const int hb = has_bomb(h) ? 1 : 0;

  // Has straight: any straight, double straight, or triple straight (no
  // bomb-rank exclusion). Direct bitset lookup — no need to enumerate full
  // legal moves.
  HandBits hb_bits = hand_bits_from_counts(h);
  thread_local std::vector<int> straight_moves;
  straight_moves_for_pass_masks_into(hb_bits.at1, hb_bits.at2, hb_bits.at3,
                                       straight_moves);
  const int hs = straight_moves.empty() ? 0 : 1;
  const int has_two = h[12] >= 1 ? 1 : 0;
  const int has_ace = h[11] >= 1 ? 1 : 0;

  // Small singles: ranks 3-9 (index 0..6) with exactly 1 card.
  int singles_small = 0;
  for (int r = 0; r <= 6; ++r) {
    if (h[r] == 1) ++singles_small;
  }
  // Doubles: total ranks with exactly 2 cards.
  int doubles = 0;
  for (int r = 0; r < 12; ++r) {
    if (h[r] == 2) ++doubles;
  }
  // Triples: ranks with exactly 3 cards (excluding A, which is a bomb).
  int triples = 0;
  for (int r = 0; r < 11; ++r) {
    if (h[r] == 3) ++triples;
  }

  uint32_t s = 0;
  s = s * 2 + init;
  s = s * 4 + opp_b;
  s = s * 4 + our_b;
  s = s * 2 + hb;
  s = s * 2 + hs;
  s = s * 2 + has_two;
  s = s * 2 + has_ace;
  s = s * 3 + bucket_count_3(singles_small);
  s = s * 3 + bucket_count_3(doubles);
  s = s * 2 + bucket_count_2(triples);
  return s;
}

}  // namespace typed_search
