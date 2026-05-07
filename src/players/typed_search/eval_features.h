#ifndef TYPED_SEARCH_EVAL_FEATURES_H
#define TYPED_SEARCH_EVAL_FEATURES_H

#include <array>
#include <cstdint>

namespace typed_search {

// Total state space sizes (used by tables for sizing / sanity).
constexpr uint32_t kMainStateCount = 442368u;
constexpr uint32_t kFallbackStateCount = 9216u;
// Extended state encoding. Drops the binary "has_straight" (Δ≈0.045 in the
// previous schema, mostly redundant with singles features). Reverts triples
// to a 2-region split (small=3-8, large=9-K) at radix 3, like in earlier
// generations. Adds three opponent-aware features that capture high-card
// threats: opp_2_bit (could opp hold a 2?), opp_A_count (max aces opp could
// hold, bucketed 0/1/2+), and opp_high (max(opp_max_Q, opp_max_K), 0/1/2+).
//
// Total: 2 * 4 * 4 * 2  * 2 * 2 * 4 * 3 * 3 * 3 * 3 * 3 * 3  * 2 * 3 * 3
//      = 13_436_928 (same as the previous schema by construction)
constexpr uint32_t kExtendedStateCount = 13436928u;

// Inputs needed to compute eval state at a leaf (just-passed) position.
// initiative: 0 = we have init (opp passed), 1 = opp has init (we passed).
struct LeafContext {
  std::array<int, 13> player_hand;
  std::array<int, 13> discard_pile;
  int opp_count;
  int initiative;
};

// If the position is trivially won/lost (terminal or one-move clear),
// returns {true, value}; else {false, _}.
struct ImputedValue {
  bool has_value;
  float value;
};
ImputedValue impute_terminal(const LeafContext &ctx);

// Compute main eval state ID. Returns uint32_t < kMainStateCount.
uint32_t main_state_id(const LeafContext &ctx);

// Compute fallback eval state ID. Returns uint32_t < kFallbackStateCount.
uint32_t fallback_state_id(const LeafContext &ctx);

// Compute extended eval state ID (finest tier). < kExtendedStateCount.
uint32_t extended_state_id(const LeafContext &ctx);

// Helper exposed for testing: compute, for size k in {1,2,3}, the highest
// rank index where the opponent could still hold ≥ k cards. -1 if none.
// k=2 returns -1 if opp_count < 2; k=3 returns -1 if opp_count < 3.
struct GuaranteedLargestThresholds {
  int r_star[4];  // index by k; r_star[0] unused
};
GuaranteedLargestThresholds compute_r_star(const std::array<int, 13> &hand,
                                            const std::array<int, 13> &discard,
                                            int opp_count);

}  // namespace typed_search

#endif
