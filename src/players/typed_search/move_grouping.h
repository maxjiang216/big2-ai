#ifndef TYPED_SEARCH_MOVE_GROUPING_H
#define TYPED_SEARCH_MOVE_GROUPING_H

#include <array>
#include <vector>

namespace typed_search {

struct MoveGroup {
  // Opponent move IDs in this group.
  std::vector<int> moves;
  // Parallel array of per-move probabilities (relative weights). The sum is
  // the group's edge weight in the search.
  std::vector<float> probs;
};

// Group opponent moves into response-equivalence classes from our hand's POV.
//
// - Bombs of the same rank collapse into one group regardless of auxiliary.
// - Full houses of the same triple-rank collapse regardless of pair rank.
// - For other combinations (single / double / triple / each straight length
//   / each double-straight length / each triple-straight length), within a
//   given combination, two opponent moves at ranks R_a < R_b merge iff our
//   hand has no SAME-combination response at any rank r with R_a < r ≤ R_b.
//   (Our bombs are ignored: they beat both ranks equally.)
// - PASS, if present in opp_moves, becomes its own singleton group.
//
// `opp_moves` and `opp_probs` are parallel; total probability is preserved.
std::vector<MoveGroup> group_opp_moves(const std::array<int, 13> &our_hand,
                                        const std::vector<int> &opp_moves,
                                        const std::vector<float> &opp_probs);

// _into variant: writes into a caller-provided output vector. Callers in the
// search hot path use this with a thread-local scratch buffer to avoid the
// allocation traffic of returning by value.
void group_opp_moves_into(const std::array<int, 13> &our_hand,
                          const std::vector<int> &opp_moves,
                          const std::vector<float> &opp_probs,
                          std::vector<MoveGroup> &out);

}  // namespace typed_search

#endif
