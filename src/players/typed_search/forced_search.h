#ifndef TYPED_SEARCH_FORCED_SEARCH_H
#define TYPED_SEARCH_FORCED_SEARCH_H

#include <array>

namespace typed_search {

class EvalTable;

// At a "we hold initiative" position, search over sequences of moves that the
// opponent provably cannot beat. At every visited node — including
// non-terminal ones where forced moves remain unplayed — query the eval table
// and remember the value. Returns the max of:
//   - eval at the current node,
//   - eval at any reachable forced-line node,
//   - 1.0 if any line empties our hand.
//
// `default_value` is returned by query() when no eval entry is found.
// `depth_cap` limits the forced-move recursion depth (kept small; the forced-
// move tree is naturally narrow).
//
// Uses the same 3-tier chained eval (extended -> main -> fallback) as the
// search itself, so internal forced-line evaluations are consistent.
float forced_search_value(const std::array<int, 13> &hand,
                          const std::array<int, 13> &discard,
                          int opp_count,
                          const EvalTable &eval_extended,
                          const EvalTable &eval_main,
                          const EvalTable &eval_fallback,
                          float default_value = 0.5f,
                          int depth_cap = 4);

}  // namespace typed_search

#endif
