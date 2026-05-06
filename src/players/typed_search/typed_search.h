#ifndef TYPED_SEARCH_TYPED_SEARCH_H
#define TYPED_SEARCH_TYPED_SEARCH_H

#include "move.h"

#include <array>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace typed_search {

class EvalTable;
class MoveProbTable;

// One run of the typed-Shannon search at a single decision point.
// Stateless w.r.t. prior calls — caller can persist via separate caching.
class TypedSearch {
public:
  TypedSearch(const EvalTable &eval_table, const MoveProbTable &mp_table,
              std::uint64_t rng_seed);

  struct Result {
    int move_id{-1};
    float value{0.0f};
  };

  // our_hand, discard, opp_count, last_move describe the position at our turn.
  Result run(const std::array<int, 13> &our_hand,
             const std::array<int, 13> &discard, int opp_count,
             const Move &last_move);

  std::size_t memo_size() const { return memo_.size(); }

private:
  struct OurNode {
    std::array<int, 13> hand{};
    std::array<int, 13> discard{};
    int opp_count{0};
    Move last_move{Move::Combination::kPass};
    bool computed{false};
    float value{0.0f};
    int best_move{-1};
  };

  std::uint64_t make_key(const std::array<int, 13> &hand, int opp_count,
                         const Move &last_move) const;

  float visit_our(OurNode &n);
  float visit_opp(const std::array<int, 13> &hand_after_our_move,
                  const std::array<int, 13> &discard_after_our_move,
                  int opp_count_unchanged, int our_move_id);

  float eval_leaf_we_passed(const std::array<int, 13> &hand,
                             const std::array<int, 13> &discard,
                             int opp_count) const;
  float eval_leaf_we_have_init(const std::array<int, 13> &hand,
                                const std::array<int, 13> &discard,
                                int opp_count) const;

  const EvalTable &eval_table_;
  const MoveProbTable &mp_table_;
  std::mt19937_64 rng_;
  std::unordered_map<std::uint64_t, OurNode> memo_;
};

}  // namespace typed_search

#endif
