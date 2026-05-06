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

struct PruneStatsSnapshot {
  std::uint64_t calls;
  std::uint64_t calls_multi;
  std::uint64_t calls_with_drop;
  std::uint64_t total_drops_single;
  std::uint64_t total_drops_bomb_aux;
  std::uint64_t total_drops_fh_aux;
  std::uint64_t total_input_moves;
};
PruneStatsSnapshot snapshot_prune_stats_thread_local();
void reset_prune_stats_thread_local();

// One run of the typed-Shannon search at a single decision point.
// Stateless w.r.t. prior calls — caller can persist via separate caching.
class TypedSearch {
public:
  // 3-tier eval chain (ext shrinks toward main shrinks toward fb).
  TypedSearch(const EvalTable &eval_extended, const EvalTable &eval_main,
              const EvalTable &eval_fallback,
              const MoveProbTable &mp_table, std::uint64_t rng_seed);

  struct Result {
    int move_id{-1};
    float value{0.0f};
    // Root-level per-move evaluations, sorted by value descending. Populated
    // by `run()`. Each entry is (move_id, expected-value).
    std::vector<std::pair<int, float>> top_moves;
    // Number of distinct OurNodes memoized during this search call.
    std::size_t nodes_searched{0};
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

  // Set true only during the very first visit_our call inside run(); cleared
  // before recursing. Used to capture root-level per-move evaluations.
  bool collect_root_{false};
  std::vector<std::pair<int, float>> root_evals_;

  float visit_our(OurNode &n);
  float visit_opp(const std::array<int, 13> &hand_after_our_move,
                  const std::array<int, 13> &discard_after_our_move,
                  int opp_count_unchanged, int our_move_id);

  float eval_leaf_we_passed(const std::array<int, 13> &hand,
                             const std::array<int, 13> &discard,
                             int opp_count) const;
  float eval_leaf_we_have_init(const std::array<int, 13> &hand,
                                const std::array<int, 13> &discard,
                                int opp_count);

  const EvalTable &eval_extended_;
  const EvalTable &eval_main_;
  const EvalTable &eval_fallback_;
  const MoveProbTable &mp_table_;
  std::mt19937_64 rng_;
  std::unordered_map<std::uint64_t, OurNode> memo_;
};

}  // namespace typed_search

#endif
