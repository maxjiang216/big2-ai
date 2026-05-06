#ifndef TYPED_SEARCH_TYPED_SEARCH_PLAYER_H
#define TYPED_SEARCH_TYPED_SEARCH_PLAYER_H

#include "eval_table.h"
#include "move_prob_table.h"
#include "player.h"
#include "typed_search.h"

#include <cstdint>
#include <memory>

namespace typed_search {

// Bundles the four tables and shares them across player instances.
struct TypedSearchTables {
  EvalTable eval_main;
  EvalTable eval_fallback;
  MoveProbTable mp_main;
  MoveProbTable mp_fallback;

  TypedSearchTables() {
    mp_main.set_ignore_opp_count(false);
    mp_fallback.set_ignore_opp_count(true);
    eval_main.set_fallback(&eval_fallback);
    mp_main.set_fallback(&mp_fallback);
  }
};

class TypedSearchPlayer : public ::Player {
public:
  TypedSearchPlayer(std::shared_ptr<const TypedSearchTables> tables,
                    std::uint64_t rng_seed);

  // The most recent search result for this player. Empty when the move was
  // resolved by the tablebase fast-path (no search ran).
  const TypedSearch::Result &last_result() const { return last_result_; }
  bool last_used_search() const { return last_used_search_; }

protected:
  Move select_move_impl() override;

private:
  std::shared_ptr<const TypedSearchTables> tables_;
  std::uint64_t rng_seed_;
  std::uint64_t move_counter_{0};
  TypedSearch::Result last_result_;
  bool last_used_search_{false};
};

}  // namespace typed_search

#endif
