#ifndef AZ_SEARCH_AZ_SEARCH_PLAYER_H
#define AZ_SEARCH_AZ_SEARCH_PLAYER_H

// Player wrapper around the az_search core. The base Player already handles the
// tablebase root-skip (peek_tablebase_move) and keeps the PartialGame view in
// sync, so select_move_impl only runs the search. The search tree persists
// across this player's turns: each turn re-roots onto the current real state
// (advance_root), reusing the prior search via the persisted transposition memo.
// Leaf evaluation goes through a per-game CachingEvaluator wrapping the shared
// NN evaluator. Torch-dependent (-DBIG2_WITH_TORCH).

#include "player.h"

#include "az_search.h"
#include "eval_cache.h"
#include "nn_eval.h"

#include <memory>

namespace az_search {

class AzSearchPlayer : public ::Player {
public:
  AzSearchPlayer(std::shared_ptr<NNEvaluator> nn, SearchConfig cfg);

protected:
  void on_deal(const std::array<int, 13> &hand, int turn) override;
  Move select_move_impl() override;

private:
  SearchState current_state() const;

  std::shared_ptr<NNEvaluator> nn_;
  SearchConfig cfg_;
  std::unique_ptr<CachingEvaluator> cache_;
  std::unique_ptr<Search> search_;
};

}  // namespace az_search

#endif  // AZ_SEARCH_AZ_SEARCH_PLAYER_H
