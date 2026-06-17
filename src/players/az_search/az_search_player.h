#ifndef AZ_SEARCH_AZ_SEARCH_PLAYER_H
#define AZ_SEARCH_AZ_SEARCH_PLAYER_H

// Player wrapper around the az_search core. The base Player already handles the
// tablebase root-skip (peek_tablebase_move) and keeps the PartialGame view in
// sync; this wrapper additionally tracks the full MOVE HISTORY (every applied
// move, ours and the opponent's — incl. tablebase root-skips via on_self_move)
// because the seq net conditions on it. Each turn it refreshes the evaluator's
// history prefix (KV cache) and re-roots the persistent search tree onto the
// real state via the move suffix. Torch-dependent (-DBIG2_WITH_TORCH).

#include "player.h"

#include "az_search.h"
#include "nn_eval.h"

#include <memory>
#include <vector>

namespace az_search {

class AzSearchPlayer : public ::Player {
public:
  AzSearchPlayer(std::shared_ptr<NNEvaluator> nn, SearchConfig cfg);

protected:
  void on_deal(const std::array<int, 13> &hand, int turn) override;
  void on_opponent_move(const Move &move) override;
  void on_self_move(const Move &move) override;
  Move select_move_impl() override;

private:
  SearchState current_state() const;

  std::shared_ptr<NNEvaluator> nn_;
  SearchConfig cfg_;
  std::unique_ptr<Search> search_;
  std::vector<int> history_;  // every applied move since the deal
};

}  // namespace az_search

#endif  // AZ_SEARCH_AZ_SEARCH_PLAYER_H
