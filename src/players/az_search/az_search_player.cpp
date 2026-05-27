#include "az_search_player.h"

#include "move.h"

namespace az_search {

AzSearchPlayer::AzSearchPlayer(std::shared_ptr<NNEvaluator> nn, SearchConfig cfg)
    : nn_(std::move(nn)), cfg_(cfg) {}

void AzSearchPlayer::on_deal(const std::array<int, 13> & /*hand*/, int /*turn*/) {
  // New game: drop the persistent tree and the per-game NN-eval cache.
  search_.reset();
  cache_ = std::make_unique<CachingEvaluator>(*nn_);
}

SearchState AzSearchPlayer::current_state() const {
  SearchState s;
  s.our_hand = game_.player_hand();
  s.opp_size = game_.opponent_hand_size();
  s.discard = game_.discard_pile();
  s.last_move = encodeMove(game_.last_move());
  s.side = kUs;  // select_move_impl runs only on our turn
  return s;
}

Move AzSearchPlayer::select_move_impl() {
  if (!cache_) cache_ = std::make_unique<CachingEvaluator>(*nn_);
  const SearchState cur = current_state();
  if (!search_)
    search_ = std::make_unique<Search>(cur, cfg_);
  else
    search_->advance_root(cur);  // reuse prior search via the persisted memo
  search_->run(*cache_);
  return Move(search_->best_move());
}

}  // namespace az_search
