#include "az_search_player.h"

#include "move.h"

namespace az_search {

AzSearchPlayer::AzSearchPlayer(std::shared_ptr<NNEvaluator> nn, SearchConfig cfg)
    : nn_(std::move(nn)), cfg_(cfg) {}

void AzSearchPlayer::on_deal(const std::array<int, 13> & /*hand*/, int /*turn*/) {
  search_.reset();  // new game: drop the persistent tree + history
  history_.clear();
}

void AzSearchPlayer::on_opponent_move(const Move &move) {
  history_.push_back(encodeMove(move));
}

void AzSearchPlayer::on_self_move(const Move &move) {
  history_.push_back(encodeMove(move));
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
  const SearchState cur = current_state();
  // Refresh the evaluator's prefix (one encode per real turn; all of this
  // turn's leaf evals attend to the cached KV). The shared evaluator's slot 0
  // is safe across the two seats of a single-threaded game: each rewrites the
  // prefix at the start of its own turn.
  nn_->set_prefix(history_);
  if (!search_)
    search_ = std::make_unique<Search>(cur, history_, cfg_);
  else
    search_->advance_root(cur, history_);  // suffix-walk subtree reuse
  search_->run(*nn_);
  return Move(search_->best_move());
}

}  // namespace az_search
