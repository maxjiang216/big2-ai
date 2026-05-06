#include "typed_search_player.h"

namespace typed_search {

TypedSearchPlayer::TypedSearchPlayer(
    std::shared_ptr<const TypedSearchTables> tables, std::uint64_t rng_seed)
    : tables_(std::move(tables)), rng_seed_(rng_seed) {}

Move TypedSearchPlayer::select_move_impl() {
  // Per-call RNG seed mix so search sampling is deterministic per (game, turn).
  const std::uint64_t seed =
      rng_seed_ ^ (0x9E3779B97F4A7C15ull * (++move_counter_));
  TypedSearch search(tables_->eval_extended, tables_->eval_main,
                      tables_->eval_fallback, tables_->mp_disc,
                      tables_->mp_main, seed);

  auto hand = game_.player_hand();
  auto discard = game_.discard_pile();
  int opp_count = game_.opponent_hand_size();
  Move last = game_.last_move();

  last_result_ = search.run(hand, discard, opp_count, last);
  last_used_search_ = true;
  if (last_result_.move_id < 0) {
    auto legal = game_.get_legal_moves();
    if (!legal.empty()) {
      return Move(legal.front());
    }
    return Move(Move::Combination::kPass);
  }
  return Move(last_result_.move_id);
}

}  // namespace typed_search
