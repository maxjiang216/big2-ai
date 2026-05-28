#ifndef AZ_SEARCH_EVAL_CACHE_H
#define AZ_SEARCH_EVAL_CACHE_H

// Out-of-tree NN-eval cache (lc0 NNCache idea): memoises evaluator outputs on
// the actual NN *input* rather than the full game state. Both nets take our
// exact hand as input (the opp net's per-move value head needs it), so both
// keys hash the full input including the hand. Persists across simulations and
// turns. Torch-free decorator around any Evaluator.

#include "evaluator.h"

#include <array>
#include <cstdint>
#include <unordered_map>

namespace az_search {

namespace detail {
inline uint64_t fnv_counts(uint64_t h, const std::array<int, 13> &c) {
  for (int v : c) {
    h ^= static_cast<uint64_t>(v & 0xFF);
    h *= 1099511628211ull;
  }
  return h;
}
inline uint64_t fnv_int(uint64_t h, int v) {
  h ^= static_cast<uint64_t>(v & 0xFFFF);
  h *= 1099511628211ull;
  return h;
}
}  // namespace detail

// 64-bit keys over the exact feature contents. Collisions are astronomically
// unlikely and would only swap one cached eval for another valid one.
inline uint64_t player_input_key(const PlayerFeatures &f) {
  uint64_t h = 1469598103934665603ull;
  h = detail::fnv_counts(h, f.hand);
  h = detail::fnv_counts(h, f.opp_max);
  h = detail::fnv_counts(h, f.trick);
  h = detail::fnv_int(h, f.opp_size);
  h = detail::fnv_int(h, f.our_size);
  return h;
}

inline uint64_t opp_input_key(const OppFeatures &f) {
  uint64_t h = 1469598103934665603ull;
  h = detail::fnv_counts(h, f.hand);
  h = detail::fnv_counts(h, f.opp_max);
  h = detail::fnv_counts(h, f.trick);
  h = detail::fnv_int(h, f.opp_size);
  h = detail::fnv_int(h, f.our_size);
  return h;
}

class CachingEvaluator : public Evaluator {
public:
  explicit CachingEvaluator(Evaluator &base) : base_(base) {}

  PlayerEval eval_player(const PlayerFeatures &f) override {
    const uint64_t k = player_input_key(f);
    auto it = pcache_.find(k);
    if (it != pcache_.end()) return it->second;
    ++player_misses_;
    PlayerEval e = base_.eval_player(f);
    pcache_.emplace(k, e);
    return e;
  }

  OppEval eval_opp(const OppFeatures &f) override {
    const uint64_t k = opp_input_key(f);
    auto it = ocache_.find(k);
    if (it != ocache_.end()) return it->second;
    ++opp_misses_;
    OppEval e = base_.eval_opp(f);
    ocache_.emplace(k, e);
    return e;
  }

  long player_misses() const { return player_misses_; }
  long opp_misses() const { return opp_misses_; }
  std::size_t size() const { return pcache_.size() + ocache_.size(); }

private:
  Evaluator &base_;
  std::unordered_map<uint64_t, PlayerEval> pcache_;
  std::unordered_map<uint64_t, OppEval> ocache_;
  long player_misses_ = 0, opp_misses_ = 0;
};

}  // namespace az_search

#endif  // AZ_SEARCH_EVAL_CACHE_H
