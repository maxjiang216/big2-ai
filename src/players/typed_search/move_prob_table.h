#ifndef TYPED_SEARCH_MOVE_PROB_TABLE_H
#define TYPED_SEARCH_MOVE_PROB_TABLE_H

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace typed_search {

struct MoveProbQueryStats {
  std::atomic<std::uint64_t> queries{0};
  std::atomic<std::uint64_t> main_hits{0};      // main had nonzero mass
  std::atomic<std::uint64_t> fallback_hits{0};  // main empty, fallback had mass
  std::atomic<std::uint64_t> uniform{0};        // both empty
  void reset() {
    queries = 0;
    main_hits = 0;
    fallback_hits = 0;
    uniform = 0;
  }
};

// Tabular distribution over opponent's response moves, conditioned on:
//   key = (player_move_id, opp_count, our_hand_size_bucket)  [main table]
//   key = (player_move_id)                                    [fallback table]
//
// our_hand_size_bucket: 0 = [1-4], 1 = [5-7], 2 = [8-11], 3 = [12-16].
//
// Storage: counts[468] per visited state. Decay = scale all counts by alpha.
// Query returns a normalized distribution over a caller-supplied list of
// candidate response move IDs (legal_moves), with uniform fallback when
// nothing has been observed.
class MoveProbTable {
public:
  static constexpr int kNumMoves = 468;
  static constexpr int kOurBuckets = 4;

  // Per-cell, per-response stats. counts[y] = # times opp played y when y was
  // a deduced-feasible response. trials[y] = # times y was deduced-feasible
  // (whether or not opp picked it). The estimator P(y | y feasible) is then
  // shrunken Bayesian: (kappa * prior + counts[y]) / (kappa + trials[y]).
  struct Entry {
    std::array<float, kNumMoves> counts{};
    std::array<float, kNumMoves> trials{};
  };

  MoveProbTable() = default;

  // ignore_opp_count = true for fallback table.
  void set_ignore_opp_count(bool b) { ignore_opp_count_ = b; }
  bool ignore_opp_count() const { return ignore_opp_count_; }

  void load(const std::string &path);
  void save(const std::string &path) const;

  void set_fallback(const MoveProbTable *fb) { fallback_ = fb; }

  // Fill out_probs (parallel to legal_moves) with a normalized distribution.
  // If neither this nor fallback has any mass for the state-and-legal subset,
  // returns uniform.
  void query(int player_move, int opp_count, int our_hand_size_bucket,
             const std::vector<int> &legal_moves,
             std::vector<float> &out_probs) const;

  MoveProbQueryStats &stats() const { return stats_; }

  // Training. For each response in feasible_set, increment trials by weight.
  // For played_response (which must also be in feasible_set), increment its
  // count by weight.
  void add_observation(int player_move, int opp_count,
                       int our_hand_size_bucket,
                       const std::vector<int> &feasible_set,
                       int played_response, float weight = 1.0f);
  void decay(float alpha);

  std::size_t size() const { return entries_.size(); }

  static int size_bucket(int n) {
    if (n <= 4) return 0;
    if (n <= 7) return 1;
    if (n <= 11) return 2;
    return 3;
  }

private:
  std::uint32_t make_key(int player_move, int opp_count,
                          int our_hand_size_bucket) const {
    if (ignore_opp_count_) return static_cast<std::uint32_t>(player_move);
    return ((static_cast<std::uint32_t>(player_move) * 17u +
              static_cast<std::uint32_t>(opp_count)) *
              kOurBuckets) +
            static_cast<std::uint32_t>(our_hand_size_bucket);
  }

  std::unordered_map<std::uint32_t, Entry> entries_;
  bool ignore_opp_count_{false};
  const MoveProbTable *fallback_{nullptr};
  mutable MoveProbQueryStats stats_;
};

}  // namespace typed_search

#endif
