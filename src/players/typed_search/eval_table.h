#ifndef TYPED_SEARCH_EVAL_TABLE_H
#define TYPED_SEARCH_EVAL_TABLE_H

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace typed_search {

struct EvalQueryStats {
  std::atomic<std::uint64_t> queries{0};
  std::atomic<std::uint64_t> main_hits{0};
  std::atomic<std::uint64_t> fallback_hits{0};
  std::atomic<std::uint64_t> defaults{0};
  void reset() {
    queries = 0;
    main_hits = 0;
    fallback_hits = 0;
    defaults = 0;
  }
};

// Tabular leaf-position evaluator. Store running totals so that decay between
// generations is just `entry *= alpha`. Probability = total_wins / visit_count.
//
// Two-level fallback: this table -> fallback (set via set_fallback) -> default.
// "Use this entry" requires visit_count >= min_visits at query time.
class EvalTable {
public:
  struct Entry {
    float total_wins{0.0f};
    float visit_count{0.0f};
  };

  EvalTable() = default;

  // Reads binary file written by save(). Silently no-ops if path doesn't exist.
  void load(const std::string &path);
  void save(const std::string &path) const;

  void set_fallback(const EvalTable *fb) { fallback_ = fb; }

  // Bayesian-shrinkage query.
  //
  //   value = (kappa * prior_value + main_total_wins) / (kappa + main_visits)
  //
  // where prior_value is the fallback table's win-probability for
  // `fallback_state_id` (or `default_value` if the fallback entry has fewer
  // than `fb_min_visits` observations). `kappa` is the equivalent-visit weight
  // of the prior. With kappa=20: 20 main visits gives a 50/50 blend with the
  // fallback prior; 100 main visits is ~83% main / 17% fallback.
  //
  // When the main entry is missing entirely we return prior_value directly.
  // Stats counters classify the call by which path dominated the outcome.
  float query(uint32_t state_id, uint32_t fallback_state_id,
              float kappa = 20.0f, float fb_min_visits = 5.0f,
              float default_value = 0.5f) const;

  // Raw lookup — returns whether the entry exists, its win_prob, and its
  // visit_count. No shrinkage, no fallback. Used by external callers to
  // build multi-tier shrinkage chains.
  struct LookupResult {
    bool found{false};
    float win_prob{0.0f};
    float visit_count{0.0f};
  };
  LookupResult lookup(uint32_t state_id) const;

  // Apply Beta-Binomial shrinkage with `prior_value` as the prior and
  // (win_prob, visit_count) as the data. With kappa=20 and visit_count=20,
  // the result is a 50/50 blend of prior and data.
  static float shrink(float prior_value, float win_prob, float visit_count,
                       float kappa) {
    return (kappa * prior_value + win_prob * visit_count) /
            (kappa + visit_count);
  }

  EvalQueryStats &stats() const { return stats_; }

  // Training updates.
  void add_observation(uint32_t state_id, float winner, float weight = 1.0f);
  void decay(float alpha);

  std::size_t size() const { return entries_.size(); }
  const std::unordered_map<uint32_t, Entry> &entries() const { return entries_; }

private:
  std::unordered_map<uint32_t, Entry> entries_;
  const EvalTable *fallback_{nullptr};
  mutable EvalQueryStats stats_;
};

}  // namespace typed_search

#endif
