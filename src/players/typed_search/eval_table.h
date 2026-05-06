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

  // Returns win probability in [0, 1]. If neither table has enough visits at
  // (state_id), returns `default_value`. Increments stats_ counters.
  float query(uint32_t state_id, uint32_t fallback_state_id,
              float min_visits = 5.0f, float default_value = 0.5f) const;

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
