#include "eval_table.h"

#include <cstdint>
#include <fstream>

namespace typed_search {

namespace {
constexpr char kMagic[4] = {'E', 'V', 'A', 'L'};
}  // namespace

void EvalTable::load(const std::string &path) {
  entries_.clear();
  std::ifstream in(path, std::ios::binary);
  if (!in) return;

  char magic[4];
  in.read(magic, 4);
  if (in.gcount() != 4 || magic[0] != kMagic[0] || magic[1] != kMagic[1] ||
      magic[2] != kMagic[2] || magic[3] != kMagic[3]) {
    return;
  }
  std::uint32_t n = 0;
  in.read(reinterpret_cast<char *>(&n), 4);
  if (!in) return;

  entries_.reserve(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint32_t sid = 0;
    float total_wins = 0.0f, visit_count = 0.0f;
    in.read(reinterpret_cast<char *>(&sid), 4);
    in.read(reinterpret_cast<char *>(&total_wins), 4);
    in.read(reinterpret_cast<char *>(&visit_count), 4);
    if (!in) break;
    entries_[sid] = Entry{total_wins, visit_count};
  }
}

void EvalTable::save(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  out.write(kMagic, 4);
  std::uint32_t n = static_cast<std::uint32_t>(entries_.size());
  out.write(reinterpret_cast<const char *>(&n), 4);
  for (const auto &kv : entries_) {
    std::uint32_t sid = kv.first;
    out.write(reinterpret_cast<const char *>(&sid), 4);
    out.write(reinterpret_cast<const char *>(&kv.second.total_wins), 4);
    out.write(reinterpret_cast<const char *>(&kv.second.visit_count), 4);
  }
}

float EvalTable::query(uint32_t state_id, uint32_t fb_state_id, float min_visits,
                        float default_value) const {
  stats_.queries.fetch_add(1, std::memory_order_relaxed);
  auto it = entries_.find(state_id);
  if (it != entries_.end() && it->second.visit_count >= min_visits) {
    stats_.main_hits.fetch_add(1, std::memory_order_relaxed);
    return it->second.total_wins / it->second.visit_count;
  }
  if (fallback_) {
    auto it2 = fallback_->entries_.find(fb_state_id);
    if (it2 != fallback_->entries_.end() &&
        it2->second.visit_count >= min_visits) {
      stats_.fallback_hits.fetch_add(1, std::memory_order_relaxed);
      return it2->second.total_wins / it2->second.visit_count;
    }
  }
  stats_.defaults.fetch_add(1, std::memory_order_relaxed);
  return default_value;
}

void EvalTable::add_observation(uint32_t state_id, float winner, float weight) {
  Entry &e = entries_[state_id];
  e.total_wins += winner * weight;
  e.visit_count += weight;
}

void EvalTable::decay(float alpha) {
  for (auto &kv : entries_) {
    kv.second.total_wins *= alpha;
    kv.second.visit_count *= alpha;
  }
}

}  // namespace typed_search
