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

float EvalTable::query(uint32_t state_id, uint32_t fb_state_id, float kappa,
                        float fb_min_visits, float default_value) const {
  stats_.queries.fetch_add(1, std::memory_order_relaxed);

  // Resolve the fallback prior. fb_min_visits guards against using an
  // unreliable fallback estimate as the prior.
  float prior_value = default_value;
  bool have_fb = false;
  if (fallback_) {
    auto it_fb = fallback_->entries_.find(fb_state_id);
    if (it_fb != fallback_->entries_.end() &&
        it_fb->second.visit_count >= fb_min_visits) {
      prior_value = it_fb->second.total_wins / it_fb->second.visit_count;
      have_fb = true;
    }
  }

  auto it_main = entries_.find(state_id);
  if (it_main == entries_.end()) {
    if (have_fb) {
      stats_.fallback_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
      stats_.defaults.fetch_add(1, std::memory_order_relaxed);
    }
    return prior_value;
  }
  const float vc = it_main->second.visit_count;
  const float tw = it_main->second.total_wins;
  // Posterior mean of Beta(kappa*prior, kappa*(1-prior)) updated with
  // (tw, vc-tw) = (wins, losses).
  const float blended = (kappa * prior_value + tw) / (kappa + vc);
  // Classify which signal dominated for the stats counter.
  if (vc >= kappa) {
    stats_.main_hits.fetch_add(1, std::memory_order_relaxed);
  } else if (have_fb) {
    stats_.fallback_hits.fetch_add(1, std::memory_order_relaxed);
  } else {
    stats_.defaults.fetch_add(1, std::memory_order_relaxed);
  }
  return blended;
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
