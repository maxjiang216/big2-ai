#include "move_prob_table.h"

#include <cstdint>
#include <fstream>

namespace typed_search {

namespace {
constexpr char kMagicMain[4] = {'M', 'P', 'R', 'B'};
constexpr char kMagicFallback[4] = {'M', 'P', 'R', 'F'};
}  // namespace

void MoveProbTable::load(const std::string &path) {
  entries_.clear();
  std::ifstream in(path, std::ios::binary);
  if (!in) return;

  char magic[4];
  in.read(magic, 4);
  if (in.gcount() != 4) return;
  const char (&want)[4] = ignore_opp_count_ ? kMagicFallback : kMagicMain;
  if (magic[0] != want[0] || magic[1] != want[1] || magic[2] != want[2] ||
      magic[3] != want[3]) {
    return;
  }
  std::uint32_t n = 0;
  in.read(reinterpret_cast<char *>(&n), 4);
  if (!in) return;

  entries_.reserve(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint16_t player_move = 0;
    std::uint8_t opp_count = 0, our_b = 0;
    in.read(reinterpret_cast<char *>(&player_move), 2);
    in.read(reinterpret_cast<char *>(&opp_count), 1);
    in.read(reinterpret_cast<char *>(&our_b), 1);
    Entry e;
    in.read(reinterpret_cast<char *>(e.counts.data()),
            sizeof(float) * kNumMoves);
    if (!in) break;
    std::uint32_t key = make_key(static_cast<int>(player_move),
                                  static_cast<int>(opp_count),
                                  static_cast<int>(our_b));
    entries_[key] = std::move(e);
  }
}

void MoveProbTable::save(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  const char (&want)[4] = ignore_opp_count_ ? kMagicFallback : kMagicMain;
  out.write(want, 4);
  std::uint32_t n = static_cast<std::uint32_t>(entries_.size());
  out.write(reinterpret_cast<const char *>(&n), 4);
  for (const auto &kv : entries_) {
    std::uint16_t player_move;
    std::uint8_t opp_count = 0;
    std::uint8_t our_b = 0;
    if (ignore_opp_count_) {
      player_move = static_cast<std::uint16_t>(kv.first);
    } else {
      std::uint32_t k = kv.first;
      our_b = static_cast<std::uint8_t>(k % kOurBuckets);
      k /= kOurBuckets;
      opp_count = static_cast<std::uint8_t>(k % 17u);
      player_move = static_cast<std::uint16_t>(k / 17u);
    }
    out.write(reinterpret_cast<const char *>(&player_move), 2);
    out.write(reinterpret_cast<const char *>(&opp_count), 1);
    out.write(reinterpret_cast<const char *>(&our_b), 1);
    out.write(reinterpret_cast<const char *>(kv.second.counts.data()),
              sizeof(float) * kNumMoves);
  }
}

void MoveProbTable::query(int player_move, int opp_count,
                           int our_hand_size_bucket,
                           const std::vector<int> &legal_moves,
                           std::vector<float> &out_probs) const {
  stats_.queries.fetch_add(1, std::memory_order_relaxed);
  out_probs.assign(legal_moves.size(), 0.0f);
  if (legal_moves.empty()) return;

  // Try this table first.
  std::uint32_t key = make_key(player_move, opp_count, our_hand_size_bucket);
  auto it = entries_.find(key);
  double sum = 0.0;
  if (it != entries_.end()) {
    for (std::size_t i = 0; i < legal_moves.size(); ++i) {
      float c = it->second.counts[legal_moves[i]];
      out_probs[i] = c;
      sum += c;
    }
  }
  bool from_main = sum > 0.0;
  if (sum <= 0.0 && fallback_) {
    std::uint32_t fk = fallback_->make_key(player_move, opp_count,
                                             our_hand_size_bucket);
    auto it2 = fallback_->entries_.find(fk);
    if (it2 != fallback_->entries_.end()) {
      for (std::size_t i = 0; i < legal_moves.size(); ++i) {
        float c = it2->second.counts[legal_moves[i]];
        out_probs[i] = c;
        sum += c;
      }
    }
  }
  if (sum <= 0.0) {
    stats_.uniform.fetch_add(1, std::memory_order_relaxed);
    float u = 1.0f / static_cast<float>(legal_moves.size());
    for (auto &p : out_probs) p = u;
    return;
  }
  if (from_main) {
    stats_.main_hits.fetch_add(1, std::memory_order_relaxed);
  } else {
    stats_.fallback_hits.fetch_add(1, std::memory_order_relaxed);
  }
  float inv = static_cast<float>(1.0 / sum);
  for (auto &p : out_probs) p *= inv;
}

void MoveProbTable::add_observation(int player_move, int opp_count,
                                     int our_hand_size_bucket, int opp_move,
                                     float weight) {
  std::uint32_t key = make_key(player_move, opp_count, our_hand_size_bucket);
  Entry &e = entries_[key];
  if (opp_move >= 0 && opp_move < kNumMoves) {
    e.counts[opp_move] += weight;
  }
}

void MoveProbTable::decay(float alpha) {
  for (auto &kv : entries_) {
    for (auto &c : kv.second.counts) c *= alpha;
  }
}

}  // namespace typed_search
