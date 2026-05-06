#include "move_prob_table.h"

#include <cstdint>
#include <fstream>

namespace typed_search {

namespace {
// Bumped from MPRB / MPRF: format now stores both counts and trials per cell.
constexpr char kMagicMain[4] = {'M', 'P', 'B', '2'};
constexpr char kMagicFallback[4] = {'M', 'P', 'F', '2'};
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
    in.read(reinterpret_cast<char *>(e.trials.data()),
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
    out.write(reinterpret_cast<const char *>(kv.second.trials.data()),
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

  constexpr float kKappa = 20.0f;
  const std::size_t N = legal_moves.size();

  // Step 1: build the prior. If there's a fallback table, recurse for it;
  // otherwise the prior is uniform 1/N over the candidate set.
  std::vector<float> prior(N, 1.0f / static_cast<float>(N));
  if (fallback_) {
    fallback_->query(player_move, opp_count, our_hand_size_bucket, legal_moves,
                      prior);
  }

  // Step 2: look up this tier's entry.
  std::uint32_t key = make_key(player_move, opp_count, our_hand_size_bucket);
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    out_probs = prior;
    if (fallback_) {
      stats_.fallback_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
      stats_.uniform.fetch_add(1, std::memory_order_relaxed);
    }
    return;
  }

  // Step 3: per-response Bayesian shrinkage. P(y | y feasible) is estimated
  // as (kappa * prior_y + counts[y]) / (kappa + trials[y]). Then renormalize
  // over the candidate set to get a valid distribution.
  double sum = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    int y = legal_moves[i];
    float c = it->second.counts[y];
    float t = it->second.trials[y];
    float val = (kKappa * prior[i] + c) / (kKappa + t);
    out_probs[i] = val;
    sum += val;
  }
  if (sum > 0.0) {
    float inv = static_cast<float>(1.0 / sum);
    for (auto &p : out_probs) p *= inv;
    stats_.main_hits.fetch_add(1, std::memory_order_relaxed);
  } else {
    out_probs = prior;
    if (fallback_) {
      stats_.fallback_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
      stats_.uniform.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

void MoveProbTable::add_observation(int player_move, int opp_count,
                                     int our_hand_size_bucket,
                                     const std::vector<int> &feasible_set,
                                     int played_response, float weight) {
  std::uint32_t key = make_key(player_move, opp_count, our_hand_size_bucket);
  Entry &e = entries_[key];
  for (int y : feasible_set) {
    if (y >= 0 && y < kNumMoves) e.trials[y] += weight;
  }
  if (played_response >= 0 && played_response < kNumMoves) {
    e.counts[played_response] += weight;
  }
}

void MoveProbTable::decay(float alpha) {
  for (auto &kv : entries_) {
    for (auto &c : kv.second.counts) c *= alpha;
    for (auto &t : kv.second.trials) t *= alpha;
  }
}

}  // namespace typed_search
