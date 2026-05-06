#include "move_prob_disc_table.h"

#include "move.h"
#include "util.h"

#include <algorithm>
#include <cstdint>
#include <fstream>

namespace typed_search {

namespace {
// Bumped from MPDC: format now stores both counts and trials per cell.
constexpr char kMagic[4] = {'M', 'P', 'D', '2'};

inline int rank_idx_of_face(int face_rank) {
  return (face_rank == 2 || face_rank == 15) ? 12 : (face_rank - 3);
}
}  // namespace

bool MoveProbDiscardTable::is_eligible(int move_id) {
  // singles 3-K = move 1..11; doubles 3-K = 14..24; triples 3-Q = 26..35.
  return (move_id >= 1 && move_id <= 11) ||
          (move_id >= 14 && move_id <= 24) ||
          (move_id >= 26 && move_id <= 35);
}

std::uint16_t MoveProbDiscardTable::compute_bitmap(
    const std::array<int, 13> &hand_after,
    const std::array<int, 13> &discard_after, int opp_count) {
  std::uint16_t bits = 0;
  for (int r_idx = 1; r_idx <= 11; ++r_idx) {
    int max_in_deck = max_cards_in_deck_for_rank(r_idx);
    int opp_max = max_in_deck - hand_after[r_idx] - discard_after[r_idx];
    if (opp_max > opp_count) opp_max = opp_count;
    if (opp_max >= 3) bits |= static_cast<std::uint16_t>(1u << (r_idx - 1));
  }
  return bits;
}

std::uint16_t MoveProbDiscardTable::mask_relevant(int move_id,
                                                    std::uint16_t bitmap) {
  Move m(move_id);
  int rank_idx = rank_idx_of_face(m.rank);
  if (rank_idx >= kBitmapWidth) return 0;
  std::uint16_t mask =
      static_cast<std::uint16_t>(~((1u << rank_idx) - 1u)) & 0x7FFu;
  return static_cast<std::uint16_t>(bitmap & mask);
}

void MoveProbDiscardTable::load(const std::string &path) {
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
    std::uint32_t key = 0;
    in.read(reinterpret_cast<char *>(&key), 4);
    Entry e;
    in.read(reinterpret_cast<char *>(e.counts.data()),
            sizeof(float) * kNumMoves);
    in.read(reinterpret_cast<char *>(e.trials.data()),
            sizeof(float) * kNumMoves);
    if (!in) break;
    entries_[key] = std::move(e);
  }
}

void MoveProbDiscardTable::save(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  out.write(kMagic, 4);
  std::uint32_t n = static_cast<std::uint32_t>(entries_.size());
  out.write(reinterpret_cast<const char *>(&n), 4);
  for (const auto &kv : entries_) {
    std::uint32_t key = kv.first;
    out.write(reinterpret_cast<const char *>(&key), 4);
    out.write(reinterpret_cast<const char *>(kv.second.counts.data()),
              sizeof(float) * kNumMoves);
    out.write(reinterpret_cast<const char *>(kv.second.trials.data()),
              sizeof(float) * kNumMoves);
  }
}

void MoveProbDiscardTable::query(int move_id, int opp_count,
                                   int our_hand_size_bucket,
                                   std::uint16_t bitmap_full,
                                   const std::vector<int> &legal_moves,
                                   const MoveProbTable &mp_main,
                                   std::vector<float> &out_probs) const {
  // mp_main is the prior (already its own per-y shrunken distribution).
  mp_main.query(move_id, opp_count, our_hand_size_bucket, legal_moves,
                 out_probs);
  if (!is_eligible(move_id) || legal_moves.empty()) return;

  std::uint16_t rel = mask_relevant(move_id, bitmap_full);
  std::uint32_t key = make_key(move_id, opp_count, our_hand_size_bucket, rel);
  auto it = entries_.find(key);
  if (it == entries_.end()) return;

  // Per-response Bayesian shrinkage with this tier's data, using mp_main's
  // value as the prior. Each y has its own (counts[y], trials[y]) pair so
  // its denominator scales with how often y was actually deduced-feasible.
  constexpr float kKappa = 20.0f;
  double sum = 0.0;
  std::vector<float> posterior(legal_moves.size());
  for (std::size_t i = 0; i < legal_moves.size(); ++i) {
    int y = legal_moves[i];
    float c = it->second.counts[y];
    float t = it->second.trials[y];
    float val = (kKappa * out_probs[i] + c) / (kKappa + t);
    posterior[i] = val;
    sum += val;
  }
  if (sum > 0.0) {
    float inv = static_cast<float>(1.0 / sum);
    for (std::size_t i = 0; i < legal_moves.size(); ++i) {
      out_probs[i] = posterior[i] * inv;
    }
  }
}

void MoveProbDiscardTable::add_observation(int move_id, int opp_count,
                                             int our_hand_size_bucket,
                                             std::uint16_t bitmap_full,
                                             const std::vector<int> &feasible_set,
                                             int played_response,
                                             float weight) {
  if (!is_eligible(move_id)) return;
  std::uint16_t rel = mask_relevant(move_id, bitmap_full);
  std::uint32_t key = make_key(move_id, opp_count, our_hand_size_bucket, rel);
  Entry &e = entries_[key];
  for (int y : feasible_set) {
    if (y >= 0 && y < kNumMoves) e.trials[y] += weight;
  }
  if (played_response >= 0 && played_response < kNumMoves) {
    e.counts[played_response] += weight;
  }
}

void MoveProbDiscardTable::decay(float alpha) {
  for (auto &kv : entries_) {
    for (auto &c : kv.second.counts) c *= alpha;
    for (auto &t : kv.second.trials) t *= alpha;
  }
}

}  // namespace typed_search
