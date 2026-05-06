#include "move_grouping.h"

#include "move.h"
#include "util.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>

namespace typed_search {

namespace {

inline bool is_collapse_combo(Move::Combination c) {
  return c == Move::Combination::kBomb || c == Move::Combination::kFullHouse;
}

// Pre-bucket-key. Bombs collapse over auxiliary; full houses collapse over
// pair-rank; everything else keys by (combination, rank).
struct PreKey {
  int combo;
  int rank;
  bool operator==(const PreKey &o) const {
    return combo == o.combo && rank == o.rank;
  }
};
struct PreKeyHash {
  std::size_t operator()(const PreKey &k) const noexcept {
    return std::hash<std::int64_t>()(
        (static_cast<std::int64_t>(k.combo) << 8) | k.rank);
  }
};

// Returns the sorted ascending list of ranks at which our hand has a
// same-combination response. For bombs and full houses we return an empty list
// (those collapse to single-rank groups anyway, so merge-walk doesn't run).
std::vector<int>
our_response_ranks_for_combo(const std::array<int, 13> &hand,
                              Move::Combination c) {
  if (c == Move::Combination::kBomb || c == Move::Combination::kFullHouse) {
    return {};
  }
  // Use the precomputed legal-moves machinery as source of truth.
  Move pass(Move::Combination::kPass);
  auto legal = compute_legal_moves(hand, pass);
  std::vector<int> ranks;
  ranks.reserve(8);
  for (int mid : legal) {
    if (mid == kPASS) continue;
    Move m(mid);
    if (m.combination == c) {
      ranks.push_back(m.rank);
    }
  }
  std::sort(ranks.begin(), ranks.end());
  ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
  return ranks;
}

// True if there exists an our-response rank r with R_a < r <= R_b.
bool has_response_in_range(const std::vector<int> &sorted_ranks, int R_a,
                            int R_b) {
  // First rank > R_a:
  auto it = std::upper_bound(sorted_ranks.begin(), sorted_ranks.end(), R_a);
  if (it == sorted_ranks.end()) return false;
  return *it <= R_b;
}

}  // namespace

std::vector<MoveGroup> group_opp_moves(const std::array<int, 13> &our_hand,
                                        const std::vector<int> &opp_moves,
                                        const std::vector<float> &opp_probs) {
  std::vector<MoveGroup> out;
  if (opp_moves.empty()) return out;

  // Step 1: pre-bucket by (combination, rank) with collapsing for bomb/FH.
  // For each pre-bucket: list of (move_id, prob) pairs.
  struct Bucket {
    int combo;
    int rank;
    std::vector<int> moves;
    std::vector<float> probs;
  };
  std::unordered_map<PreKey, std::size_t, PreKeyHash> idx;
  std::vector<Bucket> buckets;
  // Reserve a special bucket for PASS (combo = -1 sentinel).
  std::size_t pass_idx = static_cast<std::size_t>(-1);

  for (std::size_t i = 0; i < opp_moves.size(); ++i) {
    int mid = opp_moves[i];
    float p = (i < opp_probs.size()) ? opp_probs[i] : 0.0f;
    if (mid == kPASS) {
      if (pass_idx == static_cast<std::size_t>(-1)) {
        pass_idx = buckets.size();
        buckets.push_back(Bucket{-1, -1, {}, {}});
      }
      buckets[pass_idx].moves.push_back(mid);
      buckets[pass_idx].probs.push_back(p);
      continue;
    }
    Move m(mid);
    PreKey k{static_cast<int>(m.combination), m.rank};
    auto it = idx.find(k);
    if (it == idx.end()) {
      it = idx.emplace(k, buckets.size()).first;
      buckets.push_back(Bucket{k.combo, k.rank, {}, {}});
    }
    Bucket &b = buckets[it->second];
    b.moves.push_back(mid);
    b.probs.push_back(p);
  }

  // Step 2: for each non-collapsing combination, walk ranks ascending and
  // merge consecutive ranks where our hand has no in-range same-combo response.
  // Group buckets by combination.
  std::unordered_map<int, std::vector<std::size_t>> by_combo;
  for (std::size_t i = 0; i < buckets.size(); ++i) {
    if (buckets[i].combo < 0) continue;  // PASS
    by_combo[buckets[i].combo].push_back(i);
  }

  for (auto &kv : by_combo) {
    auto &bucket_idxs = kv.second;
    Move::Combination c = static_cast<Move::Combination>(kv.first);
    if (is_collapse_combo(c)) {
      // Each bucket already collapsed to one rank; no further merging.
      continue;
    }
    std::sort(bucket_idxs.begin(), bucket_idxs.end(),
              [&](std::size_t a, std::size_t b) {
                return buckets[a].rank < buckets[b].rank;
              });
    auto our_ranks = our_response_ranks_for_combo(our_hand, c);

    // Merge: walk left-to-right; merge into previous if no in-range response
    // exists between previous group's max rank and current rank.
    std::vector<std::size_t> merged_lead;  // current "head" of each merged run
    int last_max_rank = -1;
    for (std::size_t bi : bucket_idxs) {
      int r = buckets[bi].rank;
      if (merged_lead.empty()) {
        merged_lead.push_back(bi);
        last_max_rank = r;
        continue;
      }
      if (!has_response_in_range(our_ranks, last_max_rank, r)) {
        // Merge into the previous bucket.
        Bucket &dst = buckets[merged_lead.back()];
        Bucket &src = buckets[bi];
        dst.moves.insert(dst.moves.end(), src.moves.begin(), src.moves.end());
        dst.probs.insert(dst.probs.end(), src.probs.begin(), src.probs.end());
        src.moves.clear();
        src.probs.clear();
        last_max_rank = r;  // new max within the merged run
      } else {
        merged_lead.push_back(bi);
        last_max_rank = r;
      }
    }
  }

  // Step 3: emit non-empty buckets as MoveGroups.
  out.reserve(buckets.size());
  for (auto &b : buckets) {
    if (b.moves.empty()) continue;
    MoveGroup g;
    g.moves = std::move(b.moves);
    g.probs = std::move(b.probs);
    out.push_back(std::move(g));
  }
  return out;
}

}  // namespace typed_search
