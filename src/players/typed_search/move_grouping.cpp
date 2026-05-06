#include "move_grouping.h"

#include "move.h"
#include "util.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace typed_search {

namespace {

inline bool is_collapse_combo(Move::Combination c) {
  return c == Move::Combination::kBomb || c == Move::Combination::kFullHouse;
}

struct OppMoveInfo {
  int combo;       // (int)Move::Combination, or -1 for PASS
  int rank;
  int orig_idx;    // index back into opp_moves / opp_probs
};

// Compute the sorted ascending list of ranks at which our hand has a same-
// combination response. Writes into `out` (caller owns the storage; it's
// cleared first). For bomb/full-house we leave it empty since those collapse
// to single-rank groups before any merge-walk would run.
void our_response_ranks_for_combo_into(const std::array<int, 13> &hand,
                                        Move::Combination c,
                                        std::vector<int> &out) {
  out.clear();
  if (c == Move::Combination::kBomb || c == Move::Combination::kFullHouse) {
    return;
  }
  Move pass(Move::Combination::kPass);
  auto legal = compute_legal_moves(hand, pass);
  out.reserve(8);
  for (int mid : legal) {
    if (mid == kPASS) continue;
    Move m(mid);
    if (m.combination == c) out.push_back(m.rank);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
}

// True if there exists an our-response rank r with R_a < r <= R_b.
inline bool has_response_in_range(const std::vector<int> &sorted_ranks,
                                    int R_a, int R_b) {
  auto it = std::upper_bound(sorted_ranks.begin(), sorted_ranks.end(), R_a);
  if (it == sorted_ranks.end()) return false;
  return *it <= R_b;
}

// Per-thread scratch buffers for the inner allocations of group_opp_moves_into.
struct GroupingScratch {
  std::vector<OppMoveInfo> infos;
  std::vector<int> our_ranks;
};
thread_local GroupingScratch g_grouping_scratch;

}  // namespace

void group_opp_moves_into(const std::array<int, 13> &our_hand,
                           const std::vector<int> &opp_moves,
                           const std::vector<float> &opp_probs,
                           std::vector<MoveGroup> &out) {
  // Preserve inner-vector capacity in recycled MoveGroup slots; clear sizes.
  for (auto &g : out) {
    g.moves.clear();
    g.probs.clear();
  }
  std::size_t used = 0;  // number of valid groups in `out`

  auto get_slot = [&]() -> MoveGroup & {
    if (used < out.size()) {
      return out[used++];
    }
    out.emplace_back();
    ++used;
    return out.back();
  };

  if (!opp_moves.empty()) {
    // Step 1: build (combo, rank, orig_idx) infos. Reuse thread-local buffer.
    auto &infos = g_grouping_scratch.infos;
    infos.clear();
    infos.reserve(opp_moves.size());
    const auto &all = all_moves();
    for (std::size_t i = 0; i < opp_moves.size(); ++i) {
      int mid = opp_moves[i];
      if (mid == kPASS) {
        infos.push_back({-1, -1, static_cast<int>(i)});
      } else {
        const Move &m = all[mid];
        infos.push_back({static_cast<int>(m.combination), m.rank,
                          static_cast<int>(i)});
      }
    }

    // Step 2: sort by (combo, rank). Stable not required.
    std::sort(infos.begin(), infos.end(),
              [](const OppMoveInfo &a, const OppMoveInfo &b) {
                if (a.combo != b.combo) return a.combo < b.combo;
                return a.rank < b.rank;
              });

    // Step 3: walk consecutive runs and emit groups.
    auto &our_ranks = g_grouping_scratch.our_ranks;
    std::size_t i = 0;
    while (i < infos.size()) {
      int combo = infos[i].combo;

      // Find end-of-combo range.
      std::size_t combo_end = i;
      while (combo_end < infos.size() && infos[combo_end].combo == combo)
        ++combo_end;

      if (combo == -1) {
        // PASS: own group.
        MoveGroup &g = get_slot();
        for (std::size_t k = i; k < combo_end; ++k) {
          int idx = infos[k].orig_idx;
          g.moves.push_back(opp_moves[idx]);
          g.probs.push_back(opp_probs[idx]);
        }
        i = combo_end;
        continue;
      }

      Move::Combination c = static_cast<Move::Combination>(combo);
      if (is_collapse_combo(c)) {
        // Each rank-bucket is its own group (collapse over auxiliary).
        std::size_t j = i;
        while (j < combo_end) {
          int r = infos[j].rank;
          MoveGroup &g = get_slot();
          while (j < combo_end && infos[j].rank == r) {
            int idx = infos[j].orig_idx;
            g.moves.push_back(opp_moves[idx]);
            g.probs.push_back(opp_probs[idx]);
            ++j;
          }
        }
      } else {
        // Walk rank-buckets, merging consecutive ranks if our hand has no
        // same-combination response in (R_a, R_b].
        our_response_ranks_for_combo_into(our_hand, c, our_ranks);
        std::size_t j = i;
        MoveGroup *current = nullptr;
        int last_max_rank = -1;
        while (j < combo_end) {
          int r = infos[j].rank;
          bool start_new = (current == nullptr) ||
                            has_response_in_range(our_ranks, last_max_rank, r);
          if (start_new) {
            current = &get_slot();
          }
          last_max_rank = r;
          while (j < combo_end && infos[j].rank == r) {
            int idx = infos[j].orig_idx;
            current->moves.push_back(opp_moves[idx]);
            current->probs.push_back(opp_probs[idx]);
            ++j;
          }
        }
      }

      i = combo_end;
    }
  }

  // Trim unused trailing slots — preserves capacity of the outer vector but
  // destroys inner vectors of unused slots. Acceptable: per-call group counts
  // are stable so this is rare in steady state.
  if (used < out.size()) {
    out.resize(used);
  }
}

std::vector<MoveGroup> group_opp_moves(const std::array<int, 13> &our_hand,
                                        const std::vector<int> &opp_moves,
                                        const std::vector<float> &opp_probs) {
  std::vector<MoveGroup> out;
  group_opp_moves_into(our_hand, opp_moves, opp_probs, out);
  return out;
}

}  // namespace typed_search
