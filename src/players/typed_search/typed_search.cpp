#include "typed_search.h"

#include "eval_features.h"
#include "eval_table.h"
#include "forced_search.h"
#include "move_grouping.h"
#include "move_prob_table.h"
#include "util.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace typed_search {

namespace {

// Thread-local scratch pool indexed by recursion depth. Reusable vectors
// retain their capacity across calls, so after the first few calls the inner
// pushes never reallocate.
struct ScratchLevel {
  std::vector<int> opp_legal;
  std::vector<float> opp_probs;
  std::vector<MoveGroup> groups;
};

struct SearchScratch {
  std::vector<ScratchLevel> levels;
  int depth = 0;
};

thread_local SearchScratch g_search_scratch;

struct ScratchGuard {
  ScratchLevel *lvl;
  ScratchGuard() {
    auto &s = g_search_scratch;
    if (static_cast<size_t>(s.depth) >= s.levels.size()) {
      s.levels.emplace_back();
    }
    lvl = &s.levels[s.depth];
    ++s.depth;
  }
  ~ScratchGuard() { --g_search_scratch.depth; }
  ScratchGuard(const ScratchGuard &) = delete;
  ScratchGuard &operator=(const ScratchGuard &) = delete;
};

inline int hand_total(const std::array<int, 13> &h) {
  int s = 0;
  for (int c : h) s += c;
  return s;
}

inline void apply_to_arrays(int move_id, std::array<int, 13> &our_hand,
                             std::array<int, 13> &discard) {
  const auto &cost = MOVE_TO_CARDS[move_id];
  for (int r = 0; r < 13; ++r) {
    our_hand[r] -= cost[r];
    discard[r] += cost[r];
  }
}

inline void apply_opp_to_discard(int move_id, std::array<int, 13> &discard) {
  const auto &cost = MOVE_TO_CARDS[move_id];
  for (int r = 0; r < 13; ++r) discard[r] += cost[r];
}

// Per-move precomputed HandBits "needs" + total card count. Used by the
// O(1) bitset version of opp_could_play.
struct MoveNeeds {
  HandBits bits;
  int count;
};

const std::array<MoveNeeds, LEGAL_MOVES_SIZE> &move_needs_table() {
  static const auto table = []() {
    std::array<MoveNeeds, LEGAL_MOVES_SIZE> t{};
    for (int mid = 0; mid < LEGAL_MOVES_SIZE; ++mid) {
      const auto &cost = MOVE_TO_CARDS[mid];
      HandBits b{};
      for (int r = 0; r < 13; ++r) {
        if (cost[r] >= 1) b.at1 |= static_cast<uint16_t>(1u << r);
        if (cost[r] >= 2) b.at2 |= static_cast<uint16_t>(1u << r);
        if (cost[r] >= 3) b.at3 |= static_cast<uint16_t>(1u << r);
        if (cost[r] >= 4) b.at4 |= static_cast<uint16_t>(1u << r);
      }
      t[mid] = {b, cost[13]};
    }
    return t;
  }();
  return table;
}

// Build HandBits representing the upper-bound on what cards the opponent
// could hold, given our hand and the discard.
inline HandBits opp_upper_bound_bits(const std::array<int, 13> &our_hand,
                                      const std::array<int, 13> &discard) {
  HandBits b{};
  for (int r = 0; r < 13; ++r) {
    int om = max_cards_in_deck_for_rank(r) - our_hand[r] - discard[r];
    if (om <= 0) continue;
    b.at1 |= static_cast<uint16_t>(1u << r);
    if (om >= 2) b.at2 |= static_cast<uint16_t>(1u << r);
    if (om >= 3) b.at3 |= static_cast<uint16_t>(1u << r);
    if (om >= 4) b.at4 |= static_cast<uint16_t>(1u << r);
  }
  return b;
}

// O(1) bitset check: can opp's possible-card upper bound cover `mid`'s needs?
inline bool opp_could_play_bits(int mid, const HandBits &opp_bits,
                                 int opp_count,
                                 const std::array<MoveNeeds, LEGAL_MOVES_SIZE> &mn_tbl) {
  const auto &mn = mn_tbl[mid];
  if (opp_count < mn.count) return false;
  return (mn.bits.at1 & opp_bits.at1) == mn.bits.at1 &&
         (mn.bits.at2 & opp_bits.at2) == mn.bits.at2 &&
         (mn.bits.at3 & opp_bits.at3) == mn.bits.at3 &&
         (mn.bits.at4 & opp_bits.at4) == mn.bits.at4;
}

}  // namespace

TypedSearch::TypedSearch(const EvalTable &eval_table,
                          const MoveProbTable &mp_table,
                          std::uint64_t rng_seed)
    : eval_table_(eval_table), mp_table_(mp_table), rng_(rng_seed) {}

std::uint64_t TypedSearch::make_key(const std::array<int, 13> &hand,
                                     int opp_count,
                                     const Move &last_move) const {
  std::uint64_t k = 0;
  for (int r = 0; r < 13; ++r) k = (k << 3) | (static_cast<std::uint64_t>(hand[r]) & 7u);
  k = (k << 5) | (static_cast<std::uint64_t>(opp_count) & 0x1Fu);
  k = (k << 5) |
      (static_cast<std::uint64_t>(static_cast<int>(last_move.combination)) & 0x1Fu);
  k = (k << 5) | (static_cast<std::uint64_t>(last_move.rank) & 0x1Fu);
  return k;
}

float TypedSearch::eval_leaf_we_passed(const std::array<int, 13> &hand,
                                        const std::array<int, 13> &discard,
                                        int opp_count) const {
  LeafContext ctx{hand, discard, opp_count, /*initiative=*/1};
  auto imp = impute_terminal(ctx);
  if (imp.has_value) return imp.value;
  uint32_t mid = main_state_id(ctx);
  uint32_t fid = fallback_state_id(ctx);
  return eval_table_.query(mid, fid);
}

float TypedSearch::eval_leaf_we_have_init(const std::array<int, 13> &hand,
                                           const std::array<int, 13> &discard,
                                           int opp_count) const {
  // Imputed wins (e.g., one-move-clear) handled inside forced_search via
  // impute_terminal.
  return forced_search_value(hand, discard, opp_count, eval_table_);
}

float TypedSearch::visit_our(OurNode &n) {
  if (n.computed) return n.value;
  n.computed = true;

  // Terminal imputation.
  if (hand_total(n.hand) == 0) {
    n.value = 1.0f;
    n.best_move = -1;
    return n.value;
  }
  if (n.opp_count <= 0) {
    n.value = 0.0f;
    n.best_move = -1;
    return n.value;
  }

  auto legal = compute_legal_moves(n.hand, n.last_move);
  // compute_legal_moves at lead position never includes PASS; at response it does.

  float best = -1.0f;
  int best_move = -1;
  for (int m : legal) {
    float v;
    if (m == kPASS) {
      // We pass; opp gains initiative. Leaf evaluated from our POV.
      v = eval_leaf_we_passed(n.hand, n.discard, n.opp_count);
    } else {
      auto new_hand = n.hand;
      auto new_discard = n.discard;
      apply_to_arrays(m, new_hand, new_discard);
      if (hand_total(new_hand) == 0) {
        v = 1.0f;  // we just emptied our hand
      } else {
        v = visit_opp(new_hand, new_discard, n.opp_count, m);
      }
    }
    if (v > best) {
      best = v;
      best_move = m;
    }
  }

  n.value = best;
  n.best_move = best_move;
  return best;
}

float TypedSearch::visit_opp(const std::array<int, 13> &our_hand_after,
                              const std::array<int, 13> &discard_after,
                              int opp_count, int our_move_id) {
  // Acquire thread-local scratch buffers for this recursion level. Capacity
  // is retained across calls so the inner pushes typically don't reallocate.
  ScratchGuard guard;
  auto &opp_legal = guard.lvl->opp_legal;
  auto &opp_probs = guard.lvl->opp_probs;
  auto &groups = guard.lvl->groups;
  opp_legal.clear();
  opp_legal.reserve(16);
  opp_legal.push_back(kPASS);
  const auto &beating = get_beating_moves();
  // Hoist: precompute the opp upper-bound HandBits ONCE, then bitmask-check
  // each candidate move in O(1).
  const HandBits opp_bits = opp_upper_bound_bits(our_hand_after, discard_after);
  const auto &mn_tbl = move_needs_table();
  for (int mid : beating[our_move_id]) {
    if (opp_could_play_bits(mid, opp_bits, opp_count, mn_tbl)) {
      opp_legal.push_back(mid);
    }
  }

  // Query move-prob distribution.
  mp_table_.query(our_move_id, opp_count, opp_legal, opp_probs);

  // Group by response equivalence (writes into scratch.groups).
  group_opp_moves_into(our_hand_after, opp_legal, opp_probs, groups);

  // For each group: sample a representative and recurse.
  float ev = 0.0f;
  for (auto &g : groups) {
    float group_prob = 0.0f;
    for (float p : g.probs) group_prob += p;
    if (group_prob <= 0.0f) continue;

    // Sample representative weighted by g.probs.
    std::uniform_real_distribution<float> U(0.0f, group_prob);
    float r = U(rng_);
    float acc = 0.0f;
    int rep = g.moves.front();
    for (std::size_t i = 0; i < g.moves.size(); ++i) {
      acc += g.probs[i];
      if (r <= acc) {
        rep = g.moves[i];
        break;
      }
    }

    float val;
    if (rep == kPASS) {
      // Opp passes; we gain initiative. Leaf eval (with forced-search).
      val = eval_leaf_we_have_init(our_hand_after, discard_after, opp_count);
    } else {
      auto new_discard = discard_after;
      apply_opp_to_discard(rep, new_discard);
      int new_opp_count = opp_count - MOVE_TO_CARDS[rep][13];
      Move rep_move(rep);
      if (new_opp_count <= 0) {
        // Opp just emptied hand: we lose.
        val = 0.0f;
      } else {
        std::uint64_t key = make_key(our_hand_after, new_opp_count, rep_move);
        auto it = memo_.find(key);
        if (it == memo_.end()) {
          OurNode child;
          child.hand = our_hand_after;
          child.discard = new_discard;
          child.opp_count = new_opp_count;
          child.last_move = rep_move;
          it = memo_.emplace(key, std::move(child)).first;
          val = visit_our(it->second);
        } else {
          val = it->second.computed ? it->second.value : visit_our(it->second);
        }
      }
    }
    ev += group_prob * val;
  }
  return ev;
}

TypedSearch::Result TypedSearch::run(const std::array<int, 13> &our_hand,
                                      const std::array<int, 13> &discard,
                                      int opp_count, const Move &last_move) {
  std::uint64_t key = make_key(our_hand, opp_count, last_move);
  OurNode root;
  root.hand = our_hand;
  root.discard = discard;
  root.opp_count = opp_count;
  root.last_move = last_move;
  auto it = memo_.emplace(key, std::move(root)).first;
  float v = visit_our(it->second);
  return Result{it->second.best_move, v};
}

}  // namespace typed_search
