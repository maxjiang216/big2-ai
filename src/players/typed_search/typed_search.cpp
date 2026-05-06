#include "typed_search.h"

#include "eval_features.h"
#include "eval_table.h"
#include "forced_search.h"
#include "move_grouping.h"
#include "move_prob_table.h"
#include "util.h"

#include <algorithm>
#include <climits>
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

// Index of `face_rank` (3..K=13, A=14, "2"=2 or 15) into our 0..12 rank space.
inline int rank_idx_of(int face_rank) {
  return (face_rank == 2 || face_rank == 15) ? 12 : (face_rank - 3);
}

// Does removing `delta` cards of `rank_idx` from `hand` change the set of
// straight / double-straight / triple-straight moves available? If yes, the
// rank participates in some straight-family combination — NOT loose.
// The four-of-a-kind bomb at the rank is ignored here (those don't get pruned
// by this rule).
bool affects_straights(const std::array<int, 13> &hand, int rank_idx,
                        int delta) {
  if (hand[rank_idx] < delta) return false;  // shouldn't happen
  HandBits hb1 = hand_bits_from_counts(hand);
  std::array<int, 13> mod = hand;
  mod[rank_idx] -= delta;
  HandBits hb2 = hand_bits_from_counts(mod);
  thread_local std::vector<int> s1, s2;
  straight_moves_for_pass_masks_into(hb1.at1, hb1.at2, hb1.at3, s1);
  straight_moves_for_pass_masks_into(hb2.at1, hb2.at2, hb2.at3, s2);
  return s1.size() != s2.size();
}

// Drop moves that are strictly dominated by a smaller-rank alternative
// using a "loose" sacrifice card (or pair). Two complementary rules:
//
// 1. Single-move domination, restricted to the 2-card endgame.
//    With exactly 2 cards in hand and 2+ legal Single moves both loose
//    (count == 1, not in any straight), the smallest-rank wins. Argument:
//    after playing X1 (smaller), we hold X2 alone — a strictly more
//    capable last card than X1. The reason this doesn't extend to 3+
//    cards is parity: in mid-game we may *want* to play a larger single
//    to force opp to pass and keep initiative, which the search will
//    weigh against the dominance gain. Restricting to hand_size == 2
//    keeps the "always-true" subset of the rule.
//
// 2. Bomb / full-house aux domination. Among bomb moves at the same
//    bomb_rank, those whose auxiliary is a "loose" sacrifice (single aux
//    with count_X == 1 in hand-after-bomb and not in any straight; or pair
//    aux with count_Y == 2 and not in any straight/DS/TS) are dominated by
//    the smallest-rank loose aux of the same kind. Same logic for full
//    houses keyed by triple_rank with loose-pair auxes. Bombs/FHs pin
//    down the trick state more tightly than singles so the dominance
//    argument carries through without parity concerns.
//
// Non-loose auxes and the bare bomb are NOT pruned — the dominance
// argument doesn't extend to those without more analysis.
// Diagnostic counters — incremented on each prune call and reported by
// typed_search_train via dump_prune_stats(). Per-thread to avoid contention.
struct PruneStats {
  std::uint64_t calls{0};
  std::uint64_t calls_multi{0};       // legal_moves_in > 1
  std::uint64_t calls_with_drop{0};
  std::uint64_t total_drops_single{0};
  std::uint64_t total_drops_bomb_aux{0};
  std::uint64_t total_drops_fh_aux{0};
  std::uint64_t total_input_moves{0};
};
thread_local PruneStats g_prune_stats;

void prune_dominated_moves(const std::array<int, 13> &hand,
                            std::vector<int> &legal) {
  g_prune_stats.calls++;
  g_prune_stats.total_input_moves += legal.size();
  if (legal.size() <= 1) return;
  g_prune_stats.calls_multi++;

  // Single-move dominance applies only in the 2-card endgame (parity
  // matters for mid-game). Bomb/FH aux dominance always applies.
  int hand_size = 0;
  for (int c : hand) hand_size += c;
  const bool prune_singles_dom = (hand_size == 2);

  // Pass 1: find smallest-rank legal-move id of each "loose" category.
  int smallest_loose_single = INT_MAX;
  // Bomb aux maps keyed by bomb_rank_idx (0..11).
  std::array<int, 13> bomb_smallest_single_aux;
  bomb_smallest_single_aux.fill(INT_MAX);
  std::array<int, 13> bomb_smallest_pair_aux;
  bomb_smallest_pair_aux.fill(INT_MAX);
  // FH aux: keyed by triple_rank_idx.
  std::array<int, 13> fh_smallest_pair_aux;
  fh_smallest_pair_aux.fill(INT_MAX);

  const auto &all = all_moves();

  auto loose_single_at = [&](const std::array<int, 13> &h, int idx) {
    return h[idx] == 1 && !affects_straights(h, idx, 1);
  };
  auto loose_pair_at = [&](const std::array<int, 13> &h, int idx) {
    return h[idx] == 2 && !affects_straights(h, idx, 2);
  };

  for (int m : legal) {
    if (m == kPASS) continue;
    const Move &mv = all[m];
    if (prune_singles_dom && mv.combination == Move::Combination::kSingle) {
      int idx = rank_idx_of(mv.rank);
      if (loose_single_at(hand, idx)) {
        if (m < smallest_loose_single) smallest_loose_single = m;
      }
    } else if (mv.combination == Move::Combination::kBomb) {
      if (mv.auxiliary == 0) continue;  // bare
      int bomb_idx = rank_idx_of(mv.rank);
      int aux_idx = rank_idx_of(mv.auxiliary);
      const auto &cost = MOVE_TO_CARDS[m];
      int aux_count = cost[aux_idx];
      // Hand after the bomb removes its base cards (but not aux).
      std::array<int, 13> h_after_base = hand;
      h_after_base[bomb_idx] -= cost[bomb_idx];
      bool loose = (aux_count == 1)
                       ? loose_single_at(h_after_base, aux_idx)
                       : loose_pair_at(h_after_base, aux_idx);
      if (loose) {
        auto &slot = (aux_count == 1) ? bomb_smallest_single_aux[bomb_idx]
                                          : bomb_smallest_pair_aux[bomb_idx];
        if (m < slot) slot = m;
      }
    } else if (mv.combination == Move::Combination::kFullHouse) {
      int triple_idx = rank_idx_of(mv.rank);
      int aux_idx = rank_idx_of(mv.auxiliary);
      std::array<int, 13> h_after_triple = hand;
      h_after_triple[triple_idx] -= 3;
      if (loose_pair_at(h_after_triple, aux_idx)) {
        auto &slot = fh_smallest_pair_aux[triple_idx];
        if (m < slot) slot = m;
      }
    }
  }

  // Pass 2: drop any move that is dominated.
  auto is_dominated = [&](int m) -> bool {
    if (m == kPASS) return false;
    const Move &mv = all[m];
    if (prune_singles_dom && mv.combination == Move::Combination::kSingle) {
      int idx = rank_idx_of(mv.rank);
      if (smallest_loose_single != INT_MAX && m != smallest_loose_single &&
          loose_single_at(hand, idx)) {
        return true;
      }
    } else if (mv.combination == Move::Combination::kBomb &&
               mv.auxiliary != 0) {
      int bomb_idx = rank_idx_of(mv.rank);
      int aux_idx = rank_idx_of(mv.auxiliary);
      const auto &cost = MOVE_TO_CARDS[m];
      int aux_count = cost[aux_idx];
      std::array<int, 13> h_after_base = hand;
      h_after_base[bomb_idx] -= cost[bomb_idx];
      bool loose = (aux_count == 1)
                       ? loose_single_at(h_after_base, aux_idx)
                       : loose_pair_at(h_after_base, aux_idx);
      if (!loose) return false;
      int slot = (aux_count == 1) ? bomb_smallest_single_aux[bomb_idx]
                                      : bomb_smallest_pair_aux[bomb_idx];
      if (slot != INT_MAX && m != slot) return true;
    } else if (mv.combination == Move::Combination::kFullHouse) {
      int triple_idx = rank_idx_of(mv.rank);
      int aux_idx = rank_idx_of(mv.auxiliary);
      std::array<int, 13> h_after_triple = hand;
      h_after_triple[triple_idx] -= 3;
      if (!loose_pair_at(h_after_triple, aux_idx)) return false;
      int slot = fh_smallest_pair_aux[triple_idx];
      if (slot != INT_MAX && m != slot) return true;
    }
    return false;
  };

  // Also count what we drop, by kind, before the erase.
  for (int m : legal) {
    if (m == kPASS) continue;
    if (!is_dominated(m)) continue;
    const Move &mv = all[m];
    if (mv.combination == Move::Combination::kSingle) {
      g_prune_stats.total_drops_single++;
    } else if (mv.combination == Move::Combination::kBomb) {
      g_prune_stats.total_drops_bomb_aux++;
    } else if (mv.combination == Move::Combination::kFullHouse) {
      g_prune_stats.total_drops_fh_aux++;
    }
  }
  std::size_t before = legal.size();
  legal.erase(std::remove_if(legal.begin(), legal.end(), is_dominated),
               legal.end());
  if (legal.size() < before) g_prune_stats.calls_with_drop++;
}

}  // namespace

PruneStatsSnapshot snapshot_prune_stats_thread_local() {
  PruneStatsSnapshot s;
  s.calls = g_prune_stats.calls;
  s.calls_multi = g_prune_stats.calls_multi;
  s.calls_with_drop = g_prune_stats.calls_with_drop;
  s.total_drops_single = g_prune_stats.total_drops_single;
  s.total_drops_bomb_aux = g_prune_stats.total_drops_bomb_aux;
  s.total_drops_fh_aux = g_prune_stats.total_drops_fh_aux;
  s.total_input_moves = g_prune_stats.total_input_moves;
  return s;
}
void reset_prune_stats_thread_local() { g_prune_stats = {}; }

TypedSearch::TypedSearch(const EvalTable &eval_extended,
                          const EvalTable &eval_main,
                          const EvalTable &eval_fallback,
                          const MoveProbTable &mp_table,
                          std::uint64_t rng_seed)
    : eval_extended_(eval_extended), eval_main_(eval_main),
      eval_fallback_(eval_fallback), mp_table_(mp_table), rng_(rng_seed) {}

namespace {
constexpr float kShrinkKappa = 20.0f;
constexpr float kFbMinVisits = 5.0f;
constexpr float kDefaultPrior = 0.5f;
}  // namespace

// 3-tier shrinkage: returns ext shrunk toward (main shrunk toward fb shrunk
// toward default). Each tier's contribution scales with its own visit count.
static float chained_eval(const EvalTable &ext, const EvalTable &main_t,
                           const EvalTable &fb, uint32_t ext_id,
                           uint32_t main_id, uint32_t fb_id) {
  // Tier 3: fallback. Use as prior only if it has enough data.
  auto fb_r = fb.lookup(fb_id);
  float fb_value = (fb_r.found && fb_r.visit_count >= kFbMinVisits)
                       ? fb_r.win_prob
                       : kDefaultPrior;
  // Tier 2: main. Shrink toward fb_value.
  auto main_r = main_t.lookup(main_id);
  float main_value = main_r.found
                         ? EvalTable::shrink(fb_value, main_r.win_prob,
                                              main_r.visit_count, kShrinkKappa)
                         : fb_value;
  // Tier 1: extended. Shrink toward main_value.
  auto ext_r = ext.lookup(ext_id);
  float ext_value = ext_r.found
                        ? EvalTable::shrink(main_value, ext_r.win_prob,
                                             ext_r.visit_count, kShrinkKappa)
                        : main_value;
  return ext_value;
}

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
  uint32_t eid = extended_state_id(ctx);
  uint32_t mid = main_state_id(ctx);
  uint32_t fid = fallback_state_id(ctx);
  return chained_eval(eval_extended_, eval_main_, eval_fallback_, eid, mid, fid);
}

float TypedSearch::eval_leaf_we_have_init(const std::array<int, 13> &hand,
                                           const std::array<int, 13> &discard,
                                           int opp_count) {
  // Cheap-tactical extension: if the search has only visited a handful of
  // nodes so far and both hands are small (endgame), continue past the
  // trick boundary by recursively searching from this lead position. This
  // gives precise look-ahead in tactical endgames where the eval table's
  // bucketed features are too coarse.
  //
  // Bounded automatically: once memo_.size() reaches the cap, no further
  // leaf extensions trigger, so recursion terminates.
  constexpr std::size_t kExtendNodeCap = 10;
  constexpr int kExtendHandCap = 7;  // strictly less than 7
  if (memo_.size() < kExtendNodeCap) {
    int hand_size = 0;
    for (int c : hand) hand_size += c;
    if (hand_size < kExtendHandCap && opp_count < kExtendHandCap &&
        hand_size > 0 && opp_count > 0) {
      Move pass_move(Move::Combination::kPass);
      std::uint64_t key = make_key(hand, opp_count, pass_move);
      auto it = memo_.find(key);
      if (it == memo_.end()) {
        OurNode node;
        node.hand = hand;
        node.discard = discard;
        node.opp_count = opp_count;
        node.last_move = pass_move;
        auto inserted = memo_.emplace(key, std::move(node));
        return visit_our(inserted.first->second);
      }
      return it->second.computed ? it->second.value : visit_our(it->second);
    }
  }
  return forced_search_value(hand, discard, opp_count, eval_extended_,
                              eval_main_, eval_fallback_);
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

  // Snapshot the collect flag — only this top-level visit captures root
  // per-move evals. Disable for any recursive visit_our calls.
  const bool collect_here = collect_root_;
  if (collect_here) collect_root_ = false;

  auto legal = compute_legal_moves(n.hand, n.last_move);
  // compute_legal_moves at lead position never includes PASS; at response it does.

  // Prune strictly-dominated moves (smallest-loose-aux dominance) so the
  // search and the eval table aren't burdened with options that an obvious
  // dominance argument already settles.
  prune_dominated_moves(n.hand, legal);

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
    if (collect_here) root_evals_.emplace_back(m, v);
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

  collect_root_ = true;
  root_evals_.clear();
  float v = visit_our(it->second);
  collect_root_ = false;

  Result r;
  r.move_id = it->second.best_move;
  r.value = v;
  r.nodes_searched = memo_.size();
  r.top_moves = std::move(root_evals_);
  std::sort(r.top_moves.begin(), r.top_moves.end(),
             [](const auto &a, const auto &b) { return a.second > b.second; });
  return r;
}

}  // namespace typed_search
