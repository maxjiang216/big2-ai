#include "forced_search.h"

#include "eval_features.h"
#include "eval_table.h"
#include "move.h"
#include "util.h"

#include <algorithm>

namespace typed_search {

namespace {

inline int hand_total(const std::array<int, 13> &h) {
  int s = 0;
  for (int c : h) s += c;
  return s;
}

inline float eval_at(const std::array<int, 13> &hand,
                      const std::array<int, 13> &discard, int opp_count,
                      const EvalTable &table, float default_value) {
  LeafContext ctx{hand, discard, opp_count, /*initiative=*/0};
  auto imp = impute_terminal(ctx);
  if (imp.has_value) return imp.value;
  uint32_t main_id = main_state_id(ctx);
  uint32_t fb_id = fallback_state_id(ctx);
  return table.query(main_id, fb_id, /*min_visits=*/5.0f, default_value);
}

float forced_search_recursive(const std::array<int, 13> &hand,
                               const std::array<int, 13> &discard,
                               int opp_count, const EvalTable &table,
                               float default_value, int depth_remaining) {
  if (hand_total(hand) == 0) return 1.0f;

  float best = eval_at(hand, discard, opp_count, table, default_value);
  if (best >= 1.0f || depth_remaining <= 0) return best;

  Move pass(Move::Combination::kPass);
  auto legal = compute_legal_moves(hand, pass);

  // Filter & order: only unbeatable, sort by num_cards desc, then move_id asc.
  std::vector<int> forced_ids;
  forced_ids.reserve(legal.size());
  for (int mid : legal) {
    if (mid == kPASS) continue;
    if (opponent_can_respond(mid, hand, discard, opp_count)) continue;
    forced_ids.push_back(mid);
  }
  std::sort(forced_ids.begin(), forced_ids.end(), [](int a, int b) {
    int ca = MOVE_TO_CARDS[a][13];
    int cb = MOVE_TO_CARDS[b][13];
    if (ca != cb) return ca > cb;
    return a < b;
  });

  for (int mid : forced_ids) {
    std::array<int, 13> new_hand = hand;
    std::array<int, 13> new_discard = discard;
    const auto &cost = MOVE_TO_CARDS[mid];
    for (int r = 0; r < 13; ++r) {
      new_hand[r] -= cost[r];
      new_discard[r] += cost[r];
    }
    if (hand_total(new_hand) == 0) return 1.0f;
    float v = forced_search_recursive(new_hand, new_discard, opp_count, table,
                                       default_value, depth_remaining - 1);
    if (v > best) best = v;
    if (best >= 1.0f) return best;
  }
  return best;
}

}  // namespace

float forced_search_value(const std::array<int, 13> &hand,
                          const std::array<int, 13> &discard, int opp_count,
                          const EvalTable &eval_table, float default_value,
                          int depth_cap) {
  // Fast path: the existing tablebase forced-win finder.
  auto fw = find_forced_win(hand, discard, opp_count);
  if (fw.has_value()) return 1.0f;
  return forced_search_recursive(hand, discard, opp_count, eval_table,
                                  default_value, depth_cap);
}

}  // namespace typed_search
