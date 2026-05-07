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
                      const EvalTable &ext, const EvalTable &main_t,
                      const EvalTable &fb, float default_value) {
  LeafContext ctx{hand, discard, opp_count, /*initiative=*/0};
  auto imp = impute_terminal(ctx);
  if (imp.has_value) return imp.value;
  uint32_t ext_id = extended_state_id(ctx);
  uint32_t main_id = main_state_id(ctx);
  uint32_t fb_id = fallback_state_id(ctx);

  constexpr float kKappa = 20.0f;
  constexpr float kFbMin = 5.0f;
  // Tier 3: fb prior.
  auto fb_r = fb.lookup(fb_id);
  float fb_value = (fb_r.found && fb_r.visit_count >= kFbMin)
                       ? fb_r.win_prob
                       : default_value;
  // Tier 2: main shrunk toward fb.
  auto main_r = main_t.lookup(main_id);
  float main_value = main_r.found ? EvalTable::shrink(fb_value, main_r.win_prob,
                                                       main_r.visit_count, kKappa)
                                   : fb_value;
  // Tier 1: ext shrunk toward main.
  auto ext_r = ext.lookup(ext_id);
  return ext_r.found ? EvalTable::shrink(main_value, ext_r.win_prob,
                                          ext_r.visit_count, kKappa)
                      : main_value;
}

// Same opp_bits / opp_count invariance as in find_forced_win — hoist
// opp_bits to the top of forced_search and pass through recursion.
struct ForcedSearchCtx {
  HandBits opp_bits;
  int opp_count;
};

float forced_search_recursive(const std::array<int, 13> &hand,
                               const std::array<int, 13> &discard,
                               const ForcedSearchCtx &ctx,
                               const EvalTable &ext, const EvalTable &main_t,
                               const EvalTable &fb, float default_value,
                               int depth_remaining) {
  if (hand_total(hand) == 0) return 1.0f;

  float best = eval_at(hand, discard, ctx.opp_count, ext, main_t, fb,
                        default_value);
  if (best >= 1.0f || depth_remaining <= 0) return best;

  Move pass(Move::Combination::kPass);
  auto legal = compute_legal_moves(hand, pass);

  std::vector<int> forced_ids;
  forced_ids.reserve(legal.size());
  for (int mid : legal) {
    if (mid == kPASS) continue;
    if (opponent_can_respond(mid, ctx.opp_bits, ctx.opp_count)) continue;
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
    float v = forced_search_recursive(new_hand, new_discard, ctx, ext, main_t,
                                       fb, default_value, depth_remaining - 1);
    if (v > best) best = v;
    if (best >= 1.0f) return best;
  }
  return best;
}

ForcedSearchCtx build_ctx(const std::array<int, 13> &hand,
                           const std::array<int, 13> &discard, int opp_count) {
  ForcedSearchCtx ctx;
  ctx.opp_count = opp_count;
  ctx.opp_bits = HandBits{};
  for (int r = 0; r < 13; ++r) {
    int om = max_cards_in_deck_for_rank(r) - hand[r] - discard[r];
    if (om <= 0) continue;
    ctx.opp_bits.at1 |= static_cast<uint16_t>(1u << r);
    if (om >= 2) ctx.opp_bits.at2 |= static_cast<uint16_t>(1u << r);
    if (om >= 3) ctx.opp_bits.at3 |= static_cast<uint16_t>(1u << r);
    if (om >= 4) ctx.opp_bits.at4 |= static_cast<uint16_t>(1u << r);
  }
  return ctx;
}

}  // namespace

float forced_search_value(const std::array<int, 13> &hand,
                          const std::array<int, 13> &discard, int opp_count,
                          const EvalTable &eval_extended,
                          const EvalTable &eval_main,
                          const EvalTable &eval_fallback,
                          float default_value, int depth_cap) {
  // Fast path: the existing tablebase forced-win finder.
  auto fw = find_forced_win(hand, discard, opp_count);
  if (fw.has_value()) return 1.0f;
  ForcedSearchCtx ctx = build_ctx(hand, discard, opp_count);
  return forced_search_recursive(hand, discard, ctx, eval_extended, eval_main,
                                  eval_fallback, default_value, depth_cap);
}

}  // namespace typed_search
