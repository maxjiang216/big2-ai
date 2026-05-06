// One-off diagnostic for the AAAK vs AAA8/AAA9 valuation question.
//
// Hardcodes the user-reported position (Game 0 turn 2):
//   P0 hand pre-move: 5x3, 8, 9, 10x2, K, Ax3   (11 cards)
//   discard before turn 2: 4x2, 6x3, 7x3, 10x2  (from turns 0+1)
//   opp_count = 11
//
// For each AAA+aux candidate, simulates the move and prints:
//   - the resulting our-init leaf state ID (main + fallback)
//   - the table entries (total_wins, visit_count, win_prob)
//   - the forced_search_value bonus
//   - the final eval (max of table value and forced search)

#include "core/util.h"
#include "core/move.h"
#include "typed_search/eval_features.h"
#include "typed_search/eval_table.h"
#include "typed_search/forced_search.h"

#include <array>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using typed_search::EvalTable;

namespace {

const char *combo_str(Move::Combination c) {
  switch (c) {
    case Move::Combination::kPass: return "Pass";
    case Move::Combination::kSingle: return "Single";
    case Move::Combination::kDouble: return "Double";
    case Move::Combination::kTriple: return "Triple";
    case Move::Combination::kFullHouse: return "FullHouse";
    case Move::Combination::kBomb: return "Bomb";
    default: return "Other";
  }
}

void dump_hand(const std::array<int, 13> &h) {
  printf("[");
  for (int r = 0; r < 13; ++r)
    for (int k = 0; k < h[r]; ++k) putchar(rankToChar(r + 3));
  printf("]");
}

void dump_pos(const char *label, const std::array<int, 13> &hand,
              const std::array<int, 13> &discard, int opp_count,
              const EvalTable &main, const EvalTable &fb) {
  printf("\n--- %s ---\n", label);
  printf("hand=");
  dump_hand(hand);
  printf("  discard=");
  dump_hand(discard);
  int total = 0; for (int c : hand) total += c;
  printf("  our_size=%d  opp_count=%d\n", total, opp_count);

  typed_search::LeafContext ctx{hand, discard, opp_count, /*initiative=*/0};
  auto imp = typed_search::impute_terminal(ctx);
  if (imp.has_value) {
    printf("  IMPUTED terminal value: %.3f\n", imp.value);
    return;
  }
  uint32_t mid = typed_search::main_state_id(ctx);
  uint32_t fid = typed_search::fallback_state_id(ctx);

  // Main table lookup with full visibility.
  auto it_main = main.entries().find(mid);
  if (it_main != main.entries().end()) {
    float vc = it_main->second.visit_count;
    float wp = (vc > 0) ? it_main->second.total_wins / vc : 0.0f;
    printf("  main  state=%u  visits=%.1f  win_prob=%.3f\n", mid, vc, wp);
  } else {
    printf("  main  state=%u  visits=0  (absent)\n", mid);
  }

  auto it_fb = fb.entries().find(fid);
  if (it_fb != fb.entries().end()) {
    float vc = it_fb->second.visit_count;
    float wp = (vc > 0) ? it_fb->second.total_wins / vc : 0.0f;
    printf("  fb    state=%u  visits=%.1f  win_prob=%.3f\n", fid, vc, wp);
  } else {
    printf("  fb    state=%u  visits=0  (absent)\n", fid);
  }

  // Show what query() resolves to with min_visits=5 (default).
  float q = main.query(mid, fid, 5.0f, 0.5f);
  printf("  query(min_visits=5)  -> %.3f\n", q);

  // Forced-search value (this is what eval_leaf_we_have_init returns).
  float fsv = typed_search::forced_search_value(hand, discard, opp_count, main);
  printf("  forced_search_value -> %.3f\n", fsv);
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string dir = "data/typed_search_v4";
  if (argc >= 2) dir = argv[1];

  EvalTable main_t, fb_t;
  main_t.load(dir + "/eval_main.bin");
  fb_t.load(dir + "/eval_fallback.bin");
  main_t.set_fallback(&fb_t);
  std::cerr << "Loaded main=" << main_t.size() << " fb=" << fb_t.size() << "\n";

  // Ranks (indices): 0=3, 1=4, 2=5, 3=6, 4=7, 5=8, 6=9, 7=10, 8=J,
  //                  9=Q, 10=K, 11=A, 12=2.
  std::array<int, 13> pre_hand{0,0,3,0,0,1,1,2,0,0,1,3,0};  // 5x3, 8, 9, 10x2, K, Ax3
  std::array<int, 13> pre_disc{0,2,0,3,3,0,0,2,0,0,0,0,0};  // 4x2, 6x3, 7x3, 10x2

  // Helper: subtract `cost` from src into dst.
  auto play = [&](const std::array<int, 13> &cost) {
    std::array<int, 13> hand_a = pre_hand;
    std::array<int, 13> disc_a = pre_disc;
    for (int r = 0; r < 13; ++r) {
      hand_a[r] -= cost[r];
      disc_a[r] += cost[r];
    }
    return std::pair{hand_a, disc_a};
  };

  // AAA-bomb cards: 3 of A (idx 11). Aux is one extra rank.
  std::array<int, 13> bomb_AAA{0,0,0,0,0,0,0,0,0,0,0,3,0};

  auto compose = [&](std::array<int, 13> base,
                      std::initializer_list<std::pair<int,int>> deltas) {
    for (auto [idx, n] : deltas) base[idx] += n;
    return base;
  };

  auto prn = [&](const char *label, const std::array<int,13> &cost) {
    auto [h, d] = play(cost);
    dump_pos(label, h, d, /*opp_count=*/11, main_t, fb_t);
  };

  // AAAK = bomb AAA + K (idx 10).
  prn("AAAK", compose(bomb_AAA, {{10, 1}}));
  // AAA8 = + 8 (idx 5).
  prn("AAA8", compose(bomb_AAA, {{5, 1}}));
  // AAA9 = + 9 (idx 6).
  prn("AAA9", compose(bomb_AAA, {{6, 1}}));
  // AAA0 = + 10 (idx 7), one of them.
  prn("AAA0", compose(bomb_AAA, {{7, 1}}));
  // AAA00 = + 10x2.
  prn("AAA00", compose(bomb_AAA, {{7, 2}}));
  // AAA5 = + 5 (idx 2), one.
  prn("AAA5", compose(bomb_AAA, {{2, 1}}));
  // AAA55 = + 5x2.
  prn("AAA55", compose(bomb_AAA, {{2, 2}}));
  // AAA = bare bomb, no aux.
  prn("AAA", bomb_AAA);
  return 0;
}
