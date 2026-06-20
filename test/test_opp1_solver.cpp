#include "../src/core/move.h"
#include "../src/core/opp1_solver.h"
#include "../src/core/series.h"
#include "../src/core/util.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <map>
#include <random>

namespace {

using C = Move::Combination;
using Arr = std::array<int, 13>;

// Empty (not loaded) series table -> binary win/loss valuation.
SeriesTable binary_table() { return SeriesTable{}; }
opp1::Belief uniform() {
  opp1::Belief b;
  b.fill(1.0 / 13.0);
  return b;
}

Arr hand_of(std::initializer_list<int> idx_counts) {
  // idx_counts: pairs (idx, count) flattened.
  Arr h{};
  auto it = idx_counts.begin();
  while (it != idx_counts.end()) {
    int idx = *it++;
    int cnt = *it++;
    h[idx] = cnt;
  }
  return h;
}

// Find the single trivial plan's exposed set via enumerate (no straights).
void test_outcome_model() {
  // Three distinct singles 3,4,5 (idx 0,1,2). No combos, no straights.
  Arr h = hand_of({0, 1, 1, 1, 2, 1});
  auto plans = opp1::enumerate_plans(h);
  assert(plans.size() == 1);
  const opp1::Plan &p = plans[0];
  assert((p.exposed == std::vector<int>{3, 4, 5}));
  // k(X): X=4 -> {3} =1 -> win ; X=5 -> {3,4}=2 -> lose by 1 ; X=6 -> {3,4,5}=3 lose2
  assert(opp1::k_below(p, 4) == 1);
  assert(opp1::k_below(p, 5) == 2);
  SeriesTable bt = binary_table();
  assert(opp1::value_vs_rank(p, 4, bt, 0, 0) == 1.0);  // win
  assert(opp1::value_vs_rank(p, 5, bt, 0, 0) == 0.0);  // loss
}

void test_bomb_aux_second_lowest() {
  // four 7s (idx4) bomb + loose singles 3 (idx0), 8 (idx5), K (idx10).
  Arr h = hand_of({0, 1, 4, 4, 5, 1, 10, 1});
  auto sol = opp1::solve_lead(h, uniform(), binary_table(), 0, 0);
  assert(!sol.moves.empty());
  Move first(sol.moves[0]);
  assert(first.combination == C::kBomb);
  assert(first.rank == 7);          // four 7s
  assert(first.auxiliary == 8);     // second-lowest loose single buried
  // Exposed should be {3, 13} (the global-lowest 3 stays free for last play).
  auto plans = opp1::enumerate_plans(h);
  assert(plans.size() == 1);
  assert((plans[0].exposed == std::vector<int>{3, 13}));
}

void test_straight_dominates() {
  // Singles 3..7 (a 5-straight, idx0..4) plus singles 9 (idx6), J (idx8).
  Arr h = hand_of({0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 6, 1, 8, 1});
  auto sol = opp1::solve_lead(h, uniform(), binary_table(), 0, 0);
  assert(sol.by_dominance);  // playing the straight strictly helps -> dominates
  Move first(sol.moves[0]);
  assert(first.combination == C::kStraight5);
  // After the straight, exposed singles are just {9, 11}.
  // Verify some plan in the enumeration realises that.
  auto plans = opp1::enumerate_plans(h);
  bool found = false;
  for (auto &p : plans)
    if (p.exposed == std::vector<int>{9, 11})
      found = true;
  assert(found);
}

void test_pairs_not_exposed() {
  // A pair (two 4s, idx1) + one single K (idx10). Pair shed safely; only K loose
  // but it is the lowest/only exposed -> always win.
  Arr h = hand_of({1, 2, 10, 1});
  auto plans = opp1::enumerate_plans(h);
  assert(plans.size() == 1);
  assert(plans[0].exposed == std::vector<int>{13});
  // |exposed| <= 1 -> win for every opp rank.
  for (int x = 3; x <= 15; ++x)
    assert(opp1::value_vs_rank(plans[0], x, binary_table(), 0, 0) == 1.0);
}

void test_empty_hand() {
  Arr h{};
  auto sol = opp1::solve_lead(h, uniform(), binary_table(), 0, 0);
  assert(sol.moves.empty());
}

// Independent ground truth: exhaustive search over OUR play sequences when opp
// holds a single of rank X (opp's responses are forced — opp beats any single
// it can, emptying to win). Returns our remaining cards on optimal loss, or -1
// if we can win. Memoised on the hand.
int brute_best(const Arr &hand, int X, std::map<Arr, int> &memo) {
  int total = 0;
  for (int r = 0; r < 13; ++r) total += hand[r];
  if (total == 0) return -1;  // already empty = won
  auto it = memo.find(hand);
  if (it != memo.end()) return it->second;
  int best = total;  // fallback (shouldn't bind: we always have a legal lead)
  for (int id : compute_legal_moves(hand, Move(C::kPass))) {
    if (id == 0) continue;
    Move m(id);
    const auto &c = MOVE_TO_CARDS[id];
    int rem_total = total - c[13];
    if (rem_total == 0) { best = -1; break; }  // empties -> win
    int cand;
    if (m.combination == C::kSingle && m.rank < X) {
      cand = rem_total;  // opp beats this single -> we lose with rem_total cards
    } else {
      Arr rem = hand;
      for (int r = 0; r < 13; ++r) rem[r] -= c[r];
      cand = brute_best(rem, X, memo);  // safe; continue leading
    }
    if (cand == -1) { best = -1; break; }
    best = std::min(best, cand);
  }
  memo[hand] = best;
  return best;
}

void test_solver_matches_bruteforce() {
  static const std::array<int, 13> cap = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1};
  std::mt19937 rng(12345);
  for (int trial = 0; trial < 3000; ++trial) {
    int n = 2 + (int)(rng() % 8);  // hands of size 2..9
    std::vector<int> pool;
    for (int r = 0; r < 13; ++r)
      for (int k = 0; k < cap[r]; ++k) pool.push_back(r);
    std::shuffle(pool.begin(), pool.end(), rng);
    Arr h{};
    for (int i = 0; i < n; ++i) h[pool[i]]++;

    auto plans = opp1::enumerate_plans(h);
    for (int X = 3; X <= 15; ++X) {
      // best achievable margin among our enumerated plans (-1 = win).
      int solver_best = 1 << 30;
      for (auto &p : plans) {
        int k = opp1::k_below(p, X);
        int v = (k <= 1) ? -1 : (k - 1);
        solver_best = std::min(solver_best, v);
      }
      std::map<Arr, int> memo;
      int bf = brute_best(h, X, memo);
      assert(solver_best == bf);  // theory is complete + outcome model exact
    }
  }
}

}  // namespace

void run_opp1_solver_tests() {
  test_solver_matches_bruteforce();
  test_outcome_model();
  test_bomb_aux_second_lowest();
  test_straight_dominates();
  test_pairs_not_exposed();
  test_empty_hand();
}
