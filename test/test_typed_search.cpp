#include "typed_search/eval_features.h"
#include "typed_search/eval_table.h"
#include "typed_search/forced_search.h"
#include "typed_search/move_grouping.h"
#include "typed_search/move_prob_table.h"
#include "typed_search/typed_search.h"
#include "move.h"
#include "util.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <unordered_set>

namespace {

// ---------------------------------------------------------------------------
// eval_features tests
// ---------------------------------------------------------------------------
void test_state_id_in_range() {
  std::array<int, 13> hand{};
  std::array<int, 13> discard{};
  hand[0] = 1; hand[1] = 1; hand[5] = 2; hand[11] = 1;
  typed_search::LeafContext ctx{hand, discard, 8, 0};
  uint32_t mid = typed_search::main_state_id(ctx);
  uint32_t fid = typed_search::fallback_state_id(ctx);
  assert(mid < typed_search::kMainStateCount);
  assert(fid < typed_search::kFallbackStateCount);

  ctx.initiative = 1;
  uint32_t mid2 = typed_search::main_state_id(ctx);
  assert(mid2 < typed_search::kMainStateCount);
  assert(mid2 != mid);  // initiative changes the encoding
}

void test_impute_terminal() {
  std::array<int, 13> empty{};
  std::array<int, 13> discard{};
  // Empty hand -> we win.
  typed_search::LeafContext ctx_empty{empty, discard, 5, 0};
  auto v = typed_search::impute_terminal(ctx_empty);
  assert(v.has_value && v.value == 1.0f);

  // Opp at 0 -> we lose (game already over).
  std::array<int, 13> some{};
  some[0] = 2;
  typed_search::LeafContext ctx_oppzero{some, discard, 0, 0};
  v = typed_search::impute_terminal(ctx_oppzero);
  assert(v.has_value && v.value == 0.0f);

  // We have init AND can play all in one move (e.g., a single 2).
  std::array<int, 13> single_2{};
  single_2[12] = 1;
  typed_search::LeafContext ctx_clear{single_2, discard, 5, 0};
  v = typed_search::impute_terminal(ctx_clear);
  assert(v.has_value && v.value == 1.0f);

  // We have init but multiple cards needing multiple moves -> no impute.
  std::array<int, 13> mixed{};
  mixed[0] = 1; mixed[5] = 1;  // a 3 and an 8 — two singles, can't play both at once
  typed_search::LeafContext ctx_mixed{mixed, discard, 5, 0};
  v = typed_search::impute_terminal(ctx_mixed);
  assert(!v.has_value);
}

void test_guaranteed_largest() {
  std::array<int, 13> hand{};
  std::array<int, 13> discard{};
  // Discard all aces (3) and all 2s (1): opp can have 0 of A (rank 11) or 2 (rank 12).
  discard[11] = 3;
  discard[12] = 1;
  // Opp has plenty of cards (so opp_count doesn't constrain).
  auto gl = typed_search::compute_r_star(hand, discard, 10);
  // Highest rank with opp_max>=1 should be at most rank 10 (K), since A and 2 are exhausted.
  assert(gl.r_star[1] <= 10);

  // Now mark every card we can: hand has all 4 of K (rank 10), all 4 of Q (rank 9), etc.
  hand[10] = 4; hand[9] = 4; hand[8] = 4; hand[7] = 4;  // K, Q, J, 10
  // Plus discard 11, 12. Opp can't have anything from index 7 onwards.
  gl = typed_search::compute_r_star(hand, discard, 10);
  assert(gl.r_star[1] <= 6);  // R*_1 should be <= rank 9 (face) i.e. index 6 (face 9)

  // Opp_count=1: doubles and triples impossible regardless.
  gl = typed_search::compute_r_star(hand, discard, 1);
  assert(gl.r_star[2] == -1);
  assert(gl.r_star[3] == -1);
}

// ---------------------------------------------------------------------------
// eval_table tests
// ---------------------------------------------------------------------------
void test_eval_table_roundtrip() {
  typed_search::EvalTable t1, t2;
  t1.add_observation(42, 1.0f, 3.0f);
  t1.add_observation(42, 0.0f, 1.0f);
  t1.add_observation(100, 0.5f, 10.0f);

  // total_wins[42] = 3, visit_count[42] = 4 -> raw 0.75
  // total_wins[100] = 5, visit_count[100] = 10 -> raw 0.5
  // kappa=0 disables shrinkage so raw ratios come through.
  assert(std::abs(t1.query(42, 0, /*kappa=*/0.0f, 0.0f, 0.0f) - 0.75f) < 1e-5f);
  assert(std::abs(t1.query(100, 0, 0.0f, 0.0f, 0.0f) - 0.5f) < 1e-5f);
  assert(t1.query(99, 0, 0.0f, 0.0f, 0.123f) == 0.123f);  // missing -> default

  const std::string path = "/tmp/typed_search_eval_test.bin";
  t1.save(path);
  t2.load(path);
  assert(std::abs(t2.query(42, 0, 0.0f, 0.0f, 0.0f) - 0.75f) < 1e-5f);
  assert(std::abs(t2.query(100, 0, 0.0f, 0.0f, 0.0f) - 0.5f) < 1e-5f);

  // Decay halves both — ratio unchanged.
  t2.decay(0.5f);
  assert(std::abs(t2.query(42, 0, 0.0f, 0.0f, 0.0f) - 0.75f) < 1e-5f);

  // Shrinkage: with kappa=20, a state with 4 visits at wp=0.75 and no
  // fallback (default prior 0.5) blends to (20*0.5 + 3) / 24 = 0.5417.
  assert(std::abs(t1.query(42, 0, 20.0f, 5.0f, 0.5f) - (13.0f / 24.0f)) <
         1e-4f);

  std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// move_prob_table tests
// ---------------------------------------------------------------------------
void test_move_prob_uniform_fallback() {
  typed_search::MoveProbTable t;
  std::vector<int> legal{0, 1, 2, 3};
  std::vector<float> probs;
  t.query(/*pm=*/5, /*oc=*/8, /*our_b=*/0, legal, probs);
  // Empty table, no fallback -> uniform prior, no data, normalized.
  for (float p : probs) assert(std::abs(p - 0.25f) < 1e-5f);
}

void test_move_prob_observation_and_normalize() {
  typed_search::MoveProbTable t;
  // For each observation, pass the played response as also "trial-feasible".
  std::vector<int> feas2{2};
  std::vector<int> feas3{3};
  t.add_observation(/*pm=*/5, /*oc=*/8, /*our_b=*/0, feas2, /*resp=*/2, 4.0f);
  t.add_observation(/*pm=*/5, /*oc=*/8, /*our_b=*/0, feas3, /*resp=*/3, 2.0f);

  std::vector<int> legal{0, 2, 3};  // include resp=0 (no obs)
  std::vector<float> probs;
  t.query(5, 8, /*our_b=*/0, legal, probs);
  // With per-y shrinkage and uniform prior 1/3 each:
  //   y=0: counts=0, trials=0  -> (kappa*1/3 + 0)/(kappa+0) = 1/3
  //   y=2: counts=4, trials=4  -> (kappa*1/3 + 4)/(kappa+4) = (20/3 + 4)/24
  //   y=3: counts=2, trials=2  -> (kappa*1/3 + 2)/(kappa+2) = (20/3 + 2)/22
  // Then normalize. We just sanity-check that probs sum to ~1 and y=2's
  // value > y=3's value (it had more trials/counts hence pulls more toward
  // its data).
  float total = probs[0] + probs[1] + probs[2];
  assert(std::abs(total - 1.0f) < 1e-4f);
  assert(probs[1] > probs[0]);  // y=2 has data, y=0 doesn't
}

// ---------------------------------------------------------------------------
// move_grouping tests
// ---------------------------------------------------------------------------
void test_grouping_singles_merge() {
  // Our hand: empty (no responses to anything except passing).
  std::array<int, 13> hand{};
  // Opp has singles at ranks 3 (idx 0), 4 (idx 1), 5 (idx 2). We have no
  // singles at all (no in-range responses), so all three should merge into one
  // group.
  std::vector<int> opp{kSINGLE_START + 0, kSINGLE_START + 1, kSINGLE_START + 2};
  std::vector<float> probs{0.3f, 0.4f, 0.3f};
  auto groups = typed_search::group_opp_moves(hand, opp, probs);
  assert(groups.size() == 1);
  assert(groups[0].moves.size() == 3);
  float sum = 0;
  for (float p : groups[0].probs) sum += p;
  assert(std::abs(sum - 1.0f) < 1e-5f);
}

void test_grouping_singles_split_when_response_in_range() {
  // Our hand has a single at rank 4 (idx 1).
  std::array<int, 13> hand{};
  hand[1] = 1;
  // Opp singles at rank 3 (idx 0) and rank 5 (idx 2). We have a response in
  // (rank 3, rank 5] (our 4 beats their 3 but not their 5), so they don't merge.
  std::vector<int> opp{kSINGLE_START + 0, kSINGLE_START + 2};
  std::vector<float> probs{0.5f, 0.5f};
  auto groups = typed_search::group_opp_moves(hand, opp, probs);
  assert(groups.size() == 2);
}

void test_grouping_bombs_collapse_by_rank() {
  std::array<int, 13> hand{};
  // Opp could play a bomb of 7 (bare or with various aux). All collapse.
  std::vector<int> opp;
  std::vector<float> probs;
  // Find all bomb moves with rank 7 (face).
  for (int mid = kBOMB_START; mid < kBOMB_START + 156; ++mid) {
    Move m(mid);
    if (m.combination == Move::Combination::kBomb && m.rank == 7) {
      opp.push_back(mid);
      probs.push_back(1.0f);
    }
  }
  assert(opp.size() > 1);  // multiple variants exist
  auto groups = typed_search::group_opp_moves(hand, opp, probs);
  // All same-rank bombs collapse into a single group.
  assert(groups.size() == 1);
  assert(groups[0].moves.size() == opp.size());
}

void test_grouping_pass_singleton() {
  std::array<int, 13> hand{};
  std::vector<int> opp{kPASS, kSINGLE_START + 5, kSINGLE_START + 6};
  std::vector<float> probs{0.5f, 0.25f, 0.25f};
  auto groups = typed_search::group_opp_moves(hand, opp, probs);
  // PASS is its own singleton group; the two singles may merge (no responses).
  bool has_pass_group = false;
  for (auto &g : groups) {
    if (g.moves.size() == 1 && g.moves[0] == kPASS) has_pass_group = true;
  }
  assert(has_pass_group);
}

}  // namespace

void run_typed_search_tests() {
  test_state_id_in_range();
  test_impute_terminal();
  test_guaranteed_largest();
  test_eval_table_roundtrip();
  test_move_prob_uniform_fallback();
  test_move_prob_observation_and_normalize();
  test_grouping_singles_merge();
  test_grouping_singles_split_when_response_in_range();
  test_grouping_bombs_collapse_by_rank();
  test_grouping_pass_singleton();
}
