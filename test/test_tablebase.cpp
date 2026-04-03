#include "game.h"
#include "game_record.h"
#include "game_simulator.h"
#include "move.h"
#include "partial_game.h"
#include "random/random_player_factory.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <random>
#include <vector>

// ------------------------------------------------------------------ helpers

// Returns true if all moves in the list are singles (no bombs, pairs, etc.).
static bool all_singles(const std::vector<int> &legal) {
  if (legal.empty()) return false;
  for (int mid : legal) {
    if (mid == kPASS) continue;
    Move m(mid);
    if (m.combination != Move::Combination::kSingle) return false;
  }
  return true;
}

// Returns the highest rank among the singles in the legal move list.
static int highest_single_rank(const std::vector<int> &legal) {
  int best = -1;
  for (int mid : legal) {
    if (mid == kPASS) continue;
    Move m(mid);
    if (m.combination == Move::Combination::kSingle && m.rank > best)
      best = m.rank;
  }
  return best;
}

// ------------------------------------------------------------------ main integration test

// Play N games via GameSimulator (both players are RandomPlayer, which now
// inherits the tablebase from Player). Scan every recorded turn.
// When the tablebase condition holds:
//   - new trick (last_move == pass)
//   - opponent has exactly 1 card
//   - all legal moves are singles
// Verify that the move that was actually played is the highest single.
static void test_tablebase_fires_highest_single() {
  constexpr int N = 1000;
  std::mt19937 rng(42);
  RandomPlayerFactory f0(1u), f1(2u);

  int tablebase_turns_found = 0;

  for (int game_idx = 0; game_idx < N; ++game_idx) {
    GameSimulator sim(f0.create_player(), f1.create_player(), rng);
    GameRecord rec = sim.run();

    for (const auto &turn : rec.turns()) {
      int curr = turn.current_player;
      const PartialGame &view = turn.views[curr];

      // Check tablebase condition.
      bool new_trick = (view.last_move().combination == Move::Combination::kPass);
      bool opp_has_one = (view.opponent_hand_size() == 1);

      if (!new_trick || !opp_has_one) continue;

      // Filter legal_moves to non-pass (on a new trick there is no pass, but
      // be defensive).
      std::vector<int> non_pass;
      for (int m : turn.legal_moves)
        if (m != kPASS) non_pass.push_back(m);

      if (!all_singles(non_pass)) continue;

      // Tablebase condition holds — the highest single must have been played.
      int best_rank = highest_single_rank(non_pass);
      assert(best_rank != -1);
      assert(turn.move.combination == Move::Combination::kSingle);
      assert(turn.move.rank == best_rank);

      ++tablebase_turns_found;
    }
  }

  // Make sure we actually exercised the tablebase at least once.
  assert(tablebase_turns_found > 0);
}

// ------------------------------------------------------------------ non-firing tests

// When the opponent has more than 1 card, the tablebase must not fire and
// the game must still proceed legally.
static void test_tablebase_does_not_fire_when_opp_has_many_cards() {
  constexpr int N = 200;
  std::mt19937 rng(7);
  RandomPlayerFactory f0(3u), f1(4u);

  for (int i = 0; i < N; ++i) {
    GameSimulator sim(f0.create_player(), f1.create_player(), rng);
    GameRecord rec = sim.run();

    // Verify every move played was legal (basic sanity; tablebase or not).
    for (const auto &turn : rec.turns()) {
      int played_id = encodeMove(turn.move);
      bool found = false;
      for (int m : turn.legal_moves) {
        if (m == played_id) { found = true; break; }
      }
      assert(found);
    }
    assert(rec.game().is_over());
  }
}

// When the current player has multi-card moves available (e.g. a pair),
// the tablebase must not fire even if the opponent has 1 card.
// Verified via record: the played move may be anything legal.
static void test_tablebase_does_not_fire_when_non_singles_exist() {
  constexpr int N = 500;
  std::mt19937 rng(99);
  RandomPlayerFactory f0(5u), f1(6u);

  for (int i = 0; i < N; ++i) {
    GameSimulator sim(f0.create_player(), f1.create_player(), rng);
    GameRecord rec = sim.run();

    for (const auto &turn : rec.turns()) {
      int curr = turn.current_player;
      const PartialGame &view = turn.views[curr];

      bool new_trick = (view.last_move().combination == Move::Combination::kPass);
      bool opp_has_one = (view.opponent_hand_size() == 1);

      if (!new_trick || !opp_has_one) continue;

      std::vector<int> non_pass;
      for (int m : turn.legal_moves)
        if (m != kPASS) non_pass.push_back(m);

      if (all_singles(non_pass)) {
        // Tablebase fired — already checked in test_tablebase_fires_highest_single.
        continue;
      }

      // Non-singles exist: tablebase must NOT have fired.
      // The played move can be anything legal — just verify it is legal.
      int played_id = encodeMove(turn.move);
      bool found = false;
      for (int m : turn.legal_moves) {
        if (m == played_id) { found = true; break; }
      }
      assert(found);
    }
  }
}

// ------------------------------------------------------------------ case 0: win-now

// Whenever there is a legal move that plays all remaining cards, the player
// must take it immediately. Verified from GameRecord across many games.
static void test_tablebase_plays_winning_move_immediately() {
  constexpr int N = 1000;
  std::mt19937 rng(17);
  RandomPlayerFactory f0(7u), f1(8u);

  int winning_turns_found = 0;

  for (int game_idx = 0; game_idx < N; ++game_idx) {
    GameSimulator sim(f0.create_player(), f1.create_player(), rng);
    GameRecord rec = sim.run();

    for (const auto &turn : rec.turns()) {
      int curr = turn.current_player;
      int hand_size = turn.game.get_player_hand_size(curr);

      // Does any legal move empty the hand? (Same rule as tablebase case 0.)
      bool winning_move_exists = false;
      for (int mid : turn.legal_moves) {
        if (mid != kPASS && MOVE_TO_CARDS[mid][13] == hand_size) {
          winning_move_exists = true;
          break;
        }
      }
      if (!winning_move_exists) continue;

      assert(MOVE_TO_CARDS[encodeMove(turn.move)][13] == hand_size);

      ++winning_turns_found;
    }
  }

  // Sanity: the condition must have been triggered at least once.
  assert(winning_turns_found > 0);
}

// ------------------------------------------------------------------ forced-win unit tests

// Construct a state where we hold two singles that the opponent provably
// cannot respond to (all bomb-forming ranks are exhausted in the discard pile
// and neither A nor 2 is available to the opponent).
//
// Hand:  A (rank 14, index 11) + 2 (rank 15, index 12)  — 2 cards total.
// Discard: every rank fully discarded except us holding 1 A and 1 two.
//   ranks 3–K (indices 0–10): 4 each → discard[0..10] = 4
//   rank A  (index 11)       : 3 copies, we hold 1, 1 discarded, 1 with opp
//   rank 2  (index 12)       : 1 copy, we hold it → unseen = 0
// Opponent: 1 card (the remaining A, rank 14).
//
// Forced win: lead the 2 (unbeatable — no card can beat rank 15 except a bomb,
// and no bombs are formable), opponent passes; then lead the A (unbeatable —
// only the 2 beats it, but that's gone), opponent passes; hand empty → win.
static void test_find_forced_win_unit() {
  std::array<int, 13> hand{};
  hand[11] = 1; // one A
  hand[12] = 1; // the only 2

  std::array<int, 13> discard{};
  for (int r = 0; r < 11; ++r)
    discard[r] = 4; // all ranks 3–K fully discarded
  discard[11] = 1;  // one A discarded (we hold 1, opp holds 1)
  discard[12] = 0;  // we hold the only 2

  int opp_count = 1; // opponent has the remaining A

  auto seq = find_forced_win(hand, discard, opp_count);

  assert(seq.has_value());
  assert(seq->size() == 2);

  // Both moves must be singles of rank 14 (A) and 15 (2) in some order.
  // (Equal card counts; tie-break is lower move_id among legal singles.)
  Move m0((*seq)[0]), m1((*seq)[1]);
  assert(m0.combination == Move::Combination::kSingle);
  assert(m1.combination == Move::Combination::kSingle);
  int r0 = m0.rank, r1 = m1.rank;
  assert((r0 == 14 && r1 == 15) || (r0 == 15 && r1 == 14));
}

// When no forced win exists (opponent has cards that can beat everything we
// hold), find_forced_win must return nullopt.
static void test_find_forced_win_no_sequence() {
  // We hold single 3 and single 4. Discard is empty. Opponent has 10 cards.
  // hand_size = 2, so neither single (numCards=1) is hand-emptying.
  // Opponent has many unseen higher cards and can respond to both → no forced win.
  std::array<int, 13> hand{};
  hand[0] = 1; // single 3 (rank 3, index 0)
  hand[1] = 1; // single 4 (rank 4, index 1)

  std::array<int, 13> discard{};

  int opp_count = 10;

  auto seq = find_forced_win(hand, discard, opp_count);
  assert(!seq.has_value());
}

void run_tablebase_tests() {
  test_find_forced_win_unit();
  test_find_forced_win_no_sequence();
  test_tablebase_plays_winning_move_immediately();
  test_tablebase_fires_highest_single();
  test_tablebase_does_not_fire_when_opp_has_many_cards();
  test_tablebase_does_not_fire_when_non_singles_exist();
}
