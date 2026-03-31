#include "game.h"
#include "move.h"
#include "util.h"

#include <cassert>
#include <numeric>
#include <random>

// After deal: each player has 16 cards, 17 remain (unseen), no card rank
// exceeds its deck limit across both hands + discard.
static void test_deal_invariants() {
  std::mt19937 rng(42);
  Game g;
  g.shuffle_deal(rng);

  assert(g.get_player_hand_size(0) == 16);
  assert(g.get_player_hand_size(1) == 16);
  assert(!g.is_over());

  auto h0 = g.player_hand(0);
  auto h1 = g.player_hand(1);
  auto dp = g.discard_pile();

  for (int rank = 0; rank < 13; ++rank) {
    int total = h0[rank] + h1[rank] + dp[rank];
    int limit = max_cards_in_deck_for_rank(rank);
    assert(h0[rank] >= 0 && h0[rank] <= limit);
    assert(h1[rank] >= 0 && h1[rank] <= limit);
    assert(dp[rank] == 0); // nothing discarded yet
    assert(total <= limit);
  }
}

// Player alternates after every move.
static void test_player_alternates() {
  std::mt19937 rng(1);
  Game g;
  g.shuffle_deal(rng);
  int before = g.current_player();
  auto legal = g.get_legal_moves();
  assert(!legal.empty());
  g.apply_move(legal.front());
  assert(g.current_player() == 1 - before);
}

// Legal moves returned for a fresh deal should all be playable (hand contains
// those cards).
static void test_legal_moves_are_affordable() {
  std::mt19937 rng(99);
  Game g;
  g.shuffle_deal(rng);

  auto hand = g.player_hand(g.current_player());
  auto legal = g.get_legal_moves();
  assert(!legal.empty());

  for (int mid : legal) {
    const auto &cost = MOVE_TO_CARDS[mid];
    for (int rank = 0; rank < 13; ++rank) {
      assert(hand[rank] >= cost[rank]);
    }
  }
}

// apply_move(Move) and apply_move(int) must produce identical states.
static void test_apply_move_overloads_agree() {
  std::mt19937 rng(7);
  Game g1;
  g1.shuffle_deal(rng);
  rng.seed(7);
  Game g2;
  g2.shuffle_deal(rng);

  auto legal = g1.get_legal_moves();
  assert(!legal.empty());
  int mid = legal.front();
  Move m(mid);

  g1.apply_move(m);
  g2.apply_move(mid);

  for (int p = 0; p < 2; ++p) {
    assert(g1.get_player_hand_size(p) == g2.get_player_hand_size(p));
    auto h1 = g1.player_hand(p);
    auto h2 = g2.player_hand(p);
    for (int rank = 0; rank < 13; ++rank)
      assert(h1[rank] == h2[rank]);
  }
  assert(g1.current_player() == g2.current_player());
}

// Pass is legal when there is an active trick (last_move != pass).
static void test_pass_legality() {
  std::mt19937 rng(3);
  Game g;
  g.shuffle_deal(rng);

  // First move of game: no active trick, so pass should NOT be in legal moves.
  auto legal_start = g.get_legal_moves();
  bool pass_at_start = false;
  for (int m : legal_start) {
    if (m == kPASS) { pass_at_start = true; break; }
  }
  assert(!pass_at_start);

  // Make a non-pass move, then pass must be in the opponent's legal set.
  // Find any non-pass legal move.
  int first_move = -1;
  for (int m : legal_start) {
    if (m != kPASS) { first_move = m; break; }
  }
  assert(first_move != -1);
  g.apply_move(first_move);

  auto legal_after = g.get_legal_moves();
  bool pass_after = false;
  for (int m : legal_after) {
    if (m == kPASS) { pass_after = true; break; }
  }
  assert(pass_after);
}

void run_game_tests() {
  test_deal_invariants();
  test_player_alternates();
  test_legal_moves_are_affordable();
  test_apply_move_overloads_agree();
  test_pass_legality();
}
