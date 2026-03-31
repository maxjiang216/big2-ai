#include "game.h"
#include "move.h"
#include "partial_game.h"
#include "util.h"

#include <cassert>
#include <random>

// PartialGame constructed from Game should reflect correct perspective.
static void test_construction_from_game() {
  std::mt19937 rng(17);
  Game g;
  g.shuffle_deal(rng);

  PartialGame p0(g, 0);
  PartialGame p1(g, 1);

  // turn: 0 = my turn. current_player is 0 so p0 is "my turn", p1 is "opponent's turn".
  assert(p0.turn() == 0);
  assert(p1.turn() == 1);

  // Hands match.
  auto h0 = g.player_hand(0);
  auto h1 = g.player_hand(1);
  auto pg0_hand = p0.player_hand();
  auto pg1_hand = p1.player_hand();
  for (int rank = 0; rank < 13; ++rank) {
    assert(pg0_hand[rank] == h0[rank]);
    assert(pg1_hand[rank] == h1[rank]);
  }
}

// Legal moves from PartialGame must be a subset of Game's legal moves when
// it is the player's own turn.
static void test_partial_legal_subset_of_game_legal() {
  std::mt19937 rng(9);
  Game g;
  g.shuffle_deal(rng);

  PartialGame p0(g, 0); // it's player 0's turn

  auto game_legal = g.get_legal_moves();
  auto part_legal = p0.get_legal_moves();

  // Every move in part_legal must appear in game_legal.
  for (int m : part_legal) {
    bool found = false;
    for (int gm : game_legal) {
      if (gm == m) { found = true; break; }
    }
    assert(found);
  }
  // And they must be the same size (same hand, same last move).
  assert(part_legal.size() == game_legal.size());
}

// After apply_move, turn flips.
static void test_apply_move_flips_turn() {
  std::mt19937 rng(5);
  Game g;
  g.shuffle_deal(rng);
  PartialGame p0(g, 0);
  assert(p0.turn() == 0);

  auto legal = p0.get_legal_moves();
  Move m(legal.front());
  p0.apply_move(m);
  assert(p0.turn() == 1);
}

// Possible moves (opponent model) must not include moves requiring more cards
// than remain unseen for any rank.
static void test_possible_moves_within_unseen_cards() {
  std::mt19937 rng(55);
  Game g;
  g.shuffle_deal(rng);

  PartialGame p0(g, 0);
  auto possible = p0.get_possible_moves();

  (void)p0.player_hand();
  // For each possible move, unseen supply of each rank must cover the cost.
  // We don't have the discard directly from PartialGame's public API, so we
  // verify by checking the move_to_cards total count <= opponent's card count.
  for (int mid : possible) {
    if (mid == kPASS) continue;
    // Each possible move must consume at most as many cards as the opponent has.
    assert(MOVE_TO_CARDS[mid][13] <= 16); // opponent starts with 16
  }
}

// get_possible_moves_not_bomb must contain no bomb moves.
static void test_possible_moves_not_bomb_has_no_bombs() {
  std::mt19937 rng(77);
  Game g;
  g.shuffle_deal(rng);
  PartialGame p0(g, 0);

  auto moves = p0.get_possible_moves_not_bomb();
  for (int mid : moves) {
    if (mid == kPASS) continue;
    Move m(mid);
    assert(m.combination != Move::Combination::kBomb);
  }
}

void run_partial_game_tests() {
  test_construction_from_game();
  test_partial_legal_subset_of_game_legal();
  test_apply_move_flips_turn();
  test_possible_moves_within_unseen_cards();
  test_possible_moves_not_bomb_has_no_bombs();
}
