#include "game.h"
#include "game_record.h"
#include "game_simulator.h"
#include "move.h"
#include "partial_game.h"
#include "random/random_player_factory.h"
#include "util.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <random>

// ------------------------------------------------------------------ helpers

// Verify all card conservation invariants hold on the given Game state.
// Returns false (with a description) if anything is wrong.
static void assert_card_conservation(const Game &g) {
  auto h0 = g.player_hand(0);
  auto h1 = g.player_hand(1);
  auto dp = g.discard_pile();

  int total_all = 0;
  for (int rank = 0; rank < 13; ++rank) {
    int limit = max_cards_in_deck_for_rank(rank);

    // No rank count goes negative.
    assert(h0[rank] >= 0);
    assert(h1[rank] >= 0);
    assert(dp[rank] >= 0);

    // No rank count exceeds deck limit.
    assert(h0[rank] <= limit);
    assert(h1[rank] <= limit);
    assert(dp[rank] <= limit);

    // Combined per-rank count does not exceed deck limit.
    int per_rank = h0[rank] + h1[rank] + dp[rank];
    assert(per_rank <= limit);

    total_all += per_rank;
  }

  // Total cards in play = 16 + 16 (dealt) = 32 (17 remain unseen and are
  // never in any of the three arrays, so sum must stay at 32 throughout).
  assert(total_all == 32);

  // Cached hand sizes must equal actual sums.
  int sum0 = 0, sum1 = 0;
  for (int rank = 0; rank < 13; ++rank) {
    sum0 += h0[rank];
    sum1 += h1[rank];
  }
  assert(g.get_player_hand_size(0) == sum0);
  assert(g.get_player_hand_size(1) == sum1);
}

// Verify that every legal move listed is actually affordable from hand.
static void assert_legal_moves_affordable(const Game &g) {
  auto legal = g.get_legal_moves();
  auto hand = g.player_hand(g.current_player());

  for (int mid : legal) {
    const auto &cost = MOVE_TO_CARDS[mid];
    for (int rank = 0; rank < 13; ++rank) {
      assert(hand[rank] >= cost[rank]);
    }
  }
}

// Verify that the played move was actually in the legal move list.
static void assert_move_was_legal(const Game &g_before_move, int move_played) {
  auto legal = g_before_move.get_legal_moves();
  bool found = false;
  for (int m : legal) {
    if (m == move_played) { found = true; break; }
  }
  assert(found);
}

// ------------------------------------------------------------------ tests

// Simulate N random-vs-random games and verify invariants throughout.
static void test_random_games_invariants() {
  constexpr int N_GAMES = 200;
  std::mt19937 rng(42);

  for (int game_idx = 0; game_idx < N_GAMES; ++game_idx) {
    Game g;
    g.shuffle_deal(rng);
    assert_card_conservation(g);

    int moves_played = 0;
    while (!g.is_over()) {
      assert_card_conservation(g);
      assert_legal_moves_affordable(g);

      auto legal = g.get_legal_moves();
      assert(!legal.empty());

      // Pick a random legal move.
      std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
      int chosen = legal[dist(rng)];

      // Verify it was legal before we play it.
      assert_move_was_legal(g, chosen);

      g.apply_move(chosen);
      ++moves_played;

      // Sanity: games can't run infinitely long (16+16 = 32 cards, at least
      // 1 card played per non-pass move; passes are bounded by structure).
      assert(moves_played <= 10000);
    }

    // Game over: exactly one player has an empty hand.
    assert_card_conservation(g);
    int winner = g.get_winner();
    assert(winner == 0 || winner == 1);
    assert(g.get_player_hand_size(winner) == 0);
    assert(g.get_player_hand_size(1 - winner) > 0);
  }
}

// Simulate N games via GameSimulator and verify the recorded turns are
// self-consistent.
static void test_random_games_via_simulator() {
  constexpr int N_GAMES = 100;
  std::mt19937 rng(7);

  RandomPlayerFactory f0(1), f1(2);
  int total_turns = 0;

  for (int game_idx = 0; game_idx < N_GAMES; ++game_idx) {
    GameSimulator sim(f0.create_player(), f1.create_player(), rng);
    GameRecord rec = sim.run();

    const auto &turns = rec.turns();
    assert(!turns.empty());

    // Each turn's legal_moves must be non-empty.
    for (const auto &turn : turns) {
      assert(!turn.legal_moves.empty());

      // The move that was played must appear in legal_moves.
      int played_id = encodeMove(turn.move);
      bool found = false;
      for (int m : turn.legal_moves) {
        if (m == played_id) { found = true; break; }
      }
      assert(found);
    }

    // Final game state in record must be over.
    assert(rec.game().is_over());

    total_turns += static_cast<int>(turns.size());
  }

  // Sanity: average game length should be plausible (greedy ~few dozen turns).
  int avg = total_turns / N_GAMES;
  assert(avg >= 2 && avg <= 5000);
}

// Verify PartialGame stays consistent with the full Game during a random game.
// At every turn, both PartialGame views are advanced in parallel with the Game.
static void test_partial_game_tracks_game() {
  constexpr int N_GAMES = 50;
  std::mt19937 rng(13);

  for (int game_idx = 0; game_idx < N_GAMES; ++game_idx) {
    Game g;
    g.shuffle_deal(rng);

    PartialGame views[2] = {PartialGame(g, 0), PartialGame(g, 1)};

    while (!g.is_over()) {
      int curr = g.current_player();

      // PartialGame's legal moves must match Game's legal moves (for current player).
      auto game_legal = g.get_legal_moves();
      auto part_legal = views[curr].get_legal_moves();
      assert(game_legal.size() == part_legal.size());
      // They should contain identical elements (order may differ).
      std::sort(game_legal.begin(), game_legal.end());
      std::sort(part_legal.begin(), part_legal.end());
      assert(game_legal == part_legal);

      // Pick a random legal move.
      std::uniform_int_distribution<size_t> dist(0, game_legal.size() - 1);
      int chosen = game_legal[dist(rng)];
      Move m(chosen);

      // Advance both.
      g.apply_move(m);
      views[0].apply_move(m);
      views[1].apply_move(m);

      // After the move, verify the views' hand sums are still sane.
      for (int p = 0; p < 2; ++p) {
        auto hand = views[p].player_hand();
        int sum = 0;
        for (int rank = 0; rank < 13; ++rank) {
          assert(hand[rank] >= 0);
          assert(hand[rank] <= max_cards_in_deck_for_rank(rank));
          sum += hand[rank];
        }
        // The view's hand size must match the Game's actual hand size for that player.
        assert(sum == g.get_player_hand_size(p));
      }
    }
  }
}

// The total number of non-pass cards played across a full game must equal 32
// (all dealt cards eventually enter the discard pile when one player wins).
static void test_cards_played_equals_dealt() {
  constexpr int N_GAMES = 100;
  std::mt19937 rng(21);

  for (int game_idx = 0; game_idx < N_GAMES; ++game_idx) {
    Game g;
    g.shuffle_deal(rng);

    while (!g.is_over()) {
      auto legal = g.get_legal_moves();
      std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
      g.apply_move(legal[dist(rng)]);
    }

    // At the end, winner has 0 cards; all 32 cards are accounted for between
    // the loser's hand and the discard pile.
    int winner = g.get_winner();
    int loser_cards = g.get_player_hand_size(1 - winner);
    auto dp = g.discard_pile();
    int discarded = 0;
    for (int rank = 0; rank < 13; ++rank)
      discarded += dp[rank];
    assert(loser_cards + discarded == 32);
  }
}

void run_random_game_tests() {
  test_random_games_invariants();
  test_random_games_via_simulator();
  test_partial_game_tracks_game();
  test_cards_played_equals_dealt();
}
