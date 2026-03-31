#include "game_simulator.h"
#include "greedy/greedy_player.h"
#include "greedy/greedy_player_factory.h"
#include "random/random_player_factory.h"
#include "util.h"

#include <cassert>
#include <memory>
#include <random>

// Single game: record is non-empty and every played move was in legal set.
static void test_greedy_vs_random_game_completes() {
  std::mt19937 rng(42);
  GreedyPlayerFactory f0;
  RandomPlayerFactory f1(99u);
  GameSimulator sim(f0.create_player(), f1.create_player(), rng);
  auto rec = sim.run();
  assert(!rec.turns().empty());
  assert(rec.game().is_over());

  for (const auto &turn : rec.turns()) {
    int played_id = encodeMove(turn.move);
    bool found = false;
    for (int m : turn.legal_moves) {
      if (m == played_id) { found = true; break; }
    }
    assert(found);
  }
}

// Greedy should win more often than random over many games.
static void test_greedy_beats_random() {
  constexpr int N = 500;
  int greedy_wins = 0;
  std::mt19937 rng(1234);

  GreedyPlayerFactory fg;
  RandomPlayerFactory fr(77u);

  for (int i = 0; i < N; ++i) {
    GameSimulator sim(fg.create_player(), fr.create_player(), rng);
    auto rec = sim.run();
    if (rec.game().get_winner() == 0) ++greedy_wins;
  }

  // Greedy should win the large majority when playing as player 0.
  assert(greedy_wins > N * 55 / 100);
}

// Greedy player must never make an illegal move.
static void test_greedy_never_plays_illegal_move() {
  constexpr int N = 100;
  std::mt19937 rng(8);
  GreedyPlayerFactory fg;
  RandomPlayerFactory fr(5u);

  for (int i = 0; i < N; ++i) {
    GameSimulator sim(fg.create_player(), fr.create_player(), rng);
    auto rec = sim.run();

    for (const auto &turn : rec.turns()) {
      int played_id = encodeMove(turn.move);
      bool found = false;
      for (int m : turn.legal_moves) {
        if (m == played_id) { found = true; break; }
      }
      assert(found);
    }
  }
}

void run_greedy_player_tests() {
  test_greedy_vs_random_game_completes();
  test_greedy_beats_random();
  test_greedy_never_plays_illegal_move();
}
