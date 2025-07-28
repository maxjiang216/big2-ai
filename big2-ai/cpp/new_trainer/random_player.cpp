#include "random_player.h"
#include "move.h"
#include <array>
#include <stdexcept>
#include <iostream>
using namespace std;

// Constructor: non-deterministic seed
RandomPlayer::RandomPlayer() : rng_(std::random_device{}()), game_() {}

// Constructor: deterministic seed
RandomPlayer::RandomPlayer(unsigned int seed) : rng_(seed), game_() {}

// Accept the initial deal: reset game state with this hand
void RandomPlayer::accept_deal(std::array<int, 13> hand, int turn) {
  // Initialize PartialGame
  game_ = PartialGame(hand, turn);
}

// Update state for opponent's move
void RandomPlayer::accept_opponent_move(const Move &move) {
  game_.apply_move(move);
}

// Select a random legal move, apply it, and return it
Move RandomPlayer::select_move() {
  std::vector<int> legal = game_.get_legal_moves();
  std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
  size_t idx = dist(rng_);
  int move_id = legal[idx];
  Move m(move_id);
  game_.apply_move(m);
  return m;
}
