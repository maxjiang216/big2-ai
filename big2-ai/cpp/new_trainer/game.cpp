// game.cpp
#include "game.h"
#include "util.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>
#include <iostream>
using namespace std;

Game::Game() : hands_{}, current_player_(0) {}

void Game::shuffle_deal(std::mt19937 &rng) {
  std::vector<int> deck;
  // Build deck: ranks 0..12 (3..2)
  for (int r = 0; r < 13; ++r) {
    int copies = (r == 11 ? 3 : (r == 12 ? 1 : 4));
    for (int i = 0; i < copies; ++i)
      deck.push_back(r);
  }
  // Shuffle
  std::shuffle(deck.begin(), deck.end(), rng);

  // Clear hands
  for (auto &h : hands_)
    h.fill(0);

  // Deal 16 cards each
  for (int i = 0; i < 16; ++i)
    hands_[0][deck[i]]++;
  for (int i = 16; i < 32; ++i)
    hands_[1][deck[i]]++;

  // Set all discards to 0
  for (int i = 0; i < 13; ++i) {
    discard_pile_[i] = 0;
  }

  // First player fixed (initiative logic can be added later)
  current_player_ = 0;
}

int Game::current_player() const { return current_player_; }

bool Game::is_over() const {
  // Game ends when one hand is empty
  return std::all_of(hands_[0].begin(), hands_[0].end(),
                     [](int c) { return c == 0; }) ||
         std::all_of(hands_[1].begin(), hands_[1].end(),
                     [](int c) { return c == 0; });
}

int Game::get_winner() const {
  // Player with no cards wins
  if (std::all_of(hands_[0].begin(), hands_[0].end(),
                  [](int c) { return c == 0; }))
    return 0;
  else
    return 1;
}

std::array<int, 13> Game::player_hand(int player) const {
  return hands_[player];
}

int Game::get_player_hand_size(int player) const {
  int sum = 0;
  for (int c : hands_[player])
    sum += c;
  return sum;
}

std::array<int, 13> Game::discard_pile() const { return discard_pile_; }

Move Game::last_move() const { return last_move_; }

void Game::apply_move(const Move &move) {

  int move_id = encodeMove(move);

  for (int rank = 0; rank < 13; ++rank) {
    hands_[current_player_][rank] -= MOVE_TO_CARDS.at(move_id)[rank];
    assert(hands_[current_player_][rank] >= 0);
    discard_pile_[rank] += MOVE_TO_CARDS.at(move_id)[rank];
    assert(discard_pile_[rank] <= 4);
    
  }
  last_move_ = move;
  // Advance turn (simplest: alternate)
  current_player_ = 1 - current_player_;
}

std::vector<int> Game::get_legal_moves() const {
  std::vector<int> legal_moves;
  legal_moves.reserve(32);
  // We can pass if it's not a new trick (a.k.a. the last move was a pass)
  if (last_move_.combination != Move::Combination::kPass) {
    legal_moves.push_back(kPASS);
  }
  for (int move_id = kSINGLE_START; move_id < LEGAL_MOVES_SIZE; ++move_id) {
    Move move(move_id);
    // If it is a new trick, any trick can be played
    if (last_move_.combination != Move::Combination::kPass) {
      if (move.combination != Move::Combination::kBomb) {
        // If not bomb, needs to be same type and larger rank
        if (move.combination != last_move_.combination ||
            move.rank <= last_move_.rank)
          continue;
      } else {
        // If bomb, just needs to be larger than possible last bomb
        if (last_move_.combination == Move::Combination::kBomb &&
            move.rank <= last_move_.rank)
          continue;
      }
    }
    bool is_legal = true;
    for (int rank = 0; rank < 13; ++rank) {
      if (hands_[current_player_][rank] < MOVE_TO_CARDS.at(move_id)[rank]) {
        is_legal = false;
        break;
      }
    }
    if (is_legal)
      legal_moves.push_back(move_id);
  }
  return legal_moves;
}

std::ostream &operator<<(std::ostream &os, const Game &game) {
    os << "==== Big 2 Game State ====\n";

    // Print player hands
    for (int p = 0; p < 2; ++p) {
        os << "Player " << p << " hand (" << game.get_player_hand_size(p) << "): [";
        for (int i = 0; i < 13; ++i) {
            for (int c = 0; c < game.hands_[p][i]; ++c) {
                os << rankToChar(i + 3);
            }
        }
        os << "]\n";
    }

    // Discard pile
    os << "Discards: [";
    for (int i = 0; i < 13; ++i) {
        for (int c = 0; c < game.discard_pile_[i]; ++c)
            os << rankToChar(i + 3);
    }
    os << "]\n";

    // Turn & last move
    os << "Current turn: Player " << game.current_player_ << "\n";
    os << "Last move: " << game.last_move_ << "\n";

    if (game.is_over()) {
        os << "*** Game Over! Winner: Player " << game.get_winner() << " ***\n";
    }

    return os;
}