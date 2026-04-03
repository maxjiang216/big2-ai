#include "game.h"
#include "util.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

Game::Game() : hands_{}, hand_size_{0, 0}, current_player_(0) {}

void Game::shuffle_deal(std::mt19937 &rng) {
  std::vector<int> deck;
  for (int r = 0; r < 13; ++r) {
    int copies = (r == 11 ? 3 : (r == 12 ? 1 : 4));
    for (int i = 0; i < copies; ++i)
      deck.push_back(r);
  }
  std::shuffle(deck.begin(), deck.end(), rng);

  for (auto &h : hands_)
    h.fill(0);

  for (int i = 0; i < 16; ++i)
    hands_[0][deck[i]]++;
  for (int i = 16; i < 32; ++i)
    hands_[1][deck[i]]++;

  for (int i = 0; i < 13; ++i) {
    discard_pile_[i] = 0;
  }

  hand_size_[0] = 16;
  hand_size_[1] = 16;
  // TODO: first player by lowest card / house rules; fixed for now.
  current_player_ = 0;
}

int Game::current_player() const { return current_player_; }

bool Game::is_over() const {
  return hand_size_[0] == 0 || hand_size_[1] == 0;
}

int Game::get_winner() const {
  if (hand_size_[0] == 0)
    return 0;
  return 1;
}

std::array<int, 13> Game::player_hand(int player) const {
  return hands_[player];
}

int Game::get_player_hand_size(int player) const { return hand_size_[player]; }

std::array<int, 13> Game::discard_pile() const { return discard_pile_; }

Move Game::last_move() const { return last_move_; }

void Game::apply_move(const Move &move) { apply_move(encodeMove(move)); }

void Game::apply_move(int move_id) {
  assert(move_id >= 0 && move_id < LEGAL_MOVES_SIZE);
  const auto &cost = MOVE_TO_CARDS[move_id];

  for (int rank = 0; rank < 13; ++rank) {
    hands_[current_player_][rank] -= cost[rank];
    assert(hands_[current_player_][rank] >= 0);
    discard_pile_[rank] += cost[rank];
    assert(discard_pile_[rank] <= max_cards_in_deck_for_rank(rank));
  }
  hand_size_[current_player_] -= cost[13];
  assert(hand_size_[current_player_] >= 0);

  last_move_ = Move(move_id);
  current_player_ = 1 - current_player_;
}

std::vector<int> Game::get_legal_moves() const {
  return compute_legal_moves(hands_[current_player_], last_move_);
}

std::ostream &operator<<(std::ostream &os, const Game &game) {
  os << "==== Big 2 Game State ====\n";

  for (int p = 0; p < 2; ++p) {
    os << "Player " << p << " hand (" << game.get_player_hand_size(p)
       << "): [";
    for (int i = 0; i < 13; ++i) {
      for (int c = 0; c < game.hands_[p][i]; ++c) {
        os << rankToChar(i + 3);
      }
    }
    os << "]\n";
  }

  os << "Discards: [";
  for (int i = 0; i < 13; ++i) {
    for (int c = 0; c < game.discard_pile_[i]; ++c)
      os << rankToChar(i + 3);
  }
  os << "]\n";

  os << "Current turn: Player " << game.current_player_ << "\n";
  os << "Last move: " << game.last_move_ << "\n";

  if (game.is_over()) {
    os << "*** Game Over! Winner: Player " << game.get_winner() << " ***\n";
  }

  return os;
}
