#ifndef GAME_H
#define GAME_H

#include "move.h"
#include "util.h"

#include <array>
#include <random>
#include <vector>

class Game {
public:
  Game();

  // Construct a Game from an explicit full state (used by PIMC rollouts).
  // hand_size_ is derived from the hand arrays.
  Game(std::array<int, 13> hand0, std::array<int, 13> hand1,
       std::array<int, 13> discard, Move last_move, int current_player);

  void shuffle_deal(std::mt19937 &rng);

  int current_player() const;

  // Override which seat leads (used to apply the series opening rule; default
  // after shuffle_deal is seat 0).
  void set_first_player(int player);

  bool is_over() const;

  void apply_move(const Move &move);
  void apply_move(int move_id);

  int get_winner() const;

  std::array<int, 13> player_hand(int player) const;
  HandBits player_hand_bits(int player) const;

  int get_player_hand_size(int player) const;

  std::array<int, 13> discard_pile() const;

  Move last_move() const;
  int  last_move_id() const;

  std::vector<int> get_legal_moves() const;

private:
  HandBits hand_bits_[2]{};
  std::array<int, 13> discard_pile_{};
  int hand_size_[2]{0, 0};
  int current_player_{0};
  Move last_move_{Move::Combination::kPass};

  friend std::ostream &operator<<(std::ostream &os, const Game &game);
};

#endif
