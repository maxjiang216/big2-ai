#ifndef PARTIAL_GAME_H
#define PARTIAL_GAME_H

#include "game.h"
#include "move.h"

#include <array>
#include <vector>

class PartialGame {
public:
  PartialGame() = default;

  PartialGame(const std::array<int, 13> &player_hand, int turn);

  PartialGame(const Game &game, int player_num);

  void apply_move(const Move &move);

  std::vector<int> get_legal_moves() const;
  std::vector<int> get_possible_moves() const;
  std::vector<int> get_possible_moves_not_bomb() const;

  std::array<int, 13> player_hand() const { return player_hand_; }
  std::array<int, 13> discard_pile() const { return discard_pile_; }

  int count_bombs() const;

  int get_trick_rank() const;

  Move last_move() const { return last_move_; }

  int turn() const { return turn_; }

  int opponent_hand_size() const { return opponent_card_count_; }

  int num_cards() const {
    int total = 0;
    for (int c : player_hand_) total += c;
    return total;
  }

private:
  int turn_{};
  std::array<int, 13> player_hand_{};
  int opponent_card_count_{16};
  std::array<int, 13> discard_pile_{};
  Move last_move_{Move::Combination::kPass};

  friend std::ostream &operator<<(std::ostream &, const PartialGame &);
};

#endif
