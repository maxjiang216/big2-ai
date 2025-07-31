#ifndef PARTIAL_GAME_H
#define PARTIAL_GAME_H

#include "game.h"
#include "move.h" // Represents a single play (type, rank, length, etc.)
#include <array>
#include <vector>

/**
 * @brief Player's partial view of the Big 2 game state.
 * Contains only information available to the player:
 *  - Their own hand
 *  - How many cards opponent has left
 *  - Cards played/discarded so far
 *  - Whose turn it is
 *  - The last move played
 */
class PartialGame {
public:
  /**
   * @brief Construct an empty PartialGame state.
   */
  PartialGame() = default;

  PartialGame(const std::array<int, 13> &player_hand, int turn);

  PartialGame(const Game &game, int player_num);

  void apply_move(const Move &move);

  std::vector<int> get_legal_moves() const;
  std::vector<int> get_possible_moves() const;
  std::vector<int> get_possible_moves_not_bomb() const;

  std::array<int, 13> player_hand() const { return player_hand_; }

  int count_bombs() const;

  int get_trick_rank() const;

  Move last_move() const { return last_move_; }

  int turn() const { return turn_; }

private:
  int turn_;                           // 0 if player's turn
  std::array<int, 13> player_hand_{};  // This player's cards (by rank)
  int opponent_card_count_{16};        // Opponent's cards left
  std::array<int, 13> discard_pile_{}; // Cards played so far (by rank)
  Move last_move_{Move::Combination::kPass};

  friend std::ostream &operator<<(std::ostream &, const PartialGame &);
};

#endif // PARTIAL_GAME_H
