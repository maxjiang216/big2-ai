#include "partial_game.h"
#include "util.h"

#include <iomanip>
#include <iostream>

PartialGame::PartialGame(const std::array<int, 13> &player_hand, int turn)
    : turn_(turn), player_hand_(player_hand), opponent_card_count_(16),
      discard_pile_{}, last_move_(Move::Combination::kPass) {
  for (int i = 0; i < 13; ++i) {
    discard_pile_[i] = 0;
  }
}

PartialGame::PartialGame(const Game &game, int player_num)
    : turn_(game.current_player() == player_num ? 0 : 1),
      player_hand_(game.player_hand(player_num)),
      opponent_card_count_(game.get_player_hand_size(1 - player_num)),
      discard_pile_(game.discard_pile()), last_move_(game.last_move()),
      last_move_id_(game.last_move_id()) {}

void PartialGame::apply_move(const Move &move) {
  int move_id = encodeMove(move);
  const auto &cost = MOVE_TO_CARDS[move_id];

  if (turn_ == 0) {
    for (int i = 0; i < 13; ++i) {
      int cnt = cost[i];
      player_hand_[i] -= cnt;
      discard_pile_[i] += cnt;
    }
  } else {
    for (int i = 0; i < 13; ++i) {
      int cnt = cost[i];
      discard_pile_[i] += cnt;
    }
    opponent_card_count_ -= cost[13];
  }

  last_move_ = move;
  last_move_id_ = move_id;
  turn_ = 1 - turn_;
}

std::vector<int> PartialGame::get_legal_moves() const {
  return compute_legal_moves(player_hand_, last_move_id_);
}

void PartialGame::get_legal_moves_into(std::vector<int> &out) const {
  compute_legal_moves_into(player_hand_, last_move_id_, out);
}

std::vector<int> PartialGame::get_possible_moves() const {
  return compute_possible_moves(player_hand_, discard_pile_, opponent_card_count_,
                                last_move_id_, false);
}

std::vector<int> PartialGame::get_possible_moves_not_bomb() const {
  return compute_possible_moves(player_hand_, discard_pile_, opponent_card_count_,
                                last_move_id_, true);
}

int PartialGame::count_bombs() const {
  int n_bombs = player_hand_[11] == 3 ? 1 : 0;
  for (int i = 0; i < 11; ++i) {
    if (player_hand_[i] == 4)
      ++n_bombs;
  }
  return n_bombs;
}

int PartialGame::get_trick_rank() const {
  if (last_move_.combination != Move::Combination::kPass) {
    return last_move_.rank;
  }
  return -1;
}

std::ostream &operator<<(std::ostream &os, const PartialGame &g) {
  os << "--- PartialGame State ---\n";
  os << "Turn: " << (g.turn_ == 0 ? "Player" : "Opponent") << "\n";
  os << "Your Hand: [";
  for (int i = 0; i < 13; ++i) {
    for (int c = 0; c < g.player_hand_[i]; ++c)
      os << rankToChar(i + 3);
  }
  os << "]\n";

  os << "Opponent cards left: " << g.opponent_card_count_ << "\n";
  os << "Discards: [";
  for (int i = 0; i < 13; ++i) {
    for (int c = 0; c < g.discard_pile_[i]; ++c)
      os << rankToChar(i + 3);
  }
  os << "]\n";

  os << "Last move: ";
  os << g.last_move_ << "\n";

  return os;
}
