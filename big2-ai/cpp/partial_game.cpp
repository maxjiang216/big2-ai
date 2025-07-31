#include "partial_game.h"
#include "util.h"  // for encodeMove and MOVE_TO_CARDS
#include <iomanip> // For std::setw
#include <iostream>

// Initialize with a given player hand; other state defaults
// Only used at the beginning of a game
PartialGame::PartialGame(const std::array<int, 13> &player_hand, int turn)
    : turn_(turn), // Indicates if it is currently the player's turn
      player_hand_(player_hand), opponent_card_count_(16), discard_pile_{},
      last_move_(Move::Combination::kPass) {
  for (int i = 0; i < 13; ++i) {
    discard_pile_[i] = 0;
  }
}

PartialGame::PartialGame(const Game &game, int player_num)
    : turn_(game.current_player() == player_num
                ? 0
                : 1), // turn 0 is the player's turn (PartialGame from their
                      // perspective)
      player_hand_(game.player_hand(player_num)),
      opponent_card_count_(game.get_player_hand_size(1 - player_num)),
      discard_pile_(game.discard_pile()), last_move_(game.last_move()) {}

// Apply a move: update hand/discard, opponent count, last_move, and flip turn
void PartialGame::apply_move(const Move &move) {
  // Determine card usage for this move
  int move_id = encodeMove(move);
  const auto &cost = MOVE_TO_CARDS.at(move_id); // array<int,13>

  // Track total cards played this move
  int total_played = 0;

  // Subtract from player's hand if it's their turn,
  // otherwise reduce opponent's unknown count
  if (turn_ == 0) {
    for (int i = 0; i < 13; ++i) {
      int cnt = cost[i];
      player_hand_[i] -= cnt;
      discard_pile_[i] += cnt;
      total_played += cnt;
    }
  } else {
    for (int i = 0; i < 13; ++i) {
      int cnt = cost[i];
      discard_pile_[i] += cnt;
    }
    // Last index holds total
    opponent_card_count_ -= cost[13];
  }

  // Record last move and switch turn
  last_move_ = move;
  turn_ = 1 - turn_;
}

std::vector<int> PartialGame::get_legal_moves() const {
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
      if (player_hand_[rank] < MOVE_TO_CARDS.at(move_id)[rank]) {
        is_legal = false;
        break;
      }
    }
    if (is_legal)
      legal_moves.push_back(move_id);
  }
  return legal_moves;
}

std::vector<int> PartialGame::get_possible_moves() const {
  std::vector<int> possible_moves;
  possible_moves.reserve(32);
  // We can pass if it's not a new trick (a.k.a. the last move was a pass)
  if (last_move_.combination != Move::Combination::kPass) {
    possible_moves.push_back(kPASS);
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
    // Check total cards
    bool is_legal = opponent_card_count_ >= MOVE_TO_CARDS.at(move_id)[13];
    if (is_legal) {
      for (int rank = 0; rank < 13; ++rank) {
        if ((rank < 11 && 4 - player_hand_[rank] - discard_pile_[rank] <
                              MOVE_TO_CARDS.at(move_id)[rank]) ||
            (rank == 11 && 3 - player_hand_[rank] - discard_pile_[rank] <
                               MOVE_TO_CARDS.at(move_id)[rank]) ||
            (rank == 12 && 1 - player_hand_[rank] - discard_pile_[rank] <
                               MOVE_TO_CARDS.at(move_id)[rank])) {
          is_legal = false;
          break;
        }
      }
    }
    if (is_legal)
      possible_moves.push_back(move_id);
  }
  return possible_moves;
}

std::vector<int> PartialGame::get_possible_moves_not_bomb() const {
  std::vector<int> possible_moves;
  possible_moves.reserve(32);
  // We can pass if it's not a new trick (a.k.a. the last move was a pass)
  if (last_move_.combination != Move::Combination::kPass) {
    possible_moves.push_back(kPASS);
  }
  for (int move_id = kSINGLE_START; move_id < LEGAL_MOVES_SIZE; ++move_id) {
    Move move(move_id);
    if (move.combination == Move::Combination::kBomb)
      continue;
    // If it is a new trick, any trick can be played
    if (last_move_.combination != Move::Combination::kPass) {
      // If not bomb, needs to be same type and larger rank
      if (move.combination != last_move_.combination ||
          move.rank <= last_move_.rank)
        continue;
    }
    // Check total cards
    bool is_legal = opponent_card_count_ >= MOVE_TO_CARDS.at(move_id)[13];
    if (is_legal) {
      for (int rank = 0; rank < 13; ++rank) {
        if ((rank < 11 && 4 - player_hand_[rank] - discard_pile_[rank] <
                              MOVE_TO_CARDS.at(move_id)[rank]) ||
            (rank == 11 && 3 - player_hand_[rank] - discard_pile_[rank] <
                               MOVE_TO_CARDS.at(move_id)[rank]) ||
            (rank == 12 && 1 - player_hand_[rank] - discard_pile_[rank] <
                               MOVE_TO_CARDS.at(move_id)[rank])) {
          is_legal = false;
          break;
        }
      }
      if (is_legal)
        possible_moves.push_back(move_id);
    }
  }
  return possible_moves;
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
  int trick_rank = -1;
  if (last_move_.combination != Move::Combination::kPass) {
    trick_rank = last_move_.rank;
  }
  return trick_rank;
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