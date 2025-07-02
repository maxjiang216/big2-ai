#include "partial_game.h"
#include "util.h"    // for encodeMove and MOVE_TO_CARDS

// Default constructor: empty state, player's turn, full opponent hand
PartialGame::PartialGame()
    : turn(true),
      player_hand_{},
      opponent_card_count_(16),
      discard_pile_{},
      last_move_(Move::Combination::kPass)
{
}

// Initialize with a given player hand; other state defaults
PartialGame::PartialGame(const std::array<int,13>& player_hand, bool turn)
    : turn(turn),
      player_hand_(player_hand),
      opponent_card_count_(16),
      discard_pile_{},
      last_move_(Move::Combination::kPass)
{
}

// Apply a move: update hand/discard, opponent count, last_move, and flip turn
void PartialGame::apply_move(const Move& move) {
    // Determine card usage for this move
    int move_id = encodeMove(move);
    const auto& cost = MOVE_TO_CARDS.at(move_id); // array<int,13>

    // Track total cards played this move
    int total_played = 0;

    // Subtract from player's hand if it's their turn,
    // otherwise reduce opponent's unknown count
    if (turn) {
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
    turn = !turn;
}

std::vector<int> PartialGame::get_legal_moves() const {
    vector<int> legal_moves;
    legal_moves.reserve(32);
    // We can pass if it's not a new trick (a.k.a. the last move was a pass)
    if (last_move_.combination != Move::Combination::kPass) {
        legal_moves.push_back(kPASS);
    }
    for (int move_id = kSINGLE_START; move_id < LEGAL_MOVES_SIZE; ++move_id) {
        Move move(move_id);
        if (move.combination != Move::Combination::kBomb) {
            // If not bomb, needs to be same type and larger rank
            if (move.combination != last_move_.combination || move.rank <= last_move_.rank) continue;
        }
        else {
            // If bomb, just needs to be larger than possible last bomb
            if (last_move_.combination == Move::Combination::kBomb && move.rank <= last_move_.rank) continue;
        }
        bool is_legal = true;
        for (int rank = 0; rank < 13; ++rank) {
            if (hands_[current_player_][rank] < MOVE_TO_CARDS.at(move_id)[rank]) {
                is_legal = false;
                break;
            }
        }
        if (is_legal) legal_moves.push_back(move_id);
    }
    return legal_moves;
}