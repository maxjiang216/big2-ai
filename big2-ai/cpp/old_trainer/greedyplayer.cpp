#include "greedyplayer.h"

#include "move.h"
#include "game.h"
#include "util.h"

#include <bitset>
#include <iostream>
#include <fstream>
#include <cstdlib>
using namespace std;

GreedyPlayer::GreedyPlayer(int deal[32], int player_turn, int id): id_(id), player_turn_(player_turn) {
    // Initialize the two games
    games_[0] = Game(!player_turn, deal);
    games_[1] = Game(player_turn, deal + 16);
}

int GreedyPlayer::getNumSamples() const {
    return samples_.size();
}

int GreedyPlayer::writeSamples(float *stateBuffer, int *turnBuffer, int *legalMaskBuffer, float *valueBuffer, int *moveBuffer) const {
    // cout << "Writing samples for player " << id_ << endl;
    for (size_t i = 0; i < samples_.size(); ++i) {
        samples_[i].writeSample(stateBuffer + i * 26, turnBuffer + i, legalMaskBuffer + i * NEURAL_NETWORK_MOVES_SIZE, valueBuffer + i, moveBuffer + i);
    }
    return samples_.size();
}

int GreedyPlayer::writeSamplesComplex(float *stateBuffer, int *turnBuffer, int *legalMaskBuffer, float *valueBuffer, int *moveBuffer) const {
    // cout << "Writing samples for player " << id_ << endl;
    for (size_t i = 0; i < samples_.size(); ++i) {
        samples_[i].writeSampleComplex(stateBuffer + i * 53, turnBuffer + i, legalMaskBuffer + i * NEURAL_NETWORK_MOVES_SIZE, valueBuffer + i, moveBuffer + i);
    }
    return samples_.size();
}

void GreedyPlayer::playGame() {

    // // // Create a file to write logs
    // string filename = "logs/greedyplayer_" + to_string(id_) + ".log";
    // ofstream log_file(filename);

    // int counter = 0;

    while (true) {

        // Choose a move
        bitset<LEGAL_MOVES_SIZE> legal_moves_bitset;
        Move move = chooseMove(legal_moves_bitset);

        // log_file << "Game state: " << endl;
        // log_file << games_[player_turn_] << endl;

        // log_file << "Move: " << move << '\n' << endl;

        // Create samples
        // The move is forced iff the player passes
        // We also ignore double straights of length 8
        // The move is also "forced" if it is possible to play all cards
        // If the move is not forced, we create a sample for the player head
        if (move.combination != Move::Combination::kPass && move.combination != Move::Combination::kDoubleStraight8 && move.numCards() < games_[1 - player_turn_].getOpponentCardNum()) {
            samples_.push_back(Sample(player_turn_, 1, games_[player_turn_], move, legal_moves_bitset));
        }
        // We create a sample for the opponent head when it is not the first move
        // Note that the MCST will never need to evaluate an initial opponent move
        if (!(games_[player_turn_].getOpponentCardNum() == 16 && games_[1 - player_turn_].getOpponentCardNum() == 16)) {
            // Get possible opponent moves
            vector<int> opponent_moves = games_[1 - player_turn_].getLegalMoves();
            // It is possible we can deduce that the opponent must pass
            // In this case, we can likewise skip the sample
            if (opponent_moves.size() > 1) {
                legal_moves_bitset.reset();
                for (int move : opponent_moves) {
                    legal_moves_bitset[move] = true;
                }
                samples_.push_back(Sample(1 - player_turn_, 0, games_[1 - player_turn_], move, legal_moves_bitset));
            }
        }

        // Now, apply the move to both games
        for (int i = 0; i < 2; ++i) {
            games_[i].doMove(move);
        }

        // If the player has no cards left, the game is over
        // Only the player who just played can win
        if (games_[1 - player_turn_].getOpponentCardNum() == 0) {
            // Now, set the result for all samples
            for (Sample& sample : samples_) {
                // If the turns are different, the player at that turn lost
                sample.result = (player_turn_ + sample.player_turn + 1) % 2;
            }
            break;
        }

        player_turn_ = 1 - player_turn_;

        // if (counter++ > 100) {
        //     cout << "Game " << id_ << " ended after " << counter << " moves" << endl;
        //     break;
        // }
    }
}

Move GreedyPlayer::chooseMove(bitset<LEGAL_MOVES_SIZE> &legal_moves_bitset) const {
    // Get the legal moves
    vector<int> legal_moves = games_[player_turn_].getLegalMoves();

    // If the move is forced, we can return immediately
    // We also do not create a sample in this case
    // This also covers the case where the player must pass
    if (legal_moves.size() == 1) {
        return Move(legal_moves[0]);
    }

    // Get bitset of legal moves
    legal_moves_bitset.reset();
    for (int move : legal_moves) {
        legal_moves_bitset[move] = true;
    }

    // Otherwise, we use a greedy algorithm
    // In general, we want to maximize the number of cards played
    // and minimize the rank of the cards played

    // 16 cards
    // Double straights of length 8
    for (int i = 0; i < 5; ++i) {
        if (legal_moves_bitset[kDOUBLESTRAIGHT8_START + i]) {
            return Move(kDOUBLESTRAIGHT8_START + i);
        }
    }

    // 15 cards
    // Triple straights of length 5
    for (int i = 0; i < 8; ++i) {
        if (legal_moves_bitset[kTRIPLESTRAIGHT5_START + i]) {
            return Move(kTRIPLESTRAIGHT5_START + i);
        }
    }

    // 14 cards
    // Double straights of length 7
    for (int i = 0; i < 6; ++i) {
        if (legal_moves_bitset[kDOUBLESTRAIGHT7_START + i]) {
            return Move(kDOUBLESTRAIGHT7_START + i);
        }
    }

    // 13 cards
    // Straight of length 13
    if (legal_moves_bitset[kSTRAIGHT13_START]) {
        return Move(kSTRAIGHT13_START);
    }

    // 12 cards
    // Triple straights of length 4
    for (int i = 0; i < 9; ++i) {
        if (legal_moves_bitset[kTRIPLESTRAIGHT4_START + i]) {
            return Move(kTRIPLESTRAIGHT4_START + i);
        }
    }
    // Double straights of length 6
    for (int i = 0; i < 7; ++i) {
        if (legal_moves_bitset[kDOUBLESTRAIGHT6_START + i]) {
            return Move(kDOUBLESTRAIGHT6_START + i);
        }
    }
    // Straights of length 12
    for (int i = 0; i < 3; ++i) {
        if (legal_moves_bitset[kSTRAIGHT12_START + i]) {
            return Move(kSTRAIGHT12_START + i);
        }
    }

    // 11 cards
    // Straights of length 11
    for (int i = 0; i < 4; ++i) {
        if (legal_moves_bitset[kSTRAIGHT11_START + i]) {
            return Move(kSTRAIGHT11_START + i);
        }
    }

    // 10 cards
    // Double straights of length 5
    for (int i = 0; i < 8; ++i) {
        if (legal_moves_bitset[kDOUBLESTRAIGHT5_START + i]) {
            return Move(kDOUBLESTRAIGHT5_START + i);
        }
    }
    // Straights of length 10
    for (int i = 0; i < 5; ++i) {
        if (legal_moves_bitset[kSTRAIGHT10_START + i]) {
            return Move(kSTRAIGHT10_START + i);
        }
    }

    // 9 cards
    // Triple straights of length 3
    for (int i = 0; i < 10; ++i) {
        if (legal_moves_bitset[kTRIPLESTRAIGHT3_START + i]) {
            return Move(kTRIPLESTRAIGHT3_START + i);
        }
    }
    // Straights of length 9
    for (int i = 0; i < 6; ++i) {
        if (legal_moves_bitset[kSTRAIGHT9_START + i]) {
            return Move(kSTRAIGHT9_START + i);
        }
    }

    // 8 cards
    // Double straights of length 4
    for (int i = 0; i < 9; ++i) {
        if (legal_moves_bitset[kDOUBLESTRAIGHT4_START + i]) {
            return Move(kDOUBLESTRAIGHT4_START + i);
        }
    }
    // Straights of length 8
    for (int i = 0; i < 7; ++i) {
        if (legal_moves_bitset[kSTRAIGHT8_START + i]) {
            return Move(kSTRAIGHT8_START + i);
        }
    }

    // 7 cards
    // Straights of length 7
    for (int i = 0; i < 8; ++i) {
        if (legal_moves_bitset[kSTRAIGHT7_START + i]) {
            return Move(kSTRAIGHT7_START + i);
        }
    }

    // 6 cards
    // Triple straights of length 2
    for (int i = 0; i < 11; ++i) {
        if (legal_moves_bitset[kTRIPLESTRAIGHT2_START + i]) {
            return Move(kTRIPLESTRAIGHT2_START + i);
        }
    }
    // Double straights of length 3
    for (int i = 0; i < 10; ++i) {
        if (legal_moves_bitset[kDOUBLESTRAIGHT3_START + i]) {
            return Move(kDOUBLESTRAIGHT3_START + i);
        }
    }
    // Straights of length 6
    for (int i = 0; i < 9; ++i) {
        if (legal_moves_bitset[kSTRAIGHT6_START + i]) {
            return Move(kSTRAIGHT6_START + i);
        }
    }

    // 5 cards
    // Full house (rarer than straight of length 5)
    // Don't use ace as triple (better as bomb)
    for (int i = 0; i < 121; ++i) {
        if (legal_moves_bitset[kFULL_HOUSE_START + i]) {
            return Move(kFULL_HOUSE_START + i);
        }
    }
    // Straights of length 5
    for (int i = 0; i < 10; ++i) {
        if (legal_moves_bitset[kSTRAIGHT5_START + i]) {
            return Move(kSTRAIGHT5_START + i);
        }
    }
    // Play bomb first if can play all cards
    if (games_[1 - player_turn_].getOpponentCardNum() == 5) {
        // Bombs w/ extra, no ace bombs
        for (int i = 0; i < 143; ++i) {
            // Skip no extra bombs
            if (i % 13 == 0) continue;
            if (legal_moves_bitset[kBOMB_START + i]) {
                return Move(kBOMB_START + i);
            }
        }
    }

    // 4 cards
    // Double straights of length 2
    for (int i = 0; i < 11; ++i) {
        if (legal_moves_bitset[kDOUBLESTRAIGHT2_START + i]) {
            return Move(kDOUBLESTRAIGHT2_START + i);
        }
    }
    // Play bomb if can play all cards
    if (games_[1 - player_turn_].getOpponentCardNum() == 4) {
        // Ace bombs w/ extra
        for (int i = 0; i < 12; ++i) {
            if (legal_moves_bitset[kBOMB_START + 144 + i]) {
                return Move(kBOMB_START + i);
            }
        }
        // Other bombs w/o extra
        for (int r = 3; r <= 14; ++r) {
            if (legal_moves_bitset[kBOMB_START + (r - 3) * 13]) {
                return Move(kBOMB_START + (r - 3) * 13);
            }
        }
    }

    // 3 cards
    // Triples
    for (int i = 0; i < 12; ++i) {
        if (legal_moves_bitset[kTRIPLE_START + i]) {
            return Move(kTRIPLE_START + i);
        }
    }
    // Play ace bomb if can play all cards (this is always playable)
    if (games_[1 - player_turn_].getOpponentCardNum() == 3 && games_[player_turn_].getPlayerCardRank(14) == 3) {
        return Move(kBOMB_START + 143);
    }

    // 2 cards
    // Doubles
    for (int i = 0; i < 13; ++i) {
        if (legal_moves_bitset[kDOUBLE_START + i]) {
            return Move(kDOUBLE_START + i);
        }
    }

    // 1 card
    // Singles
    if (games_[player_turn_].getOpponentCardNum() <= 2) {
        for (int i = 12; i >= 0; --i) {
            if (legal_moves_bitset[kSINGLE_START + i]) {
                return Move(kSINGLE_START + i);
            }
        }
    }
    else {
        for (int i = 0; i < 13; ++i) {
            if (legal_moves_bitset[kSINGLE_START + i]) {
                return Move(kSINGLE_START + i);
            }
        }
    }

    // Bombs w/ extra
    for (int i = 0; i < 156; ++i) {
        // Skip no extra bombs
        if (i % 13 == 0) continue;
        if (legal_moves_bitset[kBOMB_START + i]) {
            return Move(kBOMB_START + i);
        }
    }
    // Bombs w/o extra
    for (int r = 3; r <= 14; ++r) {
        if (legal_moves_bitset[kBOMB_START + (r - 3) * 13]) {
            return Move(kBOMB_START + (r - 3) * 13);
        }
    }
    
    // This should never happen, but return pass
    return Move(kPASS);

}