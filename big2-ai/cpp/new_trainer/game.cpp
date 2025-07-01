// game.cpp
#include "game.h"
#include <algorithm>
#include <numeric>
#include <random>

Game::Game()
    : deck_(), hands_{}, current_player_(0) {
}

void Game::shuffle_deal(std::mt19937 &rng) {
    deck_.clear();
    // Build deck: ranks 0..12 (3..2)
    for(int r = 0; r < 13; ++r) {
        int copies = (r == 11 ? 3 : (r == 12 ? 1 : 4));
        for(int i = 0; i < copies; ++i)
            deck_.push_back(r);
    }
    // Shuffle
    std::shuffle(deck_.begin(), deck_.end(), rng);
    
    // Clear hands
    for(auto &h : hands_) h.fill(0);

    // Deal 16 cards each
    for(int i = 0; i < 16; ++i)
        hands_[0][deck_[i]]++;
    for(int i = 16; i < 32; ++i)
        hands_[1][deck_[i]]++;

    // First player fixed (initiative logic can be added later)
    current_player_ = 0;
}

int Game::current_player() const {
    return current_player_;
}

bool Game::is_over() const {
    // Game ends when one hand is empty
    return std::all_of(hands_[0].begin(), hands_[0].end(), [](int c){ return c == 0; }) ||
           std::all_of(hands_[1].begin(), hands_[1].end(), [](int c){ return c == 0; });
}

void Game::apply_move(const Move &move) {
    
       // Single
    if (move.combination == Move::Combination::kSingle) {
        --hands_[current_player_][move.rank - 3];
        ++discard_pile_[move.rank - 3];
    }
    // Double
    else if (move.combination == Move::Combination::kDouble) {
        hands_[current_player_][move.rank - 3] -= 2;
        discard_pile_[move.rank - 3] += 2;
    }
    // Triple
    else if (move.combination == Move::Combination::kTriple) {
        hands_[current_player_][move.rank - 3] -= 3;
        discard_pile_[move.rank - 3] += 3;
    }
    // Full house
    if (move.combination == Move::Combination::kFullHouse) {
        hands_[current_player_][move.rank - 3] -= 3;
        hands_[current_player_][move.auxiliary - 3] -= 2;
        discard_pile_[move.rank - 3] += 3;
        discard_pile_[move.auxiliary - 3] += 2;
    }
    // Bomb
    if (move.combination == Move::Combination::kBomb) {
        // Always uses all the cards
        hands_[current_player_][move.rank - 3] = 0;
        if (move.rank == 14) {
            discard_pile_[11] = 3;
        }
        else {
            discard_pile_[move.rank - 3] = 4;
        }
        // Optional add-on card
        if (move.auxiliary != 0) {
            --hands_[current_player_][move.auxiliary - 3];
            ++discard_pile_[move.auxiliary - 3];
        }
    }
    // Straight of length 5
    if (move.combination == Move::Combination::kStraight5) {
        --hands_[current_player_][move.rank - 3];
        --hands_[current_player_][move.rank - 4];
        --hands_[current_player_][move.rank - 5];
        --hands_[current_player_][move.rank - 6];
        --hands_[current_player_][(move.rank + 6) % 13];
        ++discard_pile_[move.rank - 3];
        ++discard_pile_[move.rank - 4];
        ++discard_pile_[move.rank - 5];
        ++discard_pile_[move.rank - 6];
        ++discard_pile_[(move.rank + 6) % 13];
    }
    // Straight of length 6
    if (move.combination == Move::Combination::kStraight6) {
        --hands_[current_player_][move.rank - 3];
        --hands_[current_player_][move.rank - 4];
        --hands_[current_player_][move.rank - 5];
        --hands_[current_player_][move.rank - 6];
        --hands_[current_player_][move.rank - 7];
        --hands_[current_player_][(move.rank + 5) % 13];
        ++discard_pile_[move.rank - 3];
        ++discard_pile_[move.rank - 4];
        ++discard_pile_[move.rank - 5];
        ++discard_pile_[move.rank - 6];
        ++discard_pile_[move.rank - 7];
        ++discard_pile_[(move.rank + 5) % 13];
    }
    // Straight of length 7
    if (move.combination == Move::Combination::kStraight7) {
        --hands_[current_player_][move.rank - 3];
        --hands_[current_player_][move.rank - 4];
        --hands_[current_player_][move.rank - 5];
        --hands_[current_player_][move.rank - 6];
        --hands_[current_player_][move.rank - 7];
        --hands_[current_player_][move.rank - 8];
        --hands_[current_player_][(move.rank + 4) % 13];
        ++discard_pile_[move.rank - 3];
        ++discard_pile_[move.rank - 4];
        ++discard_pile_[move.rank - 5];
        ++discard_pile_[move.rank - 6];
        ++discard_pile_[move.rank - 7];
        ++discard_pile_[move.rank - 8];
        ++discard_pile_[(move.rank + 4) % 13];
    }
    // Straight of length 8
    if (move.combination == Move::Combination::kStraight8) {
        --hands_[current_player_][move.rank - 3];
        --hands_[current_player_][move.rank - 4];
        --hands_[current_player_][move.rank - 5];
        --hands_[current_player_][move.rank - 6];
        --hands_[current_player_][move.rank - 7];
        --hands_[current_player_][move.rank - 8];
        --hands_[current_player_][move.rank - 9];
        --hands_[current_player_][(move.rank + 3) % 13];
        ++discard_pile_[move.rank - 3];
        ++discard_pile_[move.rank - 4];
        ++discard_pile_[move.rank - 5];
        ++discard_pile_[move.rank - 6];
        ++discard_pile_[move.rank - 7];
        ++discard_pile_[move.rank - 8];
        ++discard_pile_[move.rank - 9];
        ++discard_pile_[(move.rank + 3) % 13];
    }

    // Straight of length 9
    if (move.combination == Move::Combination::kStraight9) {
        --hands_[current_player_][move.rank - 3];
        --hands_[current_player_][move.rank - 4];
        --hands_[current_player_][move.rank - 5];
        --hands_[current_player_][move.rank - 6];
        --hands_[current_player_][move.rank - 7];
        --hands_[current_player_][move.rank - 8];
        --hands_[current_player_][move.rank - 9];
        --hands_[current_player_][move.rank - 10];
        --hands_[current_player_][(move.rank + 2) % 13];
        ++discard_pile_[move.rank - 3];
        ++discard_pile_[move.rank - 4];
        ++discard_pile_[move.rank - 5];
        ++discard_pile_[move.rank - 6];
        ++discard_pile_[move.rank - 7];
        ++discard_pile_[move.rank - 8];
        ++discard_pile_[move.rank - 9];
        ++discard_pile_[move.rank - 10];
        ++discard_pile_[(move.rank + 2) % 13];
    }

    // Straight of length 10
    if (move.combination == Move::Combination::kStraight10) {
        --hands_[current_player_][move.rank - 3];
        --hands_[current_player_][move.rank - 4];
        --hands_[current_player_][move.rank - 5];
        --hands_[current_player_][move.rank - 6];
        --hands_[current_player_][move.rank - 7];
        --hands_[current_player_][move.rank - 8];
        --hands_[current_player_][move.rank - 9];
        --hands_[current_player_][move.rank - 10];
        --hands_[current_player_][move.rank - 11];
        --hands_[current_player_][(move.rank + 1) % 13];
        ++discard_pile_[move.rank - 3];
        ++discard_pile_[move.rank - 4];
        ++discard_pile_[move.rank - 5];
        ++discard_pile_[move.rank - 6];
        ++discard_pile_[move.rank - 7];
        ++discard_pile_[move.rank - 8];
        ++discard_pile_[move.rank - 9];
        ++discard_pile_[move.rank - 10];
        ++discard_pile_[move.rank - 11];
        ++discard_pile_[(move.rank + 1) % 13];
    }

    // Straight of length 11
    if (move.combination == Move::Combination::kStraight11) {
        --hands_[current_player_][move.rank - 3];
        --hands_[current_player_][move.rank - 4];
        --hands_[current_player_][move.rank - 5];
        --hands_[current_player_][move.rank - 6];
        --hands_[current_player_][move.rank - 7];
        --hands_[current_player_][move.rank - 8];
        --hands_[current_player_][move.rank - 9];
        --hands_[current_player_][move.rank - 10];
        --hands_[current_player_][move.rank - 11];
        --hands_[current_player_][move.rank - 12];
        --hands_[current_player_][move.rank % 13];
        ++discard_pile_[move.rank - 3];
        ++discard_pile_[move.rank - 4];
        ++discard_pile_[move.rank - 5];
        ++discard_pile_[move.rank - 6];
        ++discard_pile_[move.rank - 7];
        ++discard_pile_[move.rank - 8];
        ++discard_pile_[move.rank - 9];
        ++discard_pile_[move.rank - 10];
        ++discard_pile_[move.rank - 11];
        ++discard_pile_[move.rank - 12];
        ++discard_pile_[move.rank % 13];
    }

    // Straight of length 12
    if (move.combination == Move::Combination::kStraight12) {
        --hands_[current_player_][move.rank - 3];
        --hands_[current_player_][move.rank - 4];
        --hands_[current_player_][move.rank - 5];
        --hands_[current_player_][move.rank - 6];
        --hands_[current_player_][move.rank - 7];
        --hands_[current_player_][move.rank - 8];
        --hands_[current_player_][move.rank - 9];
        --hands_[current_player_][move.rank - 10];
        --hands_[current_player_][move.rank - 11];
        --hands_[current_player_][move.rank - 12];
        --hands_[current_player_][move.rank - 13];
        --hands_[current_player_][(move.rank - 1) % 13];
        ++discard_pile_[move.rank - 3];
        ++discard_pile_[move.rank - 4];
        ++discard_pile_[move.rank - 5];
        ++discard_pile_[move.rank - 6];
        ++discard_pile_[move.rank - 7];
        ++discard_pile_[move.rank - 8];
        ++discard_pile_[move.rank - 9];
        ++discard_pile_[move.rank - 10];
        ++discard_pile_[move.rank - 11];
        ++discard_pile_[move.rank - 12];
        ++discard_pile_[move.rank - 13];
        ++discard_pile_[(move.rank - 1) % 13];
    }
    // Straight of length 13
    if (move.combination == Move::Combination::kStraight13) {
        for (int i = 0; i < 13; ++i) {
            --hands_[current_player_][i];
            ++discard_pile_[i];
        }
    }
    // Sisters of length 2
    if (move.combination == Move::Combination::kDoubleStraight2) {
        hands_[current_player_][move.rank - 3] -= 2;
        hands_[current_player_][move.rank - 4] -= 2;
        discard_pile_[move.rank - 3] += 2;
        discard_pile_[move.rank - 4] += 2;
    }
    // Sisters of length 3
    if (move.combination == Move::Combination::kDoubleStraight3) {
        hands_[current_player_][move.rank - 3] -= 2;
        hands_[current_player_][move.rank - 4] -= 2;
        hands_[current_player_][move.rank - 5] -= 2;
        discard_pile_[move.rank - 3] += 2;
        discard_pile_[move.rank - 4] += 2;
        discard_pile_[move.rank - 5] += 2;
    }
    // Sisters of length 4
    if (move.combination == Move::Combination::kDoubleStraight4) {
        hands_[current_player_][move.rank - 3] -= 2;
        hands_[current_player_][move.rank - 4] -= 2;
        hands_[current_player_][move.rank - 5] -= 2;
        hands_[current_player_][move.rank - 6] -= 2;
        discard_pile_[move.rank - 3] += 2;
        discard_pile_[move.rank - 4] += 2;
        discard_pile_[move.rank - 5] += 2;
        discard_pile_[move.rank - 6] += 2;
    }
    // Sisters of length 5
    if (move.combination == Move::Combination::kDoubleStraight5) {
        hands_[current_player_][move.rank - 3] -= 2;
        hands_[current_player_][move.rank - 4] -= 2;
        hands_[current_player_][move.rank - 5] -= 2;
        hands_[current_player_][move.rank - 6] -= 2;
        hands_[current_player_][move.rank - 7] -= 2;
        discard_pile_[move.rank - 3] += 2;
        discard_pile_[move.rank - 4] += 2;
        discard_pile_[move.rank - 5] += 2;
        discard_pile_[move.rank - 6] += 2;
        discard_pile_[move.rank - 7] += 2;
    }
    // Sisters of length 6
    if (move.combination == Move::Combination::kDoubleStraight6) {
        hands_[current_player_][move.rank - 3] -= 2;
        hands_[current_player_][move.rank - 4] -= 2;
        hands_[current_player_][move.rank - 5] -= 2;
        hands_[current_player_][move.rank - 6] -= 2;
        hands_[current_player_][move.rank - 7] -= 2;
        hands_[current_player_][move.rank - 8] -= 2;
        discard_pile_[move.rank - 3] += 2;
        discard_pile_[move.rank - 4] += 2;
        discard_pile_[move.rank - 5] += 2;
        discard_pile_[move.rank - 6] += 2;
        discard_pile_[move.rank - 7] += 2;
        discard_pile_[move.rank - 8] += 2;
    }
    // Sisters of length 7
    if (move.combination == Move::Combination::kDoubleStraight7) {
        hands_[current_player_][move.rank - 3] -= 2;
        hands_[current_player_][move.rank - 4] -= 2;
        hands_[current_player_][move.rank - 5] -= 2;
        hands_[current_player_][move.rank - 6] -= 2;
        hands_[current_player_][move.rank - 7] -= 2;
        hands_[current_player_][move.rank - 8] -= 2;
        hands_[current_player_][move.rank - 9] -= 2;
        discard_pile_[move.rank - 3] += 2;
        discard_pile_[move.rank - 4] += 2;
        discard_pile_[move.rank - 5] += 2;
        discard_pile_[move.rank - 6] += 2;
        discard_pile_[move.rank - 7] += 2;
        discard_pile_[move.rank - 8] += 2;
        discard_pile_[move.rank - 9] += 2;
    }
    // Sisters of length 8
    if (move.combination == Move::Combination::kDoubleStraight8) {
        hands_[current_player_][move.rank - 3] -= 2;
        hands_[current_player_][move.rank - 4] -= 2;
        hands_[current_player_][move.rank - 5] -= 2;
        hands_[current_player_][move.rank - 6] -= 2;
        hands_[current_player_][move.rank - 7] -= 2;
        hands_[current_player_][move.rank - 8] -= 2;
        hands_[current_player_][move.rank - 9] -= 2;
        hands_[current_player_][move.rank - 10] -= 2;
        discard_pile_[move.rank - 3] += 2;
        discard_pile_[move.rank - 4] += 2;
        discard_pile_[move.rank - 5] += 2;
        discard_pile_[move.rank - 6] += 2;
        discard_pile_[move.rank - 7] += 2;
        discard_pile_[move.rank - 8] += 2;
        discard_pile_[move.rank - 9] += 2;
        discard_pile_[move.rank - 10] += 2;
    }
    // Triple straights of length 2
    if (move.combination == Move::Combination::kTripleStraight2) {
        hands_[current_player_][move.rank - 3] -= 3;
        hands_[current_player_][move.rank - 4] -= 3;
        discard_pile_[move.rank - 3] += 3;
        discard_pile_[move.rank - 4] += 3;
    }
    // Triple straights of length 3
    if (move.combination == Move::Combination::kTripleStraight3) {
        hands_[current_player_][move.rank - 3] -= 3;
        hands_[current_player_][move.rank - 4] -= 3;
        hands_[current_player_][move.rank - 5] -= 3;
        discard_pile_[move.rank - 3] += 3;
        discard_pile_[move.rank - 4] += 3;
        discard_pile_[move.rank - 5] += 3;
    }

    // Triple straights of length 4
    if (move.combination == Move::Combination::kTripleStraight4) {
        hands_[current_player_][move.rank - 3] -= 3;
        hands_[current_player_][move.rank - 4] -= 3;
        hands_[current_player_][move.rank - 5] -= 3;
        hands_[current_player_][move.rank - 6] -= 3;
        discard_pile_[move.rank - 3] += 3;
        discard_pile_[move.rank - 4] += 3;
        discard_pile_[move.rank - 5] += 3;
        discard_pile_[move.rank - 6] += 3;
    }

    // Triple straights of length 5
    if (move.combination == Move::Combination::kTripleStraight5) {
        hands_[current_player_][move.rank - 3] -= 3;
        hands_[current_player_][move.rank - 4] -= 3;
        hands_[current_player_][move.rank - 5] -= 3;
        hands_[current_player_][move.rank - 6] -= 3;
        hands_[current_player_][move.rank - 7] -= 3;
        discard_pile_[move.rank - 3] += 3;
        discard_pile_[move.rank - 4] += 3;
        discard_pile_[move.rank - 5] += 3;
        discard_pile_[move.rank - 6] += 3;
        discard_pile_[move.rank - 7] += 3;
    }
    last_move_ = move;
    // Advance turn (simplest: alternate)
    current_player_ = 1 - current_player_;
}

int Game::get_winner() const {
    // Player with no cards wins
    if(std::all_of(hands_[0].begin(), hands_[0].end(), [](int c){ return c == 0; }))
        return 0;
    else
        return 1;
}

std::vector<int> Game::get_player_hand(int player) const {
    return std::vector<int>(hands_[player].begin(), hands_[player].end());
}
