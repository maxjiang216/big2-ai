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
    // TODO: implement full move application logic (remove cards, update trick, handle pass)
    // Example for single:
    if(move.combination == Move::Combination::kSingle) {
        hands_[current_player_][move.rank] -= 1;
    }
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
