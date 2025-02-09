#ifndef GAME_H
#define GAME_H

#include "move.h"
#include <bitset>
#include <vector>   

using std::bitset;
using std::vector;

class Game {
    // A game state for Big 2

    public:
        Game() = default;
        Game(const Game& other) = default;
        ~Game() = default;

        // Initialize the deal
        Game(bool player_turn, int player_cards[13]);

        void doMove(Move move);

        vector<int> getLegalMoves() const;
        

    private:
        // True if it's the player's turn
        bool player_turn;
        // The player's cards
        int player_cards[13];
        // The opponent's cards
        int opponent_card_num{16};
        // Cards played so far
        int table_cards[13];
        // The last move
        Move last_move{Move::Combination::kPass};
        
};


#endif
