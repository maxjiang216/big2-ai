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
        Game(int seed = 0);
        ~Game() = default;

        void do_move(int move);

        vector<int> get_legal_moves() const;
        

    private:
        // True if it's the player's turn
        bool player_turn;
        // The player's cards
        int player_cards[13];
        // The opponent's cards
        int opponent_cards[13];
        // Cards played so far
        int played_cards[13];
        // The last move
        Move last_move;
        
};


#endif
