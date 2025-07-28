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
        Game(bool player_turn, int player_cards[16]);

        void doMove(Move move);

        vector<int> getLegalMoves() const;

        int getPlayerTurn() const;
        int getPlayerCardRank(int i) const;
        int getTableCardRank(int i) const;
        int getOpponentCardNum() const;
        Move getLastMove() const;
        

    private:
        // True if it's the player's turn
        bool player_turn;
        // The player's cards, how many cards of each rank
        int player_cards[13];
        // The opponent's cards, the number of cards, since we don't need to know the exact cards
        int opponent_card_num{16};
        // Cards played so far, how many cards of each rank
        int table_cards[13];
        // The last move
        Move last_move{Move::Combination::kPass};

        // Print the game state
        friend std::ostream& operator<<(std::ostream& os, const Game& game);
};

#endif
