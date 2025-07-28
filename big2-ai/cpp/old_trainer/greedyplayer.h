#ifndef GREEDYPLAYER_H
#define GREEDYPLAYER_H

#include "game.h"
#include "move.h"
#include "util.h"

#include <vector>

using std::vector;

// Greedy player
// Fast way to generate reasonable moves
// Used as supervised training and to test neural network
class GreedyPlayer {

    public:
        GreedyPlayer() = delete;
        GreedyPlayer(int deal[32], int player_turn, int id);
        void playGame();
        int getNumSamples() const;
        int writeSamples(float *stateBuffer, int *turnBuffer, int *legalMaskBuffer, float *valueBuffer, int *moveBuffer) const;
        int writeSamplesComplex(float *stateBuffer, int *turnBuffer, int *legalMaskBuffer, float *valueBuffer, int *moveBuffer) const;
    private:
        int id_{0};
        int player_turn_{0};
        Game games_[2];
        vector<Sample> samples_;

        Move chooseMove(bitset<LEGAL_MOVES_SIZE> &legal_moves_bitset) const;

};

#endif