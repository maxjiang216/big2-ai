#ifndef GREEDYTRAINER_H
#define GREEDYTRAINER_H

#include "greedyplayer.h"

#include <vector>

using std::vector;

class GreedyTrainer {

    public:

        GreedyTrainer(int num_games, int seed = 0);
        
        void writeSamples(float *state_buffer, int *turn_buffer, int *legal_mask_buffer, float *value_buffer, int *move_buffer) const;
        void writeSamplesComplex(float *state_buffer, int *turn_buffer, int *legal_mask_buffer, float *value_buffer, int *move_buffer) const;
        void playGames();
        int getNumSamples() const;

    private:

        vector<GreedyPlayer> players_;
};

#endif