#ifndef TRAINER_H
#define TRAINER_H

#include "selfplayer.h"

#include <vector>
#include <memory>

using std::vector;
using std::unique_ptr;

class Trainer {
    public:
        Trainer() = default;
        ~Trainer() = default;

        Trainer(int num_players, int seed, int max_searches, float c_puct);

        bool doIteration(double eval[] = nullptr, double probs[] = nullptr);
        
    private:

        vector<unique_ptr<SelfPlayer>> players_;

        
};

#endif
