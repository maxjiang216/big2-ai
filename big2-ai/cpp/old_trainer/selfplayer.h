#ifndef SELF_PLAYER_H
#define SELF_PLAYER_H

#include "tree.h"

class SelfPlayer {
    public:
        SelfPlayer() = default;
        ~SelfPlayer() = default;

        SelfPlayer(int seed, int max_searches, float c_puct);

        bool doIteration(double eval[] = nullptr, double probs[] = nullptr);

    private:

        int player_turn_{0};

        Tree players_[2];

};

#endif
