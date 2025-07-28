#include "trainer.h"

#include <random>

Trainer::Trainer(int num_players, int seed, int max_searches, float c_puct) {

    std::mt19937 rng(seed);
    for (int i = 0; i < num_players; ++i) {
        players_.push_back(unique_ptr<SelfPlayer>(new SelfPlayer(rng(), max_searches, c_puct)));
    }

}

bool Trainer::doIteration(double eval[] = nullptr, double probs[] = nullptr) {
    for (size_t i = 0; i < players_.size(); ++i) {
        players_[i]->doIteration(eval, probs);
    }
}