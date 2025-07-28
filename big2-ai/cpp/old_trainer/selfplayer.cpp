#include "selfplayer.h"

#include "tree.h"

#include <vector>
#include <random>
#include <algorithm>
#include <utility>

using std::vector;

SelfPlayer::SelfPlayer(int seed, int max_searches, float c_puct) {

    int deck[48];
    for (int r = 3; r <= 13; ++r) {
        for (int i = 0; i < 4; ++i) {
            deck[r * 4 + i] = r;
        }
    }
    for (int i = 0; i < 3; ++i) {
        deck[i + 44] = 14;
    }
    deck[48] = 15;

    std::mt19937 rng(seed);
    bool player_turn = rng() % 2 == 0;
    std::shuffle(deck, deck + 48, rng);

    for (int i = 0; i < 2; ++i) {
        int player_cards[13] = {0};
        for(int j = 0; j < 16; ++j) {
            player_cards[deck[i * 16 + j] - 3]++;
        }
        players_[i] = Tree(player_turn, player_cards, max_searches, c_puct);
        player_turn = !player_turn;
    }
}

bool SelfPlayer::doIteration(double eval[], double probs[]) {
    if (eval == nullptr && probs == nullptr) {
        return false;
    }

    if (eval != nullptr) {
        eval[player_turn_] = players_[player_turn_].getEval();
    }

    if (probs != nullptr) {
        players_[player_turn_].getProbs(probs);
    }

    return true;
    
    
    
}
