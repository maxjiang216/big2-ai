#include "greedytrainer.h"

#include <random>
#include <algorithm>
#include <iostream>

using namespace std;

GreedyTrainer::GreedyTrainer(int num_games, int seed) {
    players_.reserve(num_games);

    // Initialize the players
    // We can reshuffle the same deck for all games
    int deck[48];
    for (int r = 3; r <= 13; ++r) {
        for (int i = 0; i < 4; ++i) {
            deck[(r - 3) * 4 + i] = r;
        }
    }
    for (int i = 0; i < 3; ++i) {
        deck[i + 44] = 14;
    }
    deck[47] = 15;

    std::mt19937 rng(seed);
    for (int i = 0; i < num_games; ++i) {
        // Generate a random deal
        std::shuffle(deck, deck + 48, rng);
        players_.push_back(GreedyPlayer(deck, rng() % 2, i));
    }

}

void GreedyTrainer::playGames() {
    for (int i = 0; i < players_.size(); ++i) {
        players_[i].playGame();
    }
}

int GreedyTrainer::getNumSamples() const {
    int num_samples = 0;
    for (int i = 0; i < players_.size(); ++i) {
        num_samples += players_[i].getNumSamples();
    }
    return num_samples;
}

void GreedyTrainer::writeSamples(float *state_buffer, int *turn_buffer, int *legal_mask_buffer, float *value_buffer, int *move_buffer) const {
    int sample_index = 0;
    for (int i = 0; i < players_.size(); ++i) {
        players_[i].writeSamples(state_buffer + sample_index * 26, turn_buffer + sample_index, legal_mask_buffer + sample_index * NEURAL_NETWORK_MOVES_SIZE, value_buffer + sample_index, move_buffer + sample_index);
        sample_index += players_[i].getNumSamples();
    }
}

void GreedyTrainer::writeSamplesComplex(float *state_buffer, int *turn_buffer, int *legal_mask_buffer, float *value_buffer, int *move_buffer) const {
    int sample_index = 0;
    for (int i = 0; i < players_.size(); ++i) {
        players_[i].writeSamplesComplex(state_buffer + sample_index * 53, turn_buffer + sample_index, legal_mask_buffer + sample_index * NEURAL_NETWORK_MOVES_SIZE, value_buffer + sample_index, move_buffer + sample_index);
        sample_index += players_[i].getNumSamples();
    }
}