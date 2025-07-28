#include "greedytrainer.h"
#include "util.h"

#include <iostream>
#include <fstream>
#include <vector>
#include "cnpy.h"

using namespace std;

int main(int argc, char *argv[]) {

    // Get index from command line
    int index = atoi(argv[1]);

    // Test the greedy trainer
    GreedyTrainer trainer(100000, index);
    trainer.playGames();
    int num_samples = trainer.getNumSamples();
    cout << "Number of samples: " << num_samples << endl;

    // Allocate buffers
    float *state_buffer = new float[num_samples * 53];
    int *turn_buffer = new int[num_samples];
    int *legal_mask_buffer = new int[num_samples * NEURAL_NETWORK_MOVES_SIZE];
    float *value_buffer = new float[num_samples];
    int *move_buffer = new int[num_samples];

    // Write the samples to the buffers
    trainer.writeSamplesComplex(state_buffer, turn_buffer, legal_mask_buffer, value_buffer, move_buffer);

    // Define shapes for each array
    vector<size_t> state_shape = {(size_t)num_samples, 53};
    vector<size_t> turn_shape = {(size_t)num_samples};
    vector<size_t> legal_mask_shape = {(size_t)num_samples, (size_t)NEURAL_NETWORK_MOVES_SIZE};
    vector<size_t> value_shape = {(size_t)num_samples};
    vector<size_t> move_shape = {(size_t)num_samples};

    // Save individual arrays as .npy files
    cnpy::npy_save("state_samples_big.npy", state_buffer, state_shape, "w");
    cnpy::npy_save("turn_samples_big.npy", turn_buffer, turn_shape, "w");
    cnpy::npy_save("legal_mask_samples_big.npy", legal_mask_buffer, legal_mask_shape, "w");
    cnpy::npy_save("value_samples_big.npy", value_buffer, value_shape, "w");
    cnpy::npy_save("move_samples_big.npy", move_buffer, move_shape, "w");

    // Free the buffers
    delete[] state_buffer;
    delete[] turn_buffer;
    delete[] legal_mask_buffer;
    delete[] value_buffer;
    delete[] move_buffer;

    return 0;
}