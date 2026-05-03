#pragma once
// Shared NN feature encoding utilities used by generate_nn_selfplay and eval_nn_match.

#include <algorithm>
#include <array>

static constexpr int ENCODING_DIM = 48;
static constexpr int RANK_MAX[13] = {4,4,4,4,4,4,4,4,4,4,4,3,1};

// Exact one-hot encoding: bit i of rank r is set iff count[r] == i+1.
inline void encode_exact(const std::array<int,13>& counts, float* buf) {
    int off = 0;
    for (int r = 0; r < 13; ++r) {
        int c = counts[r], mx = RANK_MAX[r];
        for (int k = 1; k <= mx; ++k)
            buf[off++] = (c == k) ? 1.0f : 0.0f;
    }
}

// Thermometer (upper-bound) encoding: bit i of rank r is set iff count[r] >= i+1.
inline void encode_thermo(const std::array<int,13>& counts, float* buf) {
    int off = 0;
    for (int r = 0; r < 13; ++r) {
        int c = std::min(counts[r], RANK_MAX[r]), mx = RANK_MAX[r];
        for (int k = 1; k <= mx; ++k)
            buf[off++] = (c >= k) ? 1.0f : 0.0f;
    }
}
