#pragma once
// Shared utilities for pool-based batched NN evaluation.

#include "game.h"
#include "hint_table.h"
#include "move.h"
#include "nn_encode.h"
#include "tablebase_opp1.h"
#include "util.h"

#include <torch/script.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

// ============================================================================
// Wilson CI
// ============================================================================

struct WilsonCI { double lo, hi; };
inline WilsonCI wilson_ci(double p_hat, long n, double z = 1.96) {
    double z2 = z*z, n_inv = 1.0/(double)n;
    double center = (p_hat + z2*n_inv/2.0) / (1.0 + z2*n_inv);
    double half = z * std::sqrt(p_hat*(1.0-p_hat)*n_inv + z2*n_inv*n_inv/4.0) /
                  (1.0 + z2*n_inv);
    return {center - half, center + half};
}

inline std::string format_elapsed(long long ms) {
    if (ms < 1000) { char b[32]; std::snprintf(b, sizeof(b), "%lld ms", ms); return b; }
    double s = ms / 1000.0; char b[64];
    if (s < 60) std::snprintf(b, sizeof(b), "%.1f s", s);
    else        std::snprintf(b, sizeof(b), "%dm %ds", (int)(s/60), (int)s%60);
    return b;
}

// ============================================================================
// Game-level tablebase (mirrors nn_game_runner.cpp)
// ============================================================================

inline int tablebase_move_id(const Game& game, int cp, const std::vector<int>& legal) {
    auto hand      = game.player_hand(cp);
    int  hand_size = game.get_player_hand_size(cp);
    int  opp_size  = game.get_player_hand_size(1 - cp);

    for (int mid : legal) {
        if (mid == kPASS) continue;
        if (MOVE_TO_CARDS[mid][13] == hand_size) return mid;
    }
    if (game.last_move().combination != Move::Combination::kPass) return -1;

    auto discard = game.discard_pile();
    if (opp_size == 1) {
        int best_rank = -1; bool all_singles = true;
        for (int mid : legal) {
            if (mid == kPASS) continue;
            Move m(mid);
            if (m.combination != Move::Combination::kSingle) { all_singles = false; break; }
            if (m.rank > best_rank) best_rank = m.rank;
        }
        if (all_singles && best_rank != -1) {
            for (int mid : legal)
                if (mid != kPASS && Move(mid).rank == best_rank) return mid;
        }
        Opp1Result opp1 = lookup_opp1(hand);
        if (opp1.first_move_id != 0) {
            for (int mid : legal) if (mid == opp1.first_move_id) return mid;
        }
        if (auto def = opp1_default_strategy_move(hand)) return *def;
    }
    auto seq = find_forced_win(hand, discard, opp_size);
    if (seq) return (*seq)[0];
    return -1;
}

// ============================================================================
// Batched NN inference
//
// positions[i] = {game_ptr, cp, legal_moves_ptr}
// Returns best legal-move index (into legal_moves) for each position.
// ============================================================================

using NNPosition = std::tuple<const Game*, int, const std::vector<int>*>;

inline std::vector<int> batch_nn_select(
    torch::jit::Module& model,
    torch::Device device,
    const std::vector<NNPosition>& positions)
{
    if (positions.empty()) return {};

    int total_k = 0;
    std::vector<int> offsets(positions.size()), sizes(positions.size());
    for (int i = 0; i < (int)positions.size(); ++i) {
        offsets[i] = total_k;
        sizes[i]   = (int)std::get<2>(positions[i])->size();
        total_k   += sizes[i];
    }

    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto bool_opts  = torch::TensorOptions().dtype(torch::kBool);
    auto t_hand = torch::zeros({total_k, ENCODING_DIM}, float_opts);
    auto t_opp  = torch::zeros({total_k, ENCODING_DIM}, float_opts);
    auto t_move = torch::zeros({total_k, ENCODING_DIM}, float_opts);
    auto t_hint = torch::zeros({total_k}, float_opts);
    auto t_flag = torch::zeros({total_k}, bool_opts);
    auto t_pass = torch::zeros({total_k}, bool_opts);

    float* h_hand = t_hand.data_ptr<float>();
    float* h_opp  = t_opp.data_ptr<float>();
    float* h_move = t_move.data_ptr<float>();
    float* h_hint = t_hint.data_ptr<float>();
    bool*  h_flag = t_flag.data_ptr<bool>();
    bool*  h_pass = t_pass.data_ptr<bool>();

    for (int i = 0; i < (int)positions.size(); ++i) {
        const Game&              game  = *std::get<0>(positions[i]);
        int                      cp    = std::get<1>(positions[i]);
        const std::vector<int>&  legal = *std::get<2>(positions[i]);
        int K = sizes[i], off = offsets[i];

        auto hand_counts = game.player_hand(cp);
        auto discard     = game.discard_pile();
        int  opp_count   = game.get_player_hand_size(1 - cp);

        std::array<int,13> opp_counts{};
        HandBits opp_bits{};
        for (int r = 0; r < 13; ++r) {
            int om = std::max(0, max_cards_in_deck_for_rank(r) - hand_counts[r] - discard[r]);
            opp_counts[r] = om;
            if (om >= 1) opp_bits.at1 |= static_cast<uint16_t>(1u << r);
            if (om >= 2) opp_bits.at2 |= static_cast<uint16_t>(1u << r);
            if (om >= 3) opp_bits.at3 |= static_cast<uint16_t>(1u << r);
            if (om >= 4) opp_bits.at4 |= static_cast<uint16_t>(1u << r);
        }

        for (int j = 0; j < K; ++j) {
            int mid = legal[j];
            bool is_pass = (mid == kPASS);
            std::array<int,13> mc{}, hand_after{};
            if (!is_pass)
                for (int r = 0; r < 13; ++r) mc[r] = MOVE_TO_CARDS[mid][r];
            for (int r = 0; r < 13; ++r) hand_after[r] = hand_counts[r] - mc[r];

            encode_exact(hand_after,  h_hand + (off+j) * ENCODING_DIM);
            encode_thermo(opp_counts, h_opp  + (off+j) * ENCODING_DIM);
            encode_exact(mc,          h_move + (off+j) * ENCODING_DIM);

            bool flag = !is_pass && !opponent_can_respond(mid, opp_bits, opp_count);
            h_hint[off+j] = is_pass ? 0.0f : (flag ? 1.0f : (1.0f - kHintTable[mid]));
            h_flag[off+j] = flag;
            h_pass[off+j] = is_pass;
        }
    }

    torch::NoGradGuard no_grad;
    auto out = model.forward({
        t_hand.to(device), t_opp.to(device), t_move.to(device),
        t_hint.to(device), t_flag.to(device), t_pass.to(device)
    }).toTuple();

    auto vi = out->elements()[0].toTensor().to(torch::kCPU);
    auto vn = out->elements()[1].toTensor().to(torch::kCPU);
    auto pt = out->elements()[2].toTensor().to(torch::kCPU);
    const float* vi_p = vi.data_ptr<float>();
    const float* vn_p = vn.data_ptr<float>();
    const float* pt_p = pt.data_ptr<float>();

    std::vector<int> result(positions.size());
    for (int i = 0; i < (int)positions.size(); ++i) {
        int off = offsets[i], K = sizes[i];
        int best = 0; float best_w = -1.0f;
        for (int j = 0; j < K; ++j) {
            float w = pt_p[off+j] * vi_p[off+j] + (1.0f - pt_p[off+j]) * vn_p[off+j];
            if (w > best_w) { best_w = w; best = j; }
        }
        result[i] = best;
    }
    return result;
}
