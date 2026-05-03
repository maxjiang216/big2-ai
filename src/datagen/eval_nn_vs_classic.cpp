// eval_nn_vs_classic: head-to-head evaluation, NN model vs classic player.
//
// Uses paired deals (same shuffle, seats swapped) and reports Wilson 95% CI.
// Single-threaded; use --deals 500-1000 for reliable estimates.
//
// Usage:
//   eval_nn_vs_classic --model PATH --vs PLAYER [--vs-param F]
//                      --deals N [--seed S] [--device auto|cpu|cuda]

// LibTorch must come first to avoid macro conflicts.
#include <torch/script.h>
#include <torch/cuda.h>

#include "nn_encode.h"

#include "game_simulator.h"
#include "hint_table.h"
#include "player_factory_registry.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <string>

// ============================================================================
// NNPlayer: wraps a TorchScript model using PartialGame for feature encoding
// ============================================================================

class NNPlayer : public Player {
public:
    NNPlayer(torch::jit::Module& model, torch::Device device)
        : model_(model), device_(device) {}

protected:
    Move select_move_impl() override {
        auto legal      = game_.get_legal_moves();
        const int K     = (int)legal.size();
        auto player_hand = game_.player_hand();
        auto discard    = game_.discard_pile();
        int  opp_count  = game_.opponent_hand_size();

        std::array<int,13> opp_counts{};
        HandBits opp_bits{};
        for (int r = 0; r < 13; ++r) {
            int om = std::max(0, max_cards_in_deck_for_rank(r) - player_hand[r] - discard[r]);
            opp_counts[r] = om;
            if (om >= 1) opp_bits.at1 |= static_cast<uint16_t>(1u << r);
            if (om >= 2) opp_bits.at2 |= static_cast<uint16_t>(1u << r);
            if (om >= 3) opp_bits.at3 |= static_cast<uint16_t>(1u << r);
            if (om >= 4) opp_bits.at4 |= static_cast<uint16_t>(1u << r);
        }

        auto float_opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto bool_opts  = torch::TensorOptions().dtype(torch::kBool);
        torch::Tensor t_hand = torch::zeros({K, ENCODING_DIM}, float_opts);
        torch::Tensor t_opp  = torch::zeros({K, ENCODING_DIM}, float_opts);
        torch::Tensor t_move = torch::zeros({K, ENCODING_DIM}, float_opts);
        torch::Tensor t_hint = torch::zeros({K}, float_opts);
        torch::Tensor t_flag = torch::zeros({K}, bool_opts);
        torch::Tensor t_pass = torch::zeros({K}, bool_opts);

        float* h_hand = t_hand.data_ptr<float>();
        float* h_opp  = t_opp.data_ptr<float>();
        float* h_move = t_move.data_ptr<float>();
        float* h_hint = t_hint.data_ptr<float>();
        bool*  h_flag = t_flag.data_ptr<bool>();
        bool*  h_pass = t_pass.data_ptr<bool>();

        for (int i = 0; i < K; ++i) {
            int mid = legal[i];
            bool is_pass_move = (mid == kPASS);
            std::array<int,13> mc{};
            if (!is_pass_move)
                for (int r = 0; r < 13; ++r) mc[r] = MOVE_TO_CARDS[mid][r];
            std::array<int,13> hand_after{};
            for (int r = 0; r < 13; ++r) hand_after[r] = player_hand[r] - mc[r];

            encode_exact(hand_after,  h_hand + i * ENCODING_DIM);
            encode_thermo(opp_counts, h_opp  + i * ENCODING_DIM);
            encode_exact(mc,          h_move + i * ENCODING_DIM);

            bool flag_val = !is_pass_move && !opponent_can_respond(mid, opp_bits, opp_count);
            h_hint[i] = is_pass_move ? 0.0f : (flag_val ? 1.0f : (1.0f - kHintTable[mid]));
            h_flag[i] = flag_val;
            h_pass[i] = is_pass_move;
        }

        torch::NoGradGuard no_grad;
        auto out = model_.forward({
            t_hand.to(device_), t_opp.to(device_), t_move.to(device_),
            t_hint.to(device_), t_flag.to(device_), t_pass.to(device_)
        }).toTuple();

        auto vi  = out->elements()[0].toTensor().to(torch::kCPU);
        auto vn  = out->elements()[1].toTensor().to(torch::kCPU);
        auto pt  = out->elements()[2].toTensor().to(torch::kCPU);
        auto* vi_p = vi.data_ptr<float>();
        auto* vn_p = vn.data_ptr<float>();
        auto* pt_p = pt.data_ptr<float>();

        int   best   = 0;
        float best_w = -1.0f;
        for (int i = 0; i < K; ++i) {
            float w = pt_p[i] * vi_p[i] + (1.0f - pt_p[i]) * vn_p[i];
            if (w > best_w) { best_w = w; best = i; }
        }
        return Move(legal[best]);
    }

private:
    torch::jit::Module& model_;
    torch::Device device_;
};

class NNPlayerFactory : public PlayerFactory {
public:
    NNPlayerFactory(torch::jit::Module& model, torch::Device device)
        : model_(model), device_(device) {}

    std::unique_ptr<Player> create_player() override {
        return std::make_unique<NNPlayer>(model_, device_);
    }

private:
    torch::jit::Module& model_;
    torch::Device device_;
};

// ============================================================================
// Wilson score 95% CI
// ============================================================================

struct WilsonCI { double lo, hi; };
static WilsonCI wilson_ci(double p_hat, long n, double z = 1.96) {
    double z2 = z*z, n_inv = 1.0/(double)n;
    double center = (p_hat + z2*n_inv/2.0) / (1.0 + z2*n_inv);
    double half = z * std::sqrt(p_hat*(1.0-p_hat)*n_inv + z2*n_inv*n_inv/4.0) /
                  (1.0 + z2*n_inv);
    return {center - half, center + half};
}

static std::string format_elapsed(long long ms) {
    if (ms < 1000) { char b[32]; std::snprintf(b, sizeof(b), "%lld ms", ms); return b; }
    double s = ms/1000.0; char b[64];
    if (s < 60) std::snprintf(b, sizeof(b), "%.1f s", s);
    else        std::snprintf(b, sizeof(b), "%dm %ds", (int)(s/60), (int)s%60);
    return b;
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " --model PATH --vs PLAYER [--vs-param F] --deals N [--seed S] [--device auto|cpu|cuda]\n"
              << "Options:\n"
              << "  --model PATH     NN model (TorchScript .pt); plays as model0\n"
              << "  --vs PLAYER      Classic opponent: random, greedy, pimc, etc.\n"
              << "  --vs-param F     Parameter for classic player (default: 0.0)\n"
              << "  --deals N        Paired deals; total games = 2*N\n"
              << "  --seed S         RNG seed (default: random)\n"
              << "  --device D       auto|cpu|cuda (default: auto)\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string model_path, vs_name, device_str = "auto";
    double vs_param = 0.0;
    int n_deals = 0;
    unsigned int seed = std::random_device{}();

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--help"     || a == "-h")  { print_usage(argv[0]); return 0; }
        else if (a == "--model"    && i+1 < argc)  model_path = argv[++i];
        else if (a == "--vs"       && i+1 < argc)  vs_name    = argv[++i];
        else if (a == "--vs-param" && i+1 < argc)  vs_param   = std::stod(argv[++i]);
        else if (a == "--deals"    && i+1 < argc)  n_deals    = std::stoi(argv[++i]);
        else if (a == "--seed"     && i+1 < argc)  seed       = (unsigned)std::stoul(argv[++i]);
        else if (a == "--device"   && i+1 < argc)  device_str = argv[++i];
        else { std::cerr << "Unknown arg: " << a << "\n"; print_usage(argv[0]); return 1; }
    }
    if (model_path.empty() || vs_name.empty() || n_deals <= 0) {
        std::cerr << "Error: --model, --vs, and --deals (>0) are required\n\n";
        print_usage(argv[0]); return 1;
    }

    torch::Device device = torch::kCPU;
    if      (device_str == "auto")  device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    else if (device_str == "cuda")  device = torch::kCUDA;
    else if (device_str != "cpu") { std::cerr << "Unknown device: " << device_str << "\n"; return 1; }
    std::cout << "Device: " << (device == torch::kCUDA ? "cuda" : "cpu") << "\n";

    std::cout << "Loading model: " << model_path << "\n";
    torch::jit::Module model = torch::jit::load(model_path);
    model.eval(); model.to(device);

    NNPlayerFactory nn_factory(model, device);
    auto classic_factory = make_player_factory(vs_name, vs_param, seed);
    if (!classic_factory) {
        std::cerr << "Unknown player: " << vs_name << "\n"; return 1;
    }

    std::string vs_label = vs_name;
    if (vs_param != 0.0) {
        char buf[64]; std::snprintf(buf, sizeof(buf), "%s(%.4g)", vs_name.c_str(), vs_param);
        vs_label = buf;
    }

    long total_games = 2L * n_deals;
    std::cout << "\n=== NN vs Classic: model0 vs " << vs_label << " ===\n"
              << "Deals: " << n_deals << "  |  Games: " << total_games << "\n"
              << "Seed: " << seed << "\n\n";

    long nn_wins = 0;
    long p0_sweep = 0, split_count = 0, p1_sweep = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int deal = 0; deal < n_deals; ++deal) {
        unsigned int unit_seed = seed + (unsigned int)deal;
        int deal_nn_wins = 0;

        // Game 1: NN=p0, classic=p1
        {
            std::mt19937 rng(unit_seed);
            auto p0 = nn_factory.create_player();
            auto p1 = classic_factory->create_player();
            GameSimulator sim(std::move(p0), std::move(p1), rng);
            if (sim.run().game().get_winner() == 0) ++deal_nn_wins;
        }
        // Game 2: classic=p0, NN=p1 (same seed = same deal)
        {
            std::mt19937 rng(unit_seed);
            auto p0 = classic_factory->create_player();
            auto p1 = nn_factory.create_player();
            GameSimulator sim(std::move(p0), std::move(p1), rng);
            if (sim.run().game().get_winner() == 1) ++deal_nn_wins;
        }

        nn_wins += deal_nn_wins;
        if      (deal_nn_wins == 2) ++p0_sweep;
        else if (deal_nn_wins == 0) ++p1_sweep;
        else                        ++split_count;

        if ((deal + 1) % 50 == 0 || deal + 1 == n_deals) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_start).count();
            double rate = elapsed > 0 ? 2.0 * (deal + 1) / elapsed : 0;
            std::fprintf(stderr, "\rDeals: %d/%d  |  %.0f games/s   ",
                         deal + 1, n_deals, rate);
            std::fflush(stderr);
        }
    }
    std::fprintf(stderr, "\n");

    auto t_end = std::chrono::high_resolution_clock::now();
    long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();

    double p_hat = (double)nn_wins / (double)total_games;
    WilsonCI ci  = wilson_ci(p_hat, total_games);

    std::cout << "NN wins: " << nn_wins << " / " << total_games << std::fixed;
    std::cout.precision(2);
    std::cout << " (" << 100.0 * p_hat << "%)\n"
              << "95% Wilson CI: [" << 100.0 * ci.lo << "%, " << 100.0 * ci.hi << "%]\n";

    if      (ci.lo > 0.50) std::cout << "Result: NN is significantly better (CI excludes 50%)\n";
    else if (ci.hi < 0.50) std::cout << "Result: " << vs_label << " is significantly better (CI excludes 50%)\n";
    else                   std::cout << "Result: no significant difference (CI includes 50%)\n";

    double inv = 1.0 / n_deals;
    std::cout << "Deals: nn_sweep=" << p0_sweep << " split=" << split_count
              << " classic_sweep=" << p1_sweep << "\n";
    std::cout.precision(2);
    std::cout << "Deals: nn 2-0: " << p0_sweep
              << " (" << 100.0 * p0_sweep * inv << "%)  split 1-1: " << split_count
              << " (" << 100.0 * split_count * inv << "%)  classic 2-0: " << p1_sweep
              << " (" << 100.0 * p1_sweep * inv << "%)\n";

    long n_decisive = p0_sweep + p1_sweep;
    if (n_decisive > 0) {
        double p_dec = (double)p0_sweep / (double)n_decisive;
        WilsonCI ci_dec = wilson_ci(p_dec, n_decisive);
        std::cout << "Among decisive: nn " << p0_sweep << " / " << n_decisive
                  << " (" << 100.0 * p_dec << "%)  CI: ["
                  << 100.0 * ci_dec.lo << "%, " << 100.0 * ci_dec.hi << "%]\n";
        if      (ci_dec.lo > 0.50) std::cout << "Result (decisive): NN sweeps more\n";
        else if (ci_dec.hi < 0.50) std::cout << "Result (decisive): " << vs_label << " sweeps more\n";
        else                        std::cout << "Result (decisive): no significant difference\n";
    }

    std::cout << "Elapsed: " << format_elapsed(elapsed_ms)
              << "  (" << (long long)(elapsed_ms > 0 ? 1000.0*total_games/elapsed_ms : 0)
              << " games/s)\n\n";
    return 0;
}
