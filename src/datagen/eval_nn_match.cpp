// eval_nn_match: head-to-head evaluation between two TorchScript NN models.
//
// Pool-based batched NN inference: all 2*n_deals games run concurrently.
// Each batch round advances every game past tablebase/forced moves, then
// does one batched forward pass for model0 positions and one for model1
// positions. GPU utilization is ~2000x better than serial per-position inference.
//
// Usage:
//   eval_nn_match --model0 PATH --model1 PATH --deals N [--seed S] [--device D]

#include "eval_helpers.h"

#include <torch/cuda.h>
#include <torch/script.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Eval slot: one active game
// ============================================================================

struct EvalSlot {
    Game game;
    int  deal_id;
    int  m0_player;          // seat (0 or 1) occupied by model0
    std::vector<int> legal;  // cached legal moves when NN decision is needed
    bool done{false};
    int  winner{-1};
};

// Advance slot past tablebase/forced moves until game over or NN decision needed.
// Does NOT call any NN; just handles deterministic cases.
static void advance_to_decision(EvalSlot& slot) {
    while (!slot.game.is_over()) {
        int cp = slot.game.current_player();
        slot.legal = slot.game.get_legal_moves();
        int tb = tablebase_move_id(slot.game, cp, slot.legal);
        if (tb >= 0) { slot.game.apply_move(tb); continue; }
        if ((int)slot.legal.size() == 1) { slot.game.apply_move(slot.legal[0]); continue; }
        return;  // NN decision needed
    }
}

// ============================================================================
// Reporting
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " --model0 PATH --model1 PATH --deals N [--seed S] [--device auto|cpu|cuda]\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string model0_path, model1_path, device_str = "auto";
    int n_deals = 0;
    unsigned int seed = std::random_device{}();

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--help"   || a == "-h") { print_usage(argv[0]); return 0; }
        else if (a == "--model0" && i+1 < argc) model0_path = argv[++i];
        else if (a == "--model1" && i+1 < argc) model1_path = argv[++i];
        else if (a == "--deals"  && i+1 < argc) n_deals     = std::stoi(argv[++i]);
        else if (a == "--seed"   && i+1 < argc) seed        = (unsigned)std::stoul(argv[++i]);
        else if (a == "--device" && i+1 < argc) device_str  = argv[++i];
        else { std::cerr << "Unknown arg: " << a << "\n"; print_usage(argv[0]); return 1; }
    }
    if (model0_path.empty() || model1_path.empty() || n_deals <= 0) {
        std::cerr << "Error: --model0, --model1, and --deals (>0) are required\n\n";
        print_usage(argv[0]); return 1;
    }

    torch::Device device = torch::kCPU;
    if      (device_str == "auto") device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    else if (device_str == "cuda") device = torch::kCUDA;
    else if (device_str != "cpu") { std::cerr << "Unknown device: " << device_str << "\n"; return 1; }
    std::cout << "Device: " << (device == torch::kCUDA ? "cuda" : "cpu") << "\n";

    std::cout << "Loading models...\n";
    torch::jit::Module model0 = torch::jit::load(model0_path);
    torch::jit::Module model1 = torch::jit::load(model1_path);
    model0.eval(); model0.to(device);
    model1.eval(); model1.to(device);
    std::cout << "  model0: " << model0_path << "\n"
              << "  model1: " << model1_path << "\n\n";

    long total_games = 2L * n_deals;
    std::cout << "=== NN Eval Match: model0 vs model1 ===\n"
              << "Deals: " << n_deals << "  |  Games: " << total_games << "\n"
              << "Seed: " << seed << "\n\n";

    // ── Initialise pool (2 games per deal, seats swapped) ────────────────────
    std::vector<EvalSlot> pool(2 * n_deals);
    std::vector<int> deal_m0_wins(n_deals, 0);

    for (int d = 0; d < n_deals; ++d) {
        for (int seat = 0; seat < 2; ++seat) {
            EvalSlot& sl = pool[2*d + seat];
            std::mt19937 rng(seed + (unsigned)d);
            sl.game.shuffle_deal(rng);
            sl.deal_id   = d;
            sl.m0_player = seat;  // seat 0: m0 plays player 0; seat 1: m0 plays player 1
        }
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    int n_done = 0;

    // ── Pool-based eval loop ──────────────────────────────────────────────────
    while (n_done < (int)pool.size()) {

        // 1. Advance all active slots to decision points.
        std::vector<int> m0_idx, m1_idx;  // slot indices where m0 / m1 must decide

        for (int si = 0; si < (int)pool.size(); ++si) {
            EvalSlot& sl = pool[si];
            if (sl.done) continue;

            advance_to_decision(sl);

            if (sl.game.is_over()) {
                sl.done   = true;
                sl.winner = sl.game.get_winner();
                ++n_done;
                if (sl.winner == sl.m0_player) ++deal_m0_wins[sl.deal_id];
            } else {
                int cp = sl.game.current_player();
                if (cp == sl.m0_player) m0_idx.push_back(si);
                else                    m1_idx.push_back(si);
            }
        }

        if (n_done == (int)pool.size()) break;

        // 2. Batch forward pass for model0.
        if (!m0_idx.empty()) {
            std::vector<NNPosition> pos;
            pos.reserve(m0_idx.size());
            for (int si : m0_idx)
                pos.emplace_back(&pool[si].game, pool[si].game.current_player(), &pool[si].legal);
            auto best = batch_nn_select(model0, device, pos);
            for (int i = 0; i < (int)m0_idx.size(); ++i)
                pool[m0_idx[i]].game.apply_move(pool[m0_idx[i]].legal[best[i]]);
        }

        // 3. Batch forward pass for model1.
        if (!m1_idx.empty()) {
            std::vector<NNPosition> pos;
            pos.reserve(m1_idx.size());
            for (int si : m1_idx)
                pos.emplace_back(&pool[si].game, pool[si].game.current_player(), &pool[si].legal);
            auto best = batch_nn_select(model1, device, pos);
            for (int i = 0; i < (int)m1_idx.size(); ++i)
                pool[m1_idx[i]].game.apply_move(pool[m1_idx[i]].legal[best[i]]);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();

    // ── Aggregate results ─────────────────────────────────────────────────────
    long m0_wins = 0;
    long p0_sweep = 0, split_count = 0, p1_sweep = 0;
    for (int d = 0; d < n_deals; ++d) {
        int w = deal_m0_wins[d];
        m0_wins += w;
        if      (w == 2) ++p0_sweep;
        else if (w == 0) ++p1_sweep;
        else             ++split_count;
    }

    double p_hat = (double)m0_wins / (double)total_games;
    WilsonCI ci  = wilson_ci(p_hat, total_games);

    std::cout << "Model0 wins: " << m0_wins << " / " << total_games << std::fixed;
    std::cout.precision(2);
    std::cout << " (" << 100.0 * p_hat << "%)\n"
              << "95% Wilson CI: [" << 100.0 * ci.lo << "%, " << 100.0 * ci.hi << "%]\n";

    if      (ci.lo > 0.50) std::cout << "Result: model0 is significantly better (CI excludes 50%)\n";
    else if (ci.hi < 0.50) std::cout << "Result: model1 is significantly better (CI excludes 50%)\n";
    else                   std::cout << "Result: no significant difference (CI includes 50%)\n";

    double inv = 1.0 / n_deals;
    std::cout << "Deals: WW=" << p0_sweep
              << " (" << 100.0 * p0_sweep * inv << "%)  split=" << split_count
              << " (" << 100.0 * split_count * inv << "%)  LL=" << p1_sweep
              << " (" << 100.0 * p1_sweep * inv << "%)\n";

    long n_decisive = p0_sweep + p1_sweep;
    if (n_decisive > 0) {
        double p_dec = (double)p0_sweep / (double)n_decisive;
        WilsonCI ci_dec = wilson_ci(p_dec, n_decisive);
        std::cout << "Among decisive: model0 " << p0_sweep << " / " << n_decisive
                  << " (" << 100.0 * p_dec << "%)  CI: ["
                  << 100.0 * ci_dec.lo << "%, " << 100.0 * ci_dec.hi << "%]\n";
    }

    std::cout << "Elapsed: " << format_elapsed(elapsed_ms)
              << "  (" << (long long)(elapsed_ms > 0 ? 1000.0*total_games/elapsed_ms : 0)
              << " games/s)\n\n";
    return 0;
}
