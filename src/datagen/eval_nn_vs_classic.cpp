// eval_nn_vs_classic: head-to-head evaluation, NN model vs classic player.
//
// Pool-based batched NN inference: all 2*n_deals games run concurrently.
// Classic (random/greedy) moves are applied inline per slot; NN decisions
// are batched across all active slots into a single forward pass per round.
//
// For PIMC: a per-slot Player object keeps the classic player's PartialGame
// in sync. PIMC moves are applied sequentially (CPU-bound) between NN batches.
//
// Usage:
//   eval_nn_vs_classic --model PATH --vs PLAYER [--vs-param F]
//                      --deals N [--seed S] [--device auto|cpu|cuda]

// LibTorch must come first to avoid macro conflicts.
#include <torch/cuda.h>
#include <torch/script.h>

#include "eval_helpers.h"

#include "greedy/greedy_player.h"
#include "player.h"
#include "player_factory_registry.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Classic policy types
// ============================================================================

enum class ClassicType { Random, Greedy, Other };

static ClassicType classify(const std::string& name) {
    if (name == "random") return ClassicType::Random;
    if (name == "greedy") return ClassicType::Greedy;
    return ClassicType::Other;
}

// ============================================================================
// Inline classic move selection (no Player object needed)
// ============================================================================

static int select_random(const std::vector<int>& legal, std::mt19937& rng) {
    int idx = std::uniform_int_distribution<int>(0, (int)legal.size()-1)(rng);
    return legal[idx];
}

// Greedy: maximise GreedyEval of post-move hand; pass only if no non-pass move.
static int select_greedy(const Game& game, int cp, const std::vector<int>& legal) {
    auto hand = game.player_hand(cp);

    int best_mid = -1;
    GreedyEval best_eval{};

    for (int mid : legal) {
        if (mid == kPASS) continue;
        std::array<int,13> hand_after{};
        for (int r = 0; r < 13; ++r)
            hand_after[r] = hand[r] - MOVE_TO_CARDS[mid][r];

        int n_cards = 0, n_bombs = 0;
        for (int i = 0; i < 13; ++i) {
            n_cards += hand_after[i];
            if ((i < 11 && hand_after[i] == 4) || (i == 11 && hand_after[i] == 3)) ++n_bombs;
        }
        GreedyEval ev;
        ev.win_now       = (n_cards == 0) ? 1 : 0;
        ev.bombs         = n_bombs;
        ev.neg_num_cards = -n_cards;
        ev.num_2s        = hand_after[12]; ev.num_as  = hand_after[11]; ev.num_ks  = hand_after[10];
        ev.num_qs        = hand_after[9];  ev.num_js  = hand_after[8];  ev.num_10s = hand_after[7];
        ev.num_9s        = hand_after[6];  ev.num_8s  = hand_after[5];  ev.num_7s  = hand_after[4];
        ev.num_6s        = hand_after[3];  ev.num_5s  = hand_after[2];  ev.num_4s  = hand_after[1];

        if (best_mid == -1 || best_eval < ev) { best_eval = ev; best_mid = mid; }
    }
    return (best_mid != -1) ? best_mid : kPASS;
}

// ============================================================================
// Eval slot
// ============================================================================

struct EvalSlot {
    Game game;
    int  deal_id;
    int  nn_player;          // seat (0 or 1) occupied by NN
    std::vector<int> legal;  // cached when NN decision is needed
    std::unique_ptr<Player> classic;  // non-null for Other (e.g. PIMC)
    std::mt19937 slot_rng;            // for random classic moves
    bool done{false};
    int  nn_wins{0};
};

// Apply an NN move to the game and keep classic Player (PIMC) in sync.
static void apply_and_sync_nn_move(EvalSlot& sl, int move_id) {
    sl.game.apply_move(move_id);
    if (sl.classic) sl.classic->accept_opponent_move(Move(move_id));
}

// Advance slot until game over or NN decision is needed.
// Classic moves (random/greedy/Player) are consumed inline.
static void advance_to_nn_decision(EvalSlot& sl, ClassicType ctype) {
    while (!sl.game.is_over()) {
        int cp = sl.game.current_player();
        sl.legal = sl.game.get_legal_moves();

        // ── NN side ──────────────────────────────────────────────────────────
        if (cp == sl.nn_player) {
            int tb = tablebase_move_id(sl.game, cp, sl.legal);
            if (tb >= 0) { apply_and_sync_nn_move(sl, tb); continue; }
            if ((int)sl.legal.size() == 1) { apply_and_sync_nn_move(sl, sl.legal[0]); continue; }
            return;  // NN decision needed
        }

        // ── Classic side ─────────────────────────────────────────────────────
        if (sl.classic) {
            // Player-managed (PIMC): delegate all move selection to select_move(),
            // which handles tablebase/forced/policy; applies to classic's game_ internally.
            // Apply the same move to the authoritative Game state.
            Move m = sl.classic->select_move();
            sl.game.apply_move(encodeMove(m));
        } else {
            // Inline (random/greedy): handle tablebase/forced here for consistency.
            int tb = tablebase_move_id(sl.game, cp, sl.legal);
            if (tb >= 0) { sl.game.apply_move(tb); continue; }
            if ((int)sl.legal.size() == 1) { sl.game.apply_move(sl.legal[0]); continue; }
            int mid = (ctype == ClassicType::Random) ?
                      select_random(sl.legal, sl.slot_rng) :
                      select_greedy(sl.game, cp, sl.legal);
            sl.game.apply_move(mid);
        }
    }
}

// ============================================================================
// Reporting
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " --model PATH --vs PLAYER [--vs-param F] --deals N [--seed S] [--device D]\n"
              << "Options:\n"
              << "  --model PATH     NN model (TorchScript .pt)\n"
              << "  --vs PLAYER      Classic opponent: random, greedy, pimc, ...\n"
              << "  --vs-param F     Parameter for classic player (default: 0)\n"
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
        if      (a == "--help"     || a == "-h") { print_usage(argv[0]); return 0; }
        else if (a == "--model"    && i+1 < argc) model_path = argv[++i];
        else if (a == "--vs"       && i+1 < argc) vs_name    = argv[++i];
        else if (a == "--vs-param" && i+1 < argc) vs_param   = std::stod(argv[++i]);
        else if (a == "--deals"    && i+1 < argc) n_deals    = std::stoi(argv[++i]);
        else if (a == "--seed"     && i+1 < argc) seed       = (unsigned)std::stoul(argv[++i]);
        else if (a == "--device"   && i+1 < argc) device_str = argv[++i];
        else { std::cerr << "Unknown arg: " << a << "\n"; print_usage(argv[0]); return 1; }
    }
    if (model_path.empty() || vs_name.empty() || n_deals <= 0) {
        std::cerr << "Error: --model, --vs, and --deals (>0) are required\n\n";
        print_usage(argv[0]); return 1;
    }

    ClassicType ctype = classify(vs_name);

    // For non-inline classic players we need a factory to create per-slot players.
    std::shared_ptr<PlayerFactory> classic_factory;
    if (ctype == ClassicType::Other) {
        classic_factory = make_player_factory(vs_name, vs_param, seed);
        if (!classic_factory) {
            std::cerr << "Unknown player: " << vs_name << "\n"; return 1;
        }
    }

    torch::Device device = torch::kCPU;
    if      (device_str == "auto") device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    else if (device_str == "cuda") device = torch::kCUDA;
    else if (device_str != "cpu") { std::cerr << "Unknown device: " << device_str << "\n"; return 1; }
    std::cout << "Device: " << (device == torch::kCUDA ? "cuda" : "cpu") << "\n";

    std::cout << "Loading model: " << model_path << "\n";
    torch::jit::Module model = torch::jit::load(model_path);
    model.eval(); model.to(device);

    std::string vs_label = vs_name;
    if (vs_param != 0.0) {
        char buf[64]; std::snprintf(buf, sizeof(buf), "%s(%.4g)", vs_name.c_str(), vs_param);
        vs_label = buf;
    }

    long total_games = 2L * n_deals;
    std::cout << "\n=== NN vs Classic: NN vs " << vs_label << " ===\n"
              << "Deals: " << n_deals << "  |  Games: " << total_games << "\n"
              << "Seed: " << seed << "\n\n";

    // ── Initialise pool ───────────────────────────────────────────────────────
    std::vector<EvalSlot> pool(2 * n_deals);
    std::vector<int> deal_nn_wins(n_deals, 0);

    for (int d = 0; d < n_deals; ++d) {
        for (int seat = 0; seat < 2; ++seat) {
            EvalSlot& sl = pool[2*d + seat];
            std::mt19937 rng(seed + (unsigned)d);
            sl.game.shuffle_deal(rng);
            sl.deal_id   = d;
            sl.nn_player = seat;  // seat 0: NN is player 0; seat 1: NN is player 1
            sl.slot_rng  = std::mt19937(seed + (unsigned)(d * 1000 + seat));

            if (classic_factory) {
                sl.classic = classic_factory->create_player();
                sl.classic->accept_deal(sl.game, 1 - seat);  // classic is the other seat
            }
        }
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    int n_done = 0;

    // ── Pool-based eval loop ──────────────────────────────────────────────────
    while (n_done < (int)pool.size()) {

        // 1. Advance all active slots to NN decision points (classic moves inline).
        std::vector<int> nn_idx;

        for (int si = 0; si < (int)pool.size(); ++si) {
            EvalSlot& sl = pool[si];
            if (sl.done) continue;

            advance_to_nn_decision(sl, ctype);

            if (sl.game.is_over()) {
                sl.done = true;
                ++n_done;
                if (sl.game.get_winner() == sl.nn_player) ++deal_nn_wins[sl.deal_id];
            } else {
                nn_idx.push_back(si);
            }
        }

        if (n_done == (int)pool.size()) break;

        // 2. Batch forward pass for NN positions.
        if (!nn_idx.empty()) {
            std::vector<NNPosition> pos;
            pos.reserve(nn_idx.size());
            for (int si : nn_idx)
                pos.emplace_back(&pool[si].game, pool[si].game.current_player(), &pool[si].legal);
            auto best = batch_nn_select(model, device, pos);
            for (int i = 0; i < (int)nn_idx.size(); ++i) {
                EvalSlot& sl = pool[nn_idx[i]];
                int move_id = sl.legal[best[i]];
                apply_and_sync_nn_move(sl, move_id);
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();

    // ── Aggregate results ─────────────────────────────────────────────────────
    long nn_wins = 0;
    long p0_sweep = 0, split_count = 0, p1_sweep = 0;
    for (int d = 0; d < n_deals; ++d) {
        int w = deal_nn_wins[d];
        nn_wins += w;
        if      (w == 2) ++p0_sweep;
        else if (w == 0) ++p1_sweep;
        else             ++split_count;
    }

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
    std::cout << "Deals: WW=" << p0_sweep
              << " (" << 100.0 * p0_sweep * inv << "%)  split=" << split_count
              << " (" << 100.0 * split_count * inv << "%)  LL=" << p1_sweep
              << " (" << 100.0 * p1_sweep * inv << "%)\n";

    long n_decisive = p0_sweep + p1_sweep;
    if (n_decisive > 0) {
        double p_dec = (double)p0_sweep / (double)n_decisive;
        WilsonCI ci_dec = wilson_ci(p_dec, n_decisive);
        std::cout << "Among decisive: NN " << p0_sweep << " / " << n_decisive
                  << " (" << 100.0 * p_dec << "%)  CI: ["
                  << 100.0 * ci_dec.lo << "%, " << 100.0 * ci_dec.hi << "%]\n";
    }

    std::cout << "Elapsed: " << format_elapsed(elapsed_ms)
              << "  (" << (long long)(elapsed_ms > 0 ? 1000.0*total_games/elapsed_ms : 0)
              << " games/s)\n\n";
    return 0;
}
