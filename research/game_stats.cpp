// Empirical game-length and legal-move-count distributions.
//
// Runs simulated games using the full player infrastructure (GameSimulator +
// GameRecord) so any player type — including PIMC — can be used.  Parallelised
// with the same pattern as eval_match.
//
// Usage:
//   ./bin/game_stats --p0 <name> --p1 <name> [--p0-param N] [--p1-param N]
//                    [--games N] [--seed S] [--threads T]
//
// Example (random self-play):
//   ./bin/game_stats --p0 random --p1 random --games 200000
//
// Example (PIMC pass-rollout self-play, 100 determinizations):
//   ./bin/game_stats --p0 pimc_pass_rollout --p0-param 100
//                    --p1 pimc_pass_rollout --p1-param 100 --games 2000

#include "game_record.h"
#include "game_simulator.h"
#include "move.h"
#include "player_factory_registry.h"
#include "tablebase_opp1.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Per-thread stats accumulator
// ---------------------------------------------------------------------------

// Per-ply hand size samples (both players at the start of each ply).
struct PlyBucket {
    std::vector<int> p0, p1;  // hand sizes observed at this ply
};

struct Stats {
    std::map<int, long long> legal_all;
    std::map<int, long long> legal_lead;
    std::map<int, long long> legal_resp;
    std::map<int, long long> game_len;
    long long total_turns = 0;
    long long total_games = 0;

    std::vector<PlyBucket> ply;  // ply[i] = samples for ply i (0-indexed)

    void record_game(const GameRecord &rec) {
        int len = static_cast<int>(rec.turns().size());
        game_len[len]++;
        total_games++;

        if (static_cast<int>(ply.size()) < len)
            ply.resize(len);

        for (int i = 0; i < len; ++i) {
            const auto &t = rec.turns()[i];
            int n = static_cast<int>(t.legal_moves.size());
            bool is_lead = (t.game.last_move().combination == Move::Combination::kPass);
            legal_all[n]++;
            if (is_lead) legal_lead[n]++;
            else         legal_resp[n]++;
            total_turns++;

            ply[i].p0.push_back(t.game.get_player_hand_size(0));
            ply[i].p1.push_back(t.game.get_player_hand_size(1));
        }
    }

    void merge(const Stats &o) {
        for (auto &[k, v] : o.legal_all)  legal_all[k]  += v;
        for (auto &[k, v] : o.legal_lead) legal_lead[k] += v;
        for (auto &[k, v] : o.legal_resp) legal_resp[k] += v;
        for (auto &[k, v] : o.game_len)   game_len[k]   += v;
        total_turns += o.total_turns;
        total_games += o.total_games;

        if (ply.size() < o.ply.size()) ply.resize(o.ply.size());
        for (int i = 0; i < static_cast<int>(o.ply.size()); ++i) {
            auto &dst = ply[i];
            const auto &src = o.ply[i];
            dst.p0.insert(dst.p0.end(), src.p0.begin(), src.p0.end());
            dst.p1.insert(dst.p1.end(), src.p1.begin(), src.p1.end());
        }
    }
};

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

static void print_dist(const std::string &label,
                       const std::map<int, long long> &dist,
                       long long total) {
    if (total == 0) return;
    std::cout << "\n--- " << label << " (n=" << total << ") ---\n";
    std::cout << std::setw(8)  << "value"
              << std::setw(12) << "count"
              << std::setw(10) << "%"
              << std::setw(12) << "cumul%\n";

    long long cumul = 0;
    double mean = 0.0;
    long long p50 = -1, p90 = -1, p95 = -1, p99 = -1, pmax = -1;

    for (auto &[k, v] : dist) {
        mean += static_cast<double>(k) * v;
        pmax = k;
    }
    mean /= total;

    for (auto &[k, v] : dist) {
        cumul += v;
        double pct  = 100.0 * v      / total;
        double cpct = 100.0 * cumul  / total;
        std::cout << std::setw(8)  << k
                  << std::setw(12) << v
                  << std::setw(9)  << std::fixed << std::setprecision(2) << pct  << "%"
                  << std::setw(10) << std::fixed << std::setprecision(2) << cpct << "%\n";
        if (p50 < 0 && cumul * 2  >= total)          p50 = k;
        if (p90 < 0 && cumul * 10 >= 9  * total)     p90 = k;
        if (p95 < 0 && cumul * 20 >= 19 * total)     p95 = k;
        if (p99 < 0 && cumul * 100 >= 99 * total)    p99 = k;
    }
    std::cout << "  mean=" << std::fixed << std::setprecision(2) << mean
              << "  p50=" << p50 << "  p90=" << p90
              << "  p95=" << p95 << "  p99=" << p99
              << "  max=" << pmax << "\n";
}

static double median_of(std::vector<int> &v) {
    if (v.empty()) return 0.0;
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (v.size() % 2 == 1) return v[mid];
    // even: average of two middle elements
    int lo = *std::max_element(v.begin(), v.begin() + mid);
    return 0.5 * (lo + v[mid]);
}

static void print_ply_stats(const std::vector<PlyBucket> &ply) {
    std::cout << "\n--- HAND SIZES BY PLY (at start of each ply) ---\n";
    std::cout << std::setw(5)  << "ply"
              << std::setw(8)  << "n"
              << std::setw(12) << "mean_p0"
              << std::setw(10) << "med_p0"
              << std::setw(12) << "mean_p1"
              << std::setw(10) << "med_p1"
              << std::setw(14) << "mean_total"
              << std::setw(12) << "med_total\n";

    for (int i = 0; i < static_cast<int>(ply.size()); ++i) {
        auto p0v = ply[i].p0;  // copy for nth_element
        auto p1v = ply[i].p1;
        if (p0v.empty()) continue;

        long long n = static_cast<long long>(p0v.size());
        double mean_p0 = 0, mean_p1 = 0;
        for (int x : p0v) mean_p0 += x;
        for (int x : p1v) mean_p1 += x;
        mean_p0 /= n;
        mean_p1 /= n;

        double med_p0 = median_of(p0v);
        double med_p1 = median_of(p1v);

        std::cout << std::setw(5)  << i
                  << std::setw(8)  << n
                  << std::setw(12) << std::fixed << std::setprecision(2) << mean_p0
                  << std::setw(10) << std::fixed << std::setprecision(1) << med_p0
                  << std::setw(12) << std::fixed << std::setprecision(2) << mean_p1
                  << std::setw(10) << std::fixed << std::setprecision(1) << med_p1
                  << std::setw(14) << std::fixed << std::setprecision(2) << mean_p0 + mean_p1
                  << std::setw(12) << std::fixed << std::setprecision(1) << med_p0 + med_p1
                  << "\n";
    }
}

static void print_usage(const char *prog) {
    std::cout << "Usage: " << prog
              << " --p0 <name> --p1 <name>"
              << " [--p0-param F] [--p1-param F]"
              << " [--games N] [--seed S] [--threads T]\n"
              << "Available players: random, greedy, pimc, pimc_pass_rollout,"
              << " pimc_pass_rollout_adaptive, pimc_tree, pimc_adaptive, ...\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    std::string p0_name = "random", p1_name = "random";
    double p0_param = 0.0, p1_param = 0.0;
    int num_games = 100000;
    unsigned int seed = 42;
    int num_threads = std::max(1u, std::thread::hardware_concurrency() - 2);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--p0"       && i+1 < argc) p0_name   = argv[++i];
        else if (arg == "--p1"       && i+1 < argc) p1_name   = argv[++i];
        else if (arg == "--p0-param" && i+1 < argc) p0_param  = std::stod(argv[++i]);
        else if (arg == "--p1-param" && i+1 < argc) p1_param  = std::stod(argv[++i]);
        else if (arg == "--games"    && i+1 < argc) num_games = std::stoi(argv[++i]);
        else if (arg == "--seed"     && i+1 < argc) seed      = static_cast<unsigned>(std::stoul(argv[++i]));
        else if (arg == "--threads"  && i+1 < argc) num_threads = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        else { std::cerr << "Unknown arg: " << arg << "\n"; print_usage(argv[0]); return 1; }
    }

    load_tablebase_opp1("data/tablebase_opp1.bin");

    std::cout << "Players:  " << p0_name;
    if (p0_param != 0.0) std::cout << "(" << p0_param << ")";
    std::cout << "  vs  " << p1_name;
    if (p1_param != 0.0) std::cout << "(" << p1_param << ")";
    std::cout << "\nGames:    " << num_games
              << "\nThreads:  " << num_threads << "\n";

    std::atomic<int>  next_game{0};
    std::mutex        merge_mu;
    Stats             global;
    auto t_start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            unsigned int ts0 = seed + static_cast<unsigned>(t) * 1000003u;
            unsigned int ts1 = ts0  + 500009u;
            auto fa = make_player_factory(p0_name, p0_param, ts0);
            auto fb = make_player_factory(p1_name, p1_param, ts1);
            if (!fa || !fb) return;

            Stats local;
            int gi;
            while ((gi = next_game.fetch_add(1)) < num_games) {
                std::mt19937 rng(seed + static_cast<unsigned>(gi));
                auto p0 = fa->create_player();
                auto p1 = fb->create_player();
                GameSimulator sim(std::move(p0), std::move(p1), rng);
                GameRecord rec = sim.run();
                local.record_game(rec);
            }

            std::lock_guard<std::mutex> lk(merge_mu);
            global.merge(local);
        });
    }

    // Progress
    std::atomic<bool> done{false};
    std::thread prog([&]() {
        while (!done.load()) {
            int g = std::min(next_game.load(), num_games);
            auto now = std::chrono::steady_clock::now();
            double s = std::chrono::duration<double>(now - t_start).count();
            std::fprintf(stderr, "\r%d/%d games  (%.0f games/s)   ",
                         g, num_games, s > 0 ? g / s : 0.0);
            std::fflush(stderr);
            if (g >= num_games) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    });

    for (auto &w : workers) w.join();
    done = true;
    if (prog.joinable()) prog.join();

    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    std::fprintf(stderr, "\rDone. %.1f s  (%.0f games/s)\n",
                 elapsed, global.total_games / elapsed);

    std::cout << "\nTotal games: " << global.total_games
              << "  |  Total turns: " << global.total_turns
              << "  |  Avg turns/game: " << std::fixed << std::setprecision(1)
              << (double)global.total_turns / global.total_games << "\n";

    print_dist("GAME LENGTH (turns)", global.game_len, global.total_games);
    print_dist("LEGAL MOVES — all turns", global.legal_all, global.total_turns);

    long long lead_total = 0, resp_total = 0;
    for (auto &[k,v] : global.legal_lead) lead_total += v;
    for (auto &[k,v] : global.legal_resp) resp_total += v;

    print_dist("LEGAL MOVES — lead (new trick)", global.legal_lead, lead_total);
    print_dist("LEGAL MOVES — response (beat or pass)", global.legal_resp, resp_total);

    print_ply_stats(global.ply);

    return 0;
}
