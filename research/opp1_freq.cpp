// How often does the opp-1-card endgame actually arise?
//
// The web computer's opp-1 strategy fires only when: the side to move HAS THE
// LEAD (new trick) and the opponent has exactly 1 card. This tool plays full
// games and measures, per game:
//   any1card  : some turn has a player at exactly 1 card (loose upper bound)
//   trigger   : some turn is (lead AND opponent-to-move has exactly 1 card)
//               for EITHER seat  -> the opp-1 strategy is reachable this game
//   seat0     : same trigger but only for player 0 (a fixed "computer" seat)
//
// Frequency depends on policy (how endgames are played), so run a couple.
//
// Usage:
//   ./bin/opp1_freq --p0 greedy --p1 greedy --games 200000 [--seed S] [--threads T]

#include "game_record.h"
#include "game_simulator.h"
#include "move.h"
#include "player_factory_registry.h"
#include "tablebase_opp1.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct Stats {
    long long games = 0;
    long long any1card = 0;   // a player reaches exactly 1 card at some point
    long long trigger = 0;    // (lead && opp-to-move has 1 card), either seat
    long long seat0 = 0;      // ... but only counting player 0 in that seat

    void record_game(const GameRecord &rec) {
        ++games;
        bool a1 = false, trig = false, s0 = false;
        for (const auto &t : rec.turns()) {
            const Game &g = t.game;
            int h0 = g.get_player_hand_size(0), h1 = g.get_player_hand_size(1);
            if (h0 == 1 || h1 == 1) a1 = true;
            bool is_lead = (g.last_move().combination == Move::Combination::kPass);
            if (!is_lead) continue;
            int cur = g.current_player();
            int opp = 1 - cur;
            if (g.get_player_hand_size(opp) == 1) {
                trig = true;
                if (cur == 0) s0 = true;
            }
        }
        any1card += a1;
        trigger += trig;
        seat0 += s0;
    }

    void merge(const Stats &o) {
        games += o.games; any1card += o.any1card;
        trigger += o.trigger; seat0 += o.seat0;
    }
};

static void pct(const char *label, long long n, long long total) {
    double p = total ? 100.0 * n / total : 0.0;
    std::cout << std::setw(34) << std::left << label << std::right
              << std::setw(12) << n << "   "
              << std::fixed << std::setprecision(2) << std::setw(6) << p << "%\n";
}

int main(int argc, char **argv) {
    std::string p0_name = "greedy", p1_name = "greedy";
    double p0_param = 0.0, p1_param = 0.0;
    int num_games = 200000;
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
        else { std::cerr << "Unknown arg: " << arg << "\n"; return 1; }
    }

    load_tablebase_opp1("data/tablebase_opp1.bin");
    std::cout << "Players: " << p0_name << " vs " << p1_name
              << "  | games: " << num_games << "  | threads: " << num_threads << "\n";

    std::atomic<int> next_game{0};
    std::mutex merge_mu;
    Stats global;
    auto t_start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            unsigned int ts0 = seed + static_cast<unsigned>(t) * 1000003u;
            unsigned int ts1 = ts0 + 500009u;
            auto fa = make_player_factory(p0_name, p0_param, ts0);
            auto fb = make_player_factory(p1_name, p1_param, ts1);
            if (!fa || !fb) return;
            Stats local;
            int gi;
            while ((gi = next_game.fetch_add(1)) < num_games) {
                std::mt19937 rng(seed + static_cast<unsigned>(gi));
                GameSimulator sim(fa->create_player(), fb->create_player(), rng);
                local.record_game(sim.run());
            }
            std::lock_guard<std::mutex> lk(merge_mu);
            global.merge(local);
        });
    }
    for (auto &w : workers) w.join();

    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    std::fprintf(stderr, "Done. %.1f s (%.0f games/s)\n",
                 elapsed, global.games / elapsed);

    std::cout << "\n" << std::setw(34) << std::left << "event" << std::right
              << std::setw(12) << "games" << "   " << "  rate\n";
    pct("player ever at exactly 1 card", global.any1card, global.games);
    pct("opp-1 strategy reachable (either seat)", global.trigger, global.games);
    pct("... for a fixed seat (per-player)", global.seat0, global.games);
    return 0;
}
