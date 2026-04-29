// Simulate random-play games and record the empirical distribution of
// legal move counts at each turn.
//
// Separates lead positions (last_move == pass, player has the trick)
// from response positions (must beat the last move or pass).
// Also breaks down by remaining hand size.
//
// Usage: ./bin/legal_move_dist [num_games]

#include "game.h"
#include "move.h"
#include "util.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

static void print_dist(const std::string &label,
                       const std::map<int, long long> &dist,
                       long long total) {
    if (total == 0) return;
    std::cout << "\n--- " << label << " (n=" << total << ") ---\n";
    std::cout << std::setw(8) << "legal"
              << std::setw(12) << "count"
              << std::setw(10) << "%"
              << std::setw(12) << "cumul%\n";

    long long cumul = 0;
    double mean = 0.0;
    long long p50_val = -1, p90_val = -1, p95_val = -1, p99_val = -1;

    for (auto &[k, v] : dist)
        mean += static_cast<double>(k) * v;
    mean /= total;

    for (auto &[k, v] : dist) {
        cumul += v;
        double pct  = 100.0 * v / total;
        double cpct = 100.0 * cumul / total;
        std::cout << std::setw(8) << k
                  << std::setw(12) << v
                  << std::setw(9) << std::fixed << std::setprecision(2) << pct << "%"
                  << std::setw(10) << std::fixed << std::setprecision(2) << cpct << "%\n";
        if (p50_val < 0 && cumul * 2 >= total) p50_val = k;
        if (p90_val < 0 && cumul * 10 >= 9 * total) p90_val = k;
        if (p95_val < 0 && cumul * 20 >= 19 * total) p95_val = k;
        if (p99_val < 0 && cumul * 100 >= 99 * total) p99_val = k;
    }
    std::cout << "  mean=" << std::fixed << std::setprecision(2) << mean
              << "  p50=" << p50_val
              << "  p90=" << p90_val
              << "  p95=" << p95_val
              << "  p99=" << p99_val << "\n";
}

int main(int argc, char *argv[]) {
    int num_games = 200000;
    if (argc > 1) num_games = std::stoi(argv[1]);

    std::mt19937 rng(42);

    std::map<int, long long> lead_dist;   // last_move == pass
    std::map<int, long long> resp_dist;   // last_move != pass
    std::map<int, long long> all_dist;

    // breakdown by hand size (how many cards the current player holds)
    // hand_size -> {lead_count, resp_count}
    std::map<int, std::array<long long, 2>> by_hand;

    long long total = 0;
    long long games_played = 0;

    for (int g = 0; g < num_games; ++g) {
        Game game;
        game.shuffle_deal(rng);
        ++games_played;

        while (!game.is_over()) {
            int cp = game.current_player();
            auto legal = game.get_legal_moves();
            int n = static_cast<int>(legal.size());
            int hand_sz = game.get_player_hand_size(cp);
            bool is_lead = (game.last_move().combination == Move::Combination::kPass);

            all_dist[n]++;
            if (is_lead) lead_dist[n]++;
            else         resp_dist[n]++;
            by_hand[hand_sz][is_lead ? 0 : 1]++;
            ++total;

            std::uniform_int_distribution<int> pick(0, n - 1);
            game.apply_move(legal[pick(rng)]);
        }
    }

    std::cout << "Games simulated: " << games_played << "\n";
    std::cout << "Total turns:     " << total << "\n";
    std::cout << "Avg turns/game:  " << std::fixed << std::setprecision(1)
              << static_cast<double>(total) / games_played << "\n";

    print_dist("ALL turns", all_dist, total);
    print_dist("LEAD turns (new trick)", lead_dist,
               [&] { long long s=0; for (auto&[k,v]:lead_dist) s+=v; return s; }());
    print_dist("RESPONSE turns (must beat or pass)", resp_dist,
               [&] { long long s=0; for (auto&[k,v]:resp_dist) s+=v; return s; }());

    // Per-hand-size summary (mean legal moves, lead vs resp)
    std::cout << "\n--- By hand size ---\n";
    std::cout << std::setw(6) << "cards"
              << std::setw(14) << "lead_turns"
              << std::setw(14) << "resp_turns"
              << "\n";
    for (auto &[sz, counts] : by_hand) {
        std::cout << std::setw(6) << sz
                  << std::setw(14) << counts[0]
                  << std::setw(14) << counts[1]
                  << "\n";
    }

    return 0;
}
