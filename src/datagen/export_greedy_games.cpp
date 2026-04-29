// Outputs N greedy-vs-greedy games dealt by std::mt19937(seed).
// Line format: hand0[0..12]  hand1[0..12]  num_moves  move_id...
// Used to verify Rust greedy logic produces identical moves from the same hands.
#include "game.h"
#include "game_record.h"
#include "game_simulator.h"
#include "greedy/greedy_player.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>

int main(int argc, char **argv) {
    int n_games = 100;
    uint64_t seed = 42;
    if (argc > 1) n_games = std::atoi(argv[1]);
    if (argc > 2) seed = std::stoull(argv[2]);

    std::mt19937 rng(seed);

    for (int g = 0; g < n_games; ++g) {
        auto p0 = std::make_unique<GreedyPlayer>();
        auto p1 = std::make_unique<GreedyPlayer>();
        GameSimulator sim(std::move(p0), std::move(p1), rng);
        GameRecord record = sim.run();

        // turns()[0].game is the pre-move initial state after the deal
        const Game &initial = record.turns()[0].game;
        auto h0 = initial.player_hand(0);
        auto h1 = initial.player_hand(1);

        for (int r = 0; r < 13; ++r)
            std::cout << h0[r] << (r < 12 ? " " : "");
        std::cout << "  ";
        for (int r = 0; r < 13; ++r)
            std::cout << h1[r] << (r < 12 ? " " : "");

        const auto &turns = record.turns();
        std::cout << "  " << turns.size();
        for (const auto &t : turns)
            std::cout << " " << encodeMove(t.move);
        std::cout << "\n";
    }
    return 0;
}
