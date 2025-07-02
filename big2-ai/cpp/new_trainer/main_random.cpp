#include "game_coordinator.h"
#include "random_player_factory.h"
#include <memory>
#include <thread>
#include <random>
#include <iostream>

int main(int argc, char** argv) {
    // Configuration (could be extended to parse argc/argv)
    int num_games = 10000;
    std::string output_path = "game_records.jsonl";
    int num_threads = std::max(1u, std::thread::hardware_concurrency());
    unsigned int seed = std::random_device{}();

    std::cout << "Running " << num_games << " self-play games with "
              << num_threads << " threads, output to '" << output_path << "'...\n";

    // Create two RandomPlayer factories with distinct seeds
    auto factory0 = std::make_shared<RandomPlayerFactory>(seed);
    auto factory1 = std::make_shared<RandomPlayerFactory>(seed + 12345);

    // Initialize coordinator
    GameCoordinator coordinator(
        factory0,
        factory1,
        num_games,
        output_path,
        num_threads,
        seed
    );

    // Run self-play and export results
    coordinator.run_all();
    coordinator.export_data();

    std::cout << "Done.\n";
    return 0;
}
