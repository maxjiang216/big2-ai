#include "game_coordinator.h"
#include "player_factory.h"
#include "game_simulator.h"
#include "game_record.h"

#include <fstream>
#include <iostream>
#include <thread>
#include <atomic>

GameCoordinator::GameCoordinator(
    std::shared_ptr<PlayerFactory> player_factory_p0,
    std::shared_ptr<PlayerFactory> player_factory_p1,
    int num_games,
    const std::string& output_path,
    int num_threads,
    unsigned int random_seed
) :
    _player_factory_p0(std::move(player_factory_p0)),
    _player_factory_p1(std::move(player_factory_p1)),
    _num_games(num_games),
    _output_path(output_path),
    _num_threads(std::max(1, num_threads)),
    _rng_seed(random_seed)
{}

void GameCoordinator::run_all() {
    _records.clear();
    _records.reserve(_num_games);

    std::atomic<int> next_index{0};
    std::vector<std::thread> workers;
    std::vector<std::vector<GameRecord>> local_records(_num_threads);
    workers.reserve(_num_threads);

    // Launch threads
    for(int t = 0; t < _num_threads; ++t) {
        workers.emplace_back([this, &next_index, t, &local_records]() mutable {
            // Thread-local RNG
            std::mt19937 rng(_rng_seed + t);
            auto& out = local_records[t];
            int idx;
            while((idx = next_index.fetch_add(1)) < _num_games) {
                out.emplace_back(simulate_single_game(rng));
            }
        });
    }

    // Join threads
    for(auto& th : workers) {
        if(th.joinable()) th.join();
    }

    // Merge results
    for(auto& vec : local_records) {
        _records.insert(
            _records.end(),
            std::make_move_iterator(vec.begin()),
            std::make_move_iterator(vec.end())
        );
    }
}

GameRecord GameCoordinator::simulate_single_game(std::mt19937 &rng) {
    auto player0 = _player_factory_p0->create_player();
    auto player1 = _player_factory_p1->create_player();
    GameSimulator sim(std::move(player0), std::move(player1), rng);
    return sim.run();
}

void GameCoordinator::export_data() const {
    std::ofstream ofs(_output_path);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << _output_path << "\n";
        return;
    }

    for(const auto& rec : _records) {
        ofs << rec.to_json() << '\n';
    }
    std::cout << "Exported " << _records.size()
              << " records to " << _output_path << "\n";
}
