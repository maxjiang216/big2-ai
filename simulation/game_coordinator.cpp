#include "game_coordinator.h"

#include "game_simulator.h"
#include "parquet_export.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <thread>

GameCoordinator::GameCoordinator(
    std::shared_ptr<PlayerFactory> player_factory_p0,
    std::shared_ptr<PlayerFactory> player_factory_p1, int num_games,
    const std::string &output_path, int num_threads, unsigned int random_seed,
    const std::string &log_path,
    std::vector<std::shared_ptr<FeatureExtractor>> game_level_features,
    std::vector<std::shared_ptr<FeatureExtractor>> turn_level_features)
    : _player_factory_p0(std::move(player_factory_p0)),
      _player_factory_p1(std::move(player_factory_p1)), _num_games(num_games),
      _output_path(output_path), _num_threads(std::max(1, num_threads)),
      _log_path(log_path), _rng_seed(random_seed),
      _game_level_features(std::move(game_level_features)),
      _turn_level_features(std::move(turn_level_features)) {}

void GameCoordinator::run_all(const std::string &game_feature_out,
                                const std::string &turn_feature_out) {

  constexpr int BATCH_SIZE = 200'000;
  const std::string game_tmp_prefix = "tmp_game_batch_";
  const std::string turn_tmp_prefix = "tmp_turn_batch_";

  int games_remaining = _num_games;
  int batch_idx = 0;

  std::vector<std::string> tmp_game_files;
  std::vector<std::string> tmp_turn_files;

  while (games_remaining > 0) {

    int this_batch = std::min(BATCH_SIZE, games_remaining);
    std::cout << "[Coordinator] Starting batch " << batch_idx << "  ("
              << this_batch << " games, " << games_remaining << " remaining)\n";

    std::atomic<int> next_index{0};
    std::vector<std::thread> workers;
    std::vector<std::vector<GameRecord>> local_batch(_num_threads);
    workers.reserve(_num_threads);

    for (int t = 0; t < _num_threads; ++t) {
      workers.emplace_back([&, t]() {
        std::mt19937 rng(_rng_seed + t + batch_idx * _num_threads);
        auto &out = local_batch[t];
        int idx;
        while ((idx = next_index.fetch_add(1)) < this_batch) {
          out.emplace_back(simulate_single_game(rng));
        }
      });
    }
    for (auto &th : workers)
      if (th.joinable())
        th.join();

    _records.clear();
    for (auto &vec : local_batch)
      _records.insert(_records.end(), vec.begin(), vec.end());

    std::string gfile =
        game_tmp_prefix + std::to_string(batch_idx) + ".parquet";
    std::string tfile =
        turn_tmp_prefix + std::to_string(batch_idx) + ".parquet";
    export_features(gfile, tfile);
    tmp_game_files.push_back(gfile);
    tmp_turn_files.push_back(tfile);

    _records.clear();
    local_batch.clear();
    workers.clear();

    games_remaining -= this_batch;
    ++batch_idx;
  }

  std::cout << "[Coordinator] Concatenating " << tmp_game_files.size()
            << " game batches...\n";
  concat_parquet_files(tmp_game_files, game_feature_out);

  std::cout << "[Coordinator] Concatenating " << tmp_turn_files.size()
            << " turn batches...\n";
  concat_parquet_files(tmp_turn_files, turn_feature_out);

  for (auto &f : tmp_game_files)
    std::filesystem::remove(f);
  for (auto &f : tmp_turn_files)
    std::filesystem::remove(f);

  std::cout
      << "[Coordinator] All batches complete — final Parquet files written.\n";
}

GameRecord GameCoordinator::simulate_single_game(std::mt19937 &rng) {
  auto player0 = _player_factory_p0->create_player();
  auto player1 = _player_factory_p1->create_player();
  GameSimulator sim(std::move(player0), std::move(player1), rng, _log_path);
  return sim.run();
}

void GameCoordinator::export_features(const std::string &game_feature_out,
                                    const std::string &turn_feature_out) const {
  export_parquet_features(_records, game_feature_out, turn_feature_out,
                          _game_level_features, _turn_level_features);
}
