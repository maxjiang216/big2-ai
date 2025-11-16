#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/result.h> //  (brings in ARROW_ASSIGN_OR_RAISE)
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h> //  ←  ADD THIS
#include <parquet/arrow/writer.h>

#include "feature_extractor.h"
#include "game_coordinator.h"
#include "game_record.h"
#include "game_simulator.h"
#include "player_factory.h"

#include <atomic>
#include <filesystem>
#include <fstream>
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

  constexpr int BATCH_SIZE = 200'000; // <= 200 k games per chunk
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

    // ------------------------------------------------------------------
    // simulate
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

    // flatten into _records (re‑use member to leverage existing exporters)
    _records.clear();
    for (auto &vec : local_batch)
      _records.insert(_records.end(), vec.begin(), vec.end());

    // ------------------------------------------------------------------ export
    // batch
    std::string gfile =
        game_tmp_prefix + std::to_string(batch_idx) + ".parquet";
    std::string tfile =
        turn_tmp_prefix + std::to_string(batch_idx) + ".parquet";
    export_features(gfile, tfile);
    tmp_game_files.push_back(gfile);
    tmp_turn_files.push_back(tfile);

    // free memory before next batch
    _records.clear();
    local_batch.clear();
    workers.clear();

    games_remaining -= this_batch;
    ++batch_idx;
  }

  // ------------------------------------------------------------------
  // concatenate helper
  auto concat = [](const std::vector<std::string> &parts,
                   const std::string &out_file) {
    std::shared_ptr<arrow::Table> concat_table;

    for (size_t i = 0; i < parts.size(); ++i) {

      // Open Parquet file and read full table
      ARROW_ASSIGN_OR_RAISE(auto rf, arrow::io::ReadableFile::Open(parts[i]));

      std::unique_ptr<parquet::arrow::FileReader> reader;
      PARQUET_THROW_NOT_OK(
          parquet::arrow::OpenFile(rf, arrow::default_memory_pool(), &reader));

      std::shared_ptr<arrow::Table> piece;
      PARQUET_THROW_NOT_OK(reader->ReadTable(&piece));

      // Append to running table
      if (i == 0) {
        concat_table = piece;
      } else {
        ARROW_ASSIGN_OR_RAISE(
            concat_table,
            arrow::ConcatenateTables({concat_table, piece},
                                     arrow::ConcatenateTablesOptions{false}));
      }
    }

    // Write final table
    ARROW_ASSIGN_OR_RAISE(auto out,
                          arrow::io::FileOutputStream::Open(out_file));

    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(
        *concat_table, arrow::default_memory_pool(), out, 4096));
  };

  std::cout << "[Coordinator] Concatenating " << tmp_game_files.size()
            << " game batches...\n";
  concat(tmp_game_files, game_feature_out);

  std::cout << "[Coordinator] Concatenating " << tmp_turn_files.size()
            << " turn batches...\n";
  concat(tmp_turn_files, turn_feature_out);

  // ------------------------------------------------------------------ clean up
  // temp files
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

// --------------------------------------------------------
// Feature Extraction: Arrow table output
// --------------------------------------------------------
void GameCoordinator::export_features(
    const std::string &game_feature_out,
    const std::string &turn_feature_out) const {
  export_game_features(game_feature_out);
  export_turn_features(turn_feature_out);
}

void GameCoordinator::export_game_features(const std::string &out_file) const {
  if (_game_level_features.empty()) {
    std::cout << "No game-level features to export.\n";
    return;
  }

  if (_records.empty()) {
    std::cout << "No game records to export.\n";
    return;
  }

  std::cout << "Exporting " << _records.size() << " game records with "
            << _game_level_features.size() << " features...\n";

  try {
    // Each feature is a column (vector<int>)
    std::vector<std::shared_ptr<arrow::Array>> columns;
    std::vector<std::shared_ptr<arrow::Field>> schema_fields;

    for (const auto &extractor : _game_level_features) {
      if (!extractor) {
        std::cerr << "Warning: Null feature extractor found, skipping.\n";
        continue;
      }

      arrow::Int32Builder builder;
      for (const auto &record : _records) {
        int feature_value = extractor->gameExtract(record);
        auto status = builder.Append(feature_value);
        if (!status.ok()) {
          std::cerr << "Error appending to builder: " << status.ToString()
                    << std::endl;
          return;
        }
      }

      std::shared_ptr<arrow::Array> arr;
      auto status = builder.Finish(&arr);
      if (!status.ok()) {
        std::cerr << "Error finishing builder: " << status.ToString()
                  << std::endl;
        return;
      }

      columns.push_back(arr);
      schema_fields.push_back(arrow::field(extractor->name(), arrow::int32()));
    }

    if (columns.empty()) {
      std::cout << "No valid features to export.\n";
      return;
    }

    auto schema = std::make_shared<arrow::Schema>(schema_fields);
    auto table = arrow::Table::Make(schema, columns, _records.size());

    // Write to Parquet file
    auto file_result = arrow::io::FileOutputStream::Open(out_file);
    if (!file_result.ok()) {
      std::cerr << "Error opening output file: "
                << file_result.status().ToString() << std::endl;
      return;
    }
    auto outfile = file_result.ValueOrDie();

    auto write_status = parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), outfile, 4096);
    if (!write_status.ok()) {
      std::cerr << "Error writing Parquet file: " << write_status.ToString()
                << std::endl;
      return;
    }

    std::cout << "Successfully exported game features to: " << out_file
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Exception in export_game_features: " << e.what() << std::endl;
  }
}

void GameCoordinator::export_turn_features(const std::string &out_file) const {
  if (_turn_level_features.empty()) {
    std::cout << "No turn-level features to export.\n";
    return;
  }

  if (_records.empty()) {
    std::cout << "No game records to export.\n";
    return;
  }

  std::cout << "Exporting turn features for " << _records.size()
            << " games with " << _turn_level_features.size()
            << " features...\n";

  try {
    // Flatten all turns in all games for all features
    // All columns will have same number of entries: total_turns
    std::vector<std::vector<int>> all_columns(_turn_level_features.size());
    size_t total_turns = 0;

    for (size_t f = 0; f < _turn_level_features.size(); ++f) {
      if (!_turn_level_features[f]) {
        std::cerr << "Warning: Null turn feature extractor at index " << f
                  << ", skipping.\n";
        continue;
      }

      for (const auto &record : _records) {
        auto feature_vals = _turn_level_features[f]->turnExtract(record);
        all_columns[f].insert(all_columns[f].end(), feature_vals.begin(),
                              feature_vals.end());
      }
    }

    if (!all_columns.empty()) {
      total_turns = all_columns[0].size();
    }

    if (total_turns == 0) {
      std::cout << "No turns found to export.\n";
      return;
    }

    // Verify all columns have the same size
    for (size_t f = 0; f < all_columns.size(); ++f) {
      if (all_columns[f].size() != total_turns) {
        std::cerr << "Error: Feature column " << f << " has size "
                  << all_columns[f].size() << " but expected " << total_turns
                  << std::endl;
        return;
      }
    }

    std::cout << "Processing " << total_turns << " total turns...\n";

    // Build Arrow columns
    std::vector<std::shared_ptr<arrow::Array>> columns;
    std::vector<std::shared_ptr<arrow::Field>> schema_fields;

    for (size_t f = 0; f < _turn_level_features.size(); ++f) {
      if (!_turn_level_features[f])
        continue; // Skip null extractors

      arrow::Int32Builder builder;
      for (auto v : all_columns[f]) {
        auto status = builder.Append(v);
        if (!status.ok()) {
          std::cerr << "Error appending to turn builder: " << status.ToString()
                    << std::endl;
          return;
        }
      }

      std::shared_ptr<arrow::Array> arr;
      auto status = builder.Finish(&arr);
      if (!status.ok()) {
        std::cerr << "Error finishing turn builder: " << status.ToString()
                  << std::endl;
        return;
      }

      columns.push_back(arr);
      schema_fields.push_back(
          arrow::field(_turn_level_features[f]->name(), arrow::int32()));
    }

    if (columns.empty()) {
      std::cout << "No valid turn features to export.\n";
      return;
    }

    auto schema = std::make_shared<arrow::Schema>(schema_fields);
    auto table = arrow::Table::Make(schema, columns, total_turns);

    auto file_result = arrow::io::FileOutputStream::Open(out_file);
    if (!file_result.ok()) {
      std::cerr << "Error opening turn output file: "
                << file_result.status().ToString() << std::endl;
      return;
    }
    auto outfile = file_result.ValueOrDie();

    auto write_status = parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), outfile, 4096);
    if (!write_status.ok()) {
      std::cerr << "Error writing turn Parquet file: "
                << write_status.ToString() << std::endl;
      return;
    }

    std::cout << "Successfully exported turn features to: " << out_file
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Exception in export_turn_features: " << e.what() << std::endl;
  }
}