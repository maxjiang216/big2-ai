#include "parquet_export.hpp"

#include "game_record.h"
#include "feature_extractor.h"

#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <iostream>

void export_parquet_features(
    const std::vector<GameRecord> &records,
    const std::string &game_feature_out,
    const std::string &turn_feature_out,
    const std::vector<std::shared_ptr<FeatureExtractor>> &game_level_features,
    const std::vector<std::shared_ptr<FeatureExtractor>> &turn_level_features) {

  auto export_game = [&]() {
    if (game_level_features.empty()) {
      std::cout << "No game-level features to export.\n";
      return;
    }
    if (records.empty()) {
      std::cout << "No game records to export.\n";
      return;
    }

    std::cout << "Exporting " << records.size() << " game records with "
              << game_level_features.size() << " features...\n";

    try {
      std::vector<std::shared_ptr<arrow::Array>> columns;
      std::vector<std::shared_ptr<arrow::Field>> schema_fields;

      for (const auto &extractor : game_level_features) {
        if (!extractor) {
          std::cerr << "Warning: Null feature extractor found, skipping.\n";
          continue;
        }

        arrow::Int32Builder builder;
        for (const auto &record : records) {
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
      auto table = arrow::Table::Make(schema, columns, records.size());

      auto file_result = arrow::io::FileOutputStream::Open(game_feature_out);
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

      std::cout << "Successfully exported game features to: " << game_feature_out
                << std::endl;

    } catch (const std::exception &e) {
      std::cerr << "Exception in export_game_features: " << e.what() << std::endl;
    }
  };

  auto export_turn = [&]() {
    if (turn_level_features.empty()) {
      std::cout << "No turn-level features to export.\n";
      return;
    }
    if (records.empty()) {
      std::cout << "No game records to export.\n";
      return;
    }

    std::cout << "Exporting turn features for " << records.size()
              << " games with " << turn_level_features.size()
              << " features...\n";

    try {
      std::vector<std::vector<int>> all_columns(turn_level_features.size());
      size_t total_turns = 0;

      for (size_t f = 0; f < turn_level_features.size(); ++f) {
        if (!turn_level_features[f]) {
          std::cerr << "Warning: Null turn feature extractor at index " << f
                    << ", skipping.\n";
          continue;
        }

        for (const auto &record : records) {
          auto feature_vals = turn_level_features[f]->turnExtract(record);
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

      for (size_t f = 0; f < all_columns.size(); ++f) {
        if (all_columns[f].size() != total_turns) {
          std::cerr << "Error: Feature column " << f << " has size "
                    << all_columns[f].size() << " but expected " << total_turns
                    << std::endl;
          return;
        }
      }

      std::cout << "Processing " << total_turns << " total turns...\n";

      std::vector<std::shared_ptr<arrow::Array>> columns;
      std::vector<std::shared_ptr<arrow::Field>> schema_fields;

      for (size_t f = 0; f < turn_level_features.size(); ++f) {
        if (!turn_level_features[f])
          continue;

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
            arrow::field(turn_level_features[f]->name(), arrow::int32()));
      }

      if (columns.empty()) {
        std::cout << "No valid turn features to export.\n";
        return;
      }

      auto schema = std::make_shared<arrow::Schema>(schema_fields);
      auto table = arrow::Table::Make(schema, columns, total_turns);

      auto file_result = arrow::io::FileOutputStream::Open(turn_feature_out);
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

      std::cout << "Successfully exported turn features to: " << turn_feature_out
                << std::endl;

    } catch (const std::exception &e) {
      std::cerr << "Exception in export_turn_features: " << e.what() << std::endl;
    }
  };

  export_game();
  export_turn();
}

void concat_parquet_files(const std::vector<std::string> &parts,
                          const std::string &out_file) {
  if (parts.empty())
    return;

  std::shared_ptr<arrow::Table> concat_table;

  for (size_t i = 0; i < parts.size(); ++i) {
    auto rf_result = arrow::io::ReadableFile::Open(parts[i]);
    if (!rf_result.ok()) {
      std::cerr << "concat_parquet_files: open failed: "
                << rf_result.status().ToString() << "\n";
      return;
    }
    auto rf = rf_result.ValueOrDie();

    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(
        parquet::arrow::OpenFile(rf, arrow::default_memory_pool(), &reader));

    std::shared_ptr<arrow::Table> piece;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&piece));

    if (i == 0) {
      concat_table = piece;
    } else {
      auto concat_result =
          arrow::ConcatenateTables({concat_table, piece},
                                   arrow::ConcatenateTablesOptions{false});
      if (!concat_result.ok()) {
        std::cerr << "concat_parquet_files: concat failed: "
                  << concat_result.status().ToString() << "\n";
        return;
      }
      concat_table = concat_result.ValueOrDie();
    }
  }

  auto out_result = arrow::io::FileOutputStream::Open(out_file);
  if (!out_result.ok()) {
    std::cerr << "concat_parquet_files: output open failed: "
              << out_result.status().ToString() << "\n";
    return;
  }
  auto out = out_result.ValueOrDie();

  PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(
      *concat_table, arrow::default_memory_pool(), out, 4096));
}
