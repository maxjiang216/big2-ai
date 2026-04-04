#include "parquet_export.hpp"

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
    const std::vector<std::pair<int, GameRecord>> &indexed,
    const std::string &game_feature_out,
    const std::string &turn_feature_out,
    const std::vector<std::shared_ptr<FeatureExtractor>> &game_level_features,
    const std::vector<std::shared_ptr<FeatureExtractor>> &turn_level_features) {

  auto export_game = [&]() {
    if (indexed.empty()) {
      std::cout << "No game records to export.\n";
      return;
    }

    if (game_level_features.empty()) {
      std::cout << "Exporting " << indexed.size()
                << " game_index rows only (no game-level features).\n";
    } else {
      std::cout << "Exporting " << indexed.size() << " game records with "
                << game_level_features.size() << " features...\n";
    }

    try {
      std::vector<std::shared_ptr<arrow::Array>> columns;
      std::vector<std::shared_ptr<arrow::Field>> schema_fields;

      {
        arrow::Int32Builder idx_builder;
        for (const auto &pr : indexed) {
          auto status = idx_builder.Append(pr.first);
          if (!status.ok()) {
            std::cerr << "Error appending game_index: " << status.ToString()
                      << std::endl;
            return;
          }
        }
        std::shared_ptr<arrow::Array> arr;
        auto status = idx_builder.Finish(&arr);
        if (!status.ok()) {
          std::cerr << "Error finishing game_index: " << status.ToString()
                    << std::endl;
          return;
        }
        columns.push_back(arr);
        schema_fields.push_back(arrow::field("game_index", arrow::int32()));
      }

      for (const auto &extractor : game_level_features) {
        if (!extractor) {
          std::cerr << "Warning: Null feature extractor found, skipping.\n";
          continue;
        }

        arrow::Int32Builder builder;
        for (const auto &pr : indexed) {
          int feature_value = extractor->gameExtract(pr.second);
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
      auto table = arrow::Table::Make(schema, columns, indexed.size());

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
    if (indexed.empty()) {
      std::cout << "No game records to export.\n";
      return;
    }

    if (turn_level_features.empty()) {
      std::cout << "Exporting turn table (game_index, turn_idx, perspective only).\n";
    } else {
      std::cout << "Exporting turn features for " << indexed.size()
                << " games with " << turn_level_features.size()
                << " features...\n";
    }

    try {
      std::vector<int> game_index_col;
      std::vector<int> turn_idx_col;
      std::vector<int> perspective_col;
      for (const auto &pr : indexed) {
        int gi = pr.first;
        const auto &rec = pr.second;
        for (size_t t = 0; t < rec.turns().size(); ++t) {
          for (int p = 0; p < 2; ++p) {
            game_index_col.push_back(gi);
            turn_idx_col.push_back(static_cast<int>(t));
            perspective_col.push_back(p);
          }
        }
      }
      size_t total_turns = game_index_col.size();

      if (total_turns == 0) {
        std::cout << "No turns found to export.\n";
        return;
      }

      std::vector<std::vector<int>> all_columns(turn_level_features.size());
      for (size_t f = 0; f < turn_level_features.size(); ++f) {
        if (!turn_level_features[f]) {
          std::cerr << "Warning: Null turn feature extractor at index " << f
                    << ", skipping.\n";
          continue;
        }

        for (const auto &pr : indexed) {
          auto feature_vals = turn_level_features[f]->turnExtract(pr.second);
          all_columns[f].insert(all_columns[f].end(), feature_vals.begin(),
                                feature_vals.end());
        }
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

      auto append_int_col = [&](const char *name, const std::vector<int> &vals) {
        arrow::Int32Builder builder;
        for (auto v : vals) {
          auto status = builder.Append(v);
          if (!status.ok()) {
            std::cerr << "Error appending " << name << ": " << status.ToString()
                      << std::endl;
            return false;
          }
        }
        std::shared_ptr<arrow::Array> arr;
        auto status = builder.Finish(&arr);
        if (!status.ok())
          return false;
        columns.push_back(arr);
        schema_fields.push_back(arrow::field(name, arrow::int32()));
        return true;
      };

      if (!append_int_col("game_index", game_index_col))
        return;
      if (!append_int_col("turn_idx", turn_idx_col))
        return;
      if (!append_int_col("perspective", perspective_col))
        return;

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
