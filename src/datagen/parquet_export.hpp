#ifndef PARQUET_EXPORT_HPP
#define PARQUET_EXPORT_HPP

#include "game_record.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

class FeatureExtractor;

void export_parquet_features(
    const std::vector<std::pair<int, GameRecord>> &indexed,
    const std::string &game_feature_out,
    const std::string &turn_feature_out,
    const std::vector<std::shared_ptr<FeatureExtractor>> &game_level_features,
    const std::vector<std::shared_ptr<FeatureExtractor>> &turn_level_features);

void concat_parquet_files(const std::vector<std::string> &parts,
                          const std::string &out_file);

#endif
