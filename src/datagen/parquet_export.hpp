#ifndef PARQUET_EXPORT_HPP
#define PARQUET_EXPORT_HPP

#include <memory>
#include <string>
#include <vector>

class FeatureExtractor;
class GameRecord;

void export_parquet_features(
    const std::vector<GameRecord> &records,
    const std::string &game_feature_out,
    const std::string &turn_feature_out,
    const std::vector<std::shared_ptr<FeatureExtractor>> &game_level_features,
    const std::vector<std::shared_ptr<FeatureExtractor>> &turn_level_features);

void concat_parquet_files(const std::vector<std::string> &parts,
                          const std::string &out_file);

#endif
