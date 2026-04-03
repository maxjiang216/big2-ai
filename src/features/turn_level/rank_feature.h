#ifndef RANK_FEATURE_H
#define RANK_FEATURE_H

#include "feature_extractor.h"
#include <stdexcept>

// Turn-level feature: count of a specific rank in each perspective's hand.
// Rank index 0=3, 1=4, ..., 10=K, 11=A, 12=2.
class RankFeature : public FeatureExtractor {
public:
  explicit RankFeature(int rank, const std::string &rank_name)
      : rank_(rank), rank_name_(rank_name) {
    if (rank < 0 || rank > 12)
      throw std::invalid_argument("Rank must be in [0, 12]");
  }

  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "n_" + rank_name_; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        features.push_back(turn.views[p].player_hand()[rank_]);
      }
    }
    return features;
  }

private:
  int rank_;
  std::string rank_name_;
};

#endif
