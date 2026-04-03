#ifndef RANK_GE_FEATURE_H
#define RANK_GE_FEATURE_H

#include "feature_extractor.h"
#include <stdexcept>

// Turn-level feature: count of cards with rank >= threshold in each perspective's hand.
class RankGeFeature : public FeatureExtractor {
public:
  explicit RankGeFeature(int rank, const std::string &rank_name)
      : rank_(rank), rank_name_(rank_name) {
    if (rank < 0 || rank > 12)
      throw std::invalid_argument("Rank must be in [0, 12]");
  }

  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "n_ge_" + rank_name_; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        const auto &hand = turn.views[p].player_hand();
        int count = 0;
        for (int i = rank_; i <= 12; ++i)
          count += hand[i];
        features.push_back(count);
      }
    }
    return features;
  }

private:
  int rank_;
  std::string rank_name_;
};

#endif
