#ifndef HIGHEST_RANK_WITH_COUNT_FEATURE_H
#define HIGHEST_RANK_WITH_COUNT_FEATURE_H

#include "feature_extractor.h"
#include <stdexcept>

// Turn-level feature: highest rank index (0-12) with at least `count` copies
// in each perspective's hand. Returns -1 if no such rank exists.
class HighestRankWithCountFeature : public FeatureExtractor {
public:
  explicit HighestRankWithCountFeature(int count, const std::string &feature_name)
      : count_(count), feature_name_(feature_name) {
    if (count < 1 || count > 4)
      throw std::invalid_argument("Count must be in [1, 4]");
  }

  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "highest_" + feature_name_; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        const auto &hand = turn.views[p].player_hand();
        int highest = -1;
        for (int i = 12; i >= 0; --i) {
          if (hand[i] >= count_) { highest = i; break; }
        }
        features.push_back(highest);
      }
    }
    return features;
  }

private:
  int count_;
  std::string feature_name_;
};

#endif
