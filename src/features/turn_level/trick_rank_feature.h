#ifndef TRICK_RANK_FEATURE_H
#define TRICK_RANK_FEATURE_H

#include "feature_extractor.h"

// Turn-level feature: rank of the current trick from each perspective.
class TrickRankFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "trick_rank"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        features.push_back(turn.views[p].get_trick_rank());
      }
    }
    return features;
  }
};

#endif
