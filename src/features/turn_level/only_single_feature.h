#ifndef ONLY_SINGLE_FEATURE_H
#define ONLY_SINGLE_FEATURE_H

#include "feature_extractor.h"
#include "util.h"

// Turn-level feature: 1 if the perspective hand has no pair or better (every
// rank count is 0 or 1), else 0.
class OnlySingleFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "only_single"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        features.push_back(hand_is_only_singles(turn.views[p].player_hand()) ? 1 : 0);
      }
    }
    return features;
  }
};

#endif
