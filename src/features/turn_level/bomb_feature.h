#ifndef BOMB_FEATURE_H
#define BOMB_FEATURE_H

#include "feature_extractor.h"

// Turn-level feature: number of bomb hands in each perspective's hand.
class BombFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "n_bombs"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        features.push_back(turn.views[p].count_bombs());
      }
    }
    return features;
  }
};

#endif
