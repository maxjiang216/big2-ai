#ifndef NEXT_PLAYER_FEATURE_H
#define NEXT_PLAYER_FEATURE_H

#include "feature_extractor.h"

// Turn-level feature: 1 if it is this perspective's player's turn to move.
class NextPlayerFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "next_player"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        features.push_back(turn.current_player == p ? 1 : 0);
      }
    }
    return features;
  }
};

#endif
