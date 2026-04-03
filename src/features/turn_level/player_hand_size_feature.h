#ifndef PLAYER_HAND_SIZE_FEATURE_H
#define PLAYER_HAND_SIZE_FEATURE_H

#include "feature_extractor.h"

// Turn-level feature: hand size for each perspective's player (pre-move).
class PlayerHandSizeFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "player_hand_size"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        features.push_back(turn.game.get_player_hand_size(p));
      }
    }
    return features;
  }
};

#endif
