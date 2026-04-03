#ifndef OPPONENT_HAND_SIZE_FEATURE_H
#define OPPONENT_HAND_SIZE_FEATURE_H

#include "feature_extractor.h"

// Turn-level feature: opponent's hand size from each perspective's point of view (pre-move).
class OpponentHandSizeFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "opponent_hand_size"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        features.push_back(turn.game.get_player_hand_size(1 - p));
      }
    }
    return features;
  }
};

#endif
