// ace_feature.h

#ifndef ACE_FEATURE_H
#define ACE_FEATURE_H

#include "../../game_record.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

class AceFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "n_aces"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      // For both player perspectives
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        const auto &hand = view.player_hand();
        // Assuming As are encoded as rank 14 (standard for Big 2)
        features.push_back(hand[11]);
      }
    }
    return features;
  }
};

#endif // ACE_FEATURE_H
