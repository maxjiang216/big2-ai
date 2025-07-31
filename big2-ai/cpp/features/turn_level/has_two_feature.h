// has_two_feature.h

#ifndef HAS_TWO_FEATURE_H
#define HAS_TWO_FEATURE_H

#include "../../game_record.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

class HasTwoFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "has_two"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      // For both player perspectives
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        const auto &hand = view.player_hand();
        // Assuming 2s are encoded as rank 15 (standard for Big 2)
        bool has_two = (hand[12] > 0); // If 2 is at index 12 (0=3,...,12=2)
        features.push_back(has_two ? 1 : 0);
      }
    }
    return features;
  }
};

#endif // HAS_TWO_FEATURE_H
