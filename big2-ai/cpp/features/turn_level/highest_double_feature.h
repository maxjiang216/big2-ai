// highest_double_feature.h

#ifndef HIGHEST_DOUBLE_FEATURE_H
#define HIGHEST_DOUBLE_FEATURE_H

#include "../../game_record.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

class HighestDoubleFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "highest_double"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        const auto &hand = view.player_hand();
        int highest_rank = -1;          // 0 means no cards, otherwise 3-15
        for (int i = 11; i >= 0; --i) { // from 2 down to 3
          if (hand[i] > 1) {
            highest_rank = i;
            break;
          }
        }
        features.push_back(highest_rank);
      }
    }
    return features;
  }
};

#endif // HIGHEST_DOUBLE_FEATURE_H
