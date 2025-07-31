// trick_rank_feature.h

#ifndef TRICK_RANK_FEATURE_H
#define TRICK_RANK_FEATURE_H

#include "../../game_record.h"
#include "../../move.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

class TrickRankFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "trick_rank"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        features.push_back(view.get_trick_rank());
      }
    }
    return features;
  }
};

#endif // TRICK_RANK_FEATURE_H
