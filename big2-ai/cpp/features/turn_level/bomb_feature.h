// bomb_feature.h

#ifndef BOMB_FEATURE_H
#define BOMB_FEATURE_H

#include "../../game_record.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

class BombFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }

  std::string name() const override { return "n_bombs"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> bombs;
    for (const auto &turn : record.turns()) {
      // Output for both player perspectives
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        bombs.push_back(view.count_bombs());
      }
    }
    return bombs;
  }
};

#endif // BOMB_FEATURE_H
