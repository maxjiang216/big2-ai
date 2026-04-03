#ifndef POSSIBLE_MOVES_NOT_BOMB_FEATURE_H
#define POSSIBLE_MOVES_NOT_BOMB_FEATURE_H

#include "feature_extractor.h"

// Turn-level feature: like possible_moves but excluding bomb plays.
class PossibleMovesNotBombFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "possible_moves_not_bomb"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        const PartialGame &view = turn.views[p];
        features.push_back(view.turn() != 0
                               ? static_cast<int>(view.get_possible_moves_not_bomb().size())
                               : 0);
      }
    }
    return features;
  }
};

#endif
