#ifndef POSSIBLE_MOVES_FEATURE_H
#define POSSIBLE_MOVES_FEATURE_H

#include "feature_extractor.h"

// Turn-level feature: number of possible moves the opponent could make from
// this perspective. Non-zero only when it is NOT this player's turn
// (view.turn() != 0 means the opponent is to move from our perspective).
class PossibleMovesFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "possible_moves"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        const PartialGame &view = turn.views[p];
        features.push_back(view.turn() != 0
                               ? static_cast<int>(view.get_possible_moves().size())
                               : 0);
      }
    }
    return features;
  }
};

#endif
