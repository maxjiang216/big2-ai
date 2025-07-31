#ifndef POSSIBLE_MOVES_FEATURE_H
#define POSSIBLE_MOVES_FEATURE_H

#include "../../game_record.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

class PossibleMovesFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }

  std::string name() const override { return "possible_moves"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        // If it's NOT the player's turn (so it's opponent's turn)
        if (view.turn() != 0) {
          features.push_back(view.get_possible_moves().size());
        } else {
          features.push_back(0);
        }
      }
    }
    return features;
  }
};

#endif // POSSIBLE_MOVES_FEATURE_H
