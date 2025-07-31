#ifndef POSSIBLE_MOVES_NOT_BOMB_FEATURE_H
#define POSSIBLE_MOVES_NOT_BOMB_FEATURE_H

#include "../../game_record.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

// May be more useful as bombs are often not relevant and increase variance of
// the count by a lot
class PossibleMovesNotBombFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }

  std::string name() const override { return "possible_moves_not_bomb"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        // If it's NOT the player's turn (so it's opponent's turn)
        if (view.turn() != 0) {
          features.push_back(view.get_possible_moves_not_bomb().size());
        } else {
          features.push_back(0);
        }
      }
    }
    return features;
  }
};

#endif // POSSIBLE_MOVES_NOT_BOMB_FEATURE_H
