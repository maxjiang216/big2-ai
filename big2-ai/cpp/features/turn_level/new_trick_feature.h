// new_trick_feature.h

#ifndef NEW_TRICK_FEATURE_H
#define NEW_TRICK_FEATURE_H

#include "../../game_record.h"
#include "../../move.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

class NewTrickFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "new_trick"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (size_t i = 0; i < turns.size(); ++i) {
      // For both perspectives
      for (int perspective = 0; perspective < 2; ++perspective) {
        bool is_new_trick = false;
        if (i == 0) {
          // First move of the game is always a new trick
          is_new_trick = true;
        } else {
          // If the previous move was a pass, this is a new trick
          const Move &last_move = turns[i - 1].move;
          if (last_move.combination == Move::Combination::kPass) {
            is_new_trick = true;
          }
        }
        features.push_back(is_new_trick ? 1 : 0);
      }
    }
    return features;
  }
};

#endif // NEW_TRICK_FEATURE_H
