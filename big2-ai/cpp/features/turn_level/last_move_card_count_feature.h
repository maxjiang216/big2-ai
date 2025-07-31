// turn_level/last_move_card_count_feature.h

#ifndef LAST_MOVE_CARD_COUNT_FEATURE_H
#define LAST_MOVE_CARD_COUNT_FEATURE_H

#include "../../game_record.h"
#include "../../move.h" // For Move and encodeMove
#include "../../util.h" // For MOVE_TO_CARDS
#include "../feature_extractor.h"
#include <stdexcept>
#include <string>
#include <vector>

class LastMoveCardCountFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "last_move_card_count"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        // Get last move object (adapt accessor if needed)
        const Move &last_move = view.last_move();
        int encoded = encodeMove(last_move);

        int count = 0;
        auto it = MOVE_TO_CARDS.find(encoded);
        if (it != MOVE_TO_CARDS.end()) {
          count = it->second.back(); // Last element: total card count
        }
        features.push_back(count);
      }
    }
    return features;
  }
};

#endif // LAST_MOVE_CARD_COUNT_FEATURE_H
