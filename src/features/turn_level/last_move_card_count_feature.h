#ifndef LAST_MOVE_CARD_COUNT_FEATURE_H
#define LAST_MOVE_CARD_COUNT_FEATURE_H

#include "feature_extractor.h"
#include "move.h"
#include "util.h"

// Turn-level feature: number of cards in the last move (0 for a pass).
class LastMoveCardCountFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "last_move_card_count"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        int encoded = encodeMove(turn.views[p].last_move());
        features.push_back(MOVE_TO_CARDS[encoded][13]);
      }
    }
    return features;
  }
};

#endif
