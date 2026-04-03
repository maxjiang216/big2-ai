#ifndef TURN_OUTCOME_FEATURE_H
#define TURN_OUTCOME_FEATURE_H

#include "feature_extractor.h"

// Turn-level feature: 1 if this perspective's player ultimately won the game,
// 0 otherwise. Same value repeated for every turn of the game.
class TurnOutcomeFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "turn_outcome"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    int winner = record.game().get_winner();
    for (size_t t = 0; t < record.turns().size(); ++t) {
      for (int p = 0; p < 2; ++p) {
        features.push_back(winner == p ? 1 : 0);
      }
    }
    return features;
  }
};

#endif
