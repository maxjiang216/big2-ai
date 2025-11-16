#ifndef TURN_OUTCOME_FEATURE_H
#define TURN_OUTCOME_FEATURE_H

#include "../feature_extractor.h"
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Turn-level feature: outcome from each player's perspective for each
 * turn. For every turn, adds 1 if player i ultimately won the game, else 0.
 */
class TurnOutcomeFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "turn_outcome"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    // Assumes GameRecord provides the winner (int) and turns()
    // E.g., int winner = game.get_winner();
    int winner = record.game().get_winner();
    const auto &turns = record.turns();

    for (size_t t = 0; t < turns.size(); ++t) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        features.push_back(winner == perspective ? 1 : 0);
      }
    }
    return features;
  }
};

#endif // PLAYER_TURN_OUTCOME_FEATURE_H
