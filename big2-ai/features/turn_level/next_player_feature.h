#ifndef NEXT_PLAYER_FEATURE_H
#define NEXT_PLAYER_FEATURE_H

#include "../feature_extractor.h"
#include <string>
#include <vector>

/**
 * @brief Turn-level feature: 1 if it's this player's turn to move, else 0.
 * For each turn, outputs for both perspectives (player 0 and 1).
 */
class NextPlayerFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "next_player"; }

  std::vector<int> turnExtract(const GameRecord &game) const override {
    std::vector<int> features;
    const auto &turns = game.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        features.push_back(turn.current_player == perspective ? 1 : 0);
      }
    }
    return features;
  }
};

#endif // NEXT_PLAYER_FEATURE_H
