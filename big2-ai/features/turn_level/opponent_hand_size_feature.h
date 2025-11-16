// opponent_hand_size_feature.h

#ifndef OPPONENT_HAND_SIZE_FEATURE_H
#define OPPONENT_HAND_SIZE_FEATURE_H

#include "../feature_extractor.h"
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Turn-level feature: player's hand size at each turn.
 */
class OpponentHandSizeFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "opponent_hand_size"; }

  // Extract for every turn: output vector has length == total number of turns
  // in all games
  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    // Assuming GameRecord has method/field: const std::vector<TurnRecord>&
    // turns() const;
    const auto &turns = record.turns(); // or: record._turns;
    for (const auto &turn : turns) {
      // Get hand size for the player who made the move
      // This assumes TurnRecord has a field: int current_player;
      // and a Game or PartialGame with method get_player_hand_size(int)
      int hand_size = 0;
      // This assumes you stored the pre-move Game state in TurnRecord
      hand_size = turn.game.get_player_hand_size(1);
      features.push_back(hand_size);
      hand_size = turn.game.get_player_hand_size(0);
      features.push_back(hand_size);
    }
    return features;
  }
};

#endif // OPPONENT_HAND_SIZE_FEATURE_H
