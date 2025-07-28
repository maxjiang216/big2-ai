// player_hand_size_feature.h

#ifndef PLAYER_HAND_SIZE_FEATURE_H
#define PLAYER_HAND_SIZE_FEATURE_H

#include "../feature_extractor.h"
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Turn-level feature: player's hand size at each turn.
 */
class PlayerHandSizeFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "player_hand_size"; }

  // Extract for every turn: output vector has length == total number of turns
  // in all games
  std::vector<int> turnExtract(const GameRecord &game) const override {
    std::vector<int> features;
    // Assuming GameRecord has method/field: const std::vector<TurnRecord>&
    // turns() const;
    const auto &turns = game.turns(); // or: game._turns;
    for (const auto &turn : turns) {
      // Get hand size for the player who made the move
      // This assumes TurnRecord has a field: int current_player;
      // and a Game or PartialGame with method get_player_hand_size(int)
      int hand_size = 0;
      // This assumes you stored the pre-move Game state in TurnRecord
      hand_size = turn.game.get_player_hand_size(turn.current_player);
      features.push_back(hand_size);
    }
    return features;
  }
};

#endif // PLAYER_HAND_SIZE_FEATURE_H
