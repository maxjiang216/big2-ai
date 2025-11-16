// last_trick_is_straight_feature.h

#ifndef LAST_MOVE_IS_STRAIGHT_FEATURE_H
#define LAST_MOVE_IS_STRAIGHT_FEATURE_H

#include "../../core/game_record.h"
#include "../../core/move.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

/**
 * @brief Turn-level feature: 1 if last move is a straight of the specified type, else 0.
 * Parameterized by straight type (1=single, 2=double, 3=triple).
 */
class LastMoveIsStraightFeature : public FeatureExtractor {
public:
  /**
   * @param straight_type Type of straight (1=single, 2=double, 3=triple)
   * @param feature_name Human-readable name (e.g., "single_straight", "double_straight", "triple_straight")
   */
  explicit LastMoveIsStraightFeature(int straight_type, const std::string& feature_name)
      : straight_type_(straight_type), feature_name_(feature_name) {
    if (straight_type < 1 || straight_type > 3) {
      throw std::invalid_argument("Straight type must be 1 (single), 2 (double), or 3 (triple)");
    }
  }

  Type type() const override { return Type::TurnLevel; }
  
  std::string name() const override { 
    return "last_move_is_" + feature_name_; 
  }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        Move::Combination comb = view.last_move().combination;
        bool is_straight = checkIsStraight(comb);
        features.push_back(is_straight ? 1 : 0);
      }
    }
    return features;
  }

private:
  int straight_type_;
  std::string feature_name_;

  bool checkIsStraight(Move::Combination comb) const {
    switch (straight_type_) {
      case 1: // Single straights (5 to 13)
        return (comb >= Move::Combination::kStraight5 &&
                comb <= Move::Combination::kStraight13);
      
      case 2: // Double straights (2 to 8)
        return (comb >= Move::Combination::kDoubleStraight2 &&
                comb <= Move::Combination::kDoubleStraight8);
      
      case 3: // Triple straights (2 to 5)
        return (comb >= Move::Combination::kTripleStraight2 &&
                comb <= Move::Combination::kTripleStraight5);
      
      default:
        return false;
    }
  }
};

#endif // LAST_MOVE_IS_STRAIGHT_FEATURE_H