#ifndef LAST_MOVE_IS_STRAIGHT_FEATURE_H
#define LAST_MOVE_IS_STRAIGHT_FEATURE_H

#include "feature_extractor.h"
#include "move.h"
#include <stdexcept>

// Turn-level feature: 1 if the last move is a straight of the given type.
// straight_type: 1=single straight, 2=double straight, 3=triple straight.
class LastMoveIsStraightFeature : public FeatureExtractor {
public:
  explicit LastMoveIsStraightFeature(int straight_type,
                                     const std::string &feature_name)
      : straight_type_(straight_type), feature_name_(feature_name) {
    if (straight_type < 1 || straight_type > 3)
      throw std::invalid_argument("straight_type must be 1, 2, or 3");
  }

  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "last_move_is_" + feature_name_; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        features.push_back(is_straight(turn.views[p].last_move().combination) ? 1 : 0);
      }
    }
    return features;
  }

private:
  int straight_type_;
  std::string feature_name_;

  bool is_straight(Move::Combination c) const {
    switch (straight_type_) {
    case 1:
      return c >= Move::Combination::kStraight5 && c <= Move::Combination::kStraight13;
    case 2:
      return c >= Move::Combination::kDoubleStraight2 &&
             c <= Move::Combination::kDoubleStraight8;
    case 3:
      return c >= Move::Combination::kTripleStraight2 &&
             c <= Move::Combination::kTripleStraight5;
    default:
      return false;
    }
  }
};

#endif
