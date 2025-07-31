// last_trick_is_straight_feature.h

#ifndef LAST_TRICK_IS_STRAIGHT_FEATURE_H
#define LAST_TRICK_IS_STRAIGHT_FEATURE_H

#include "../../game_record.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

template <int Type> class LastTrickIsStraightFeature : public FeatureExtractor {
public:
  // Type: 1 = single straight (5–13), 2 = double straight (2–8), 3 = triple
  // straight (2–5)

  // Assuming FeatureExtractor::Type is an enum with TurnLevel as a member
  // Adjust this return value based on your actual FeatureExtractor::Type enum
  FeatureExtractor::Type type() const override {
    return FeatureExtractor::Type::TurnLevel;
  }

  std::string name() const override;

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
  bool checkIsStraight(Move::Combination comb) const;
};

// Template specializations for name() function
template <> inline std::string LastTrickIsStraightFeature<1>::name() const {
  return "last_trick_is_single_straight";
}

template <> inline std::string LastTrickIsStraightFeature<2>::name() const {
  return "last_trick_is_double_straight";
}

template <> inline std::string LastTrickIsStraightFeature<3>::name() const {
  return "last_trick_is_triple_straight";
}

// Template specializations for checkIsStraight() function
template <>
inline bool
LastTrickIsStraightFeature<1>::checkIsStraight(Move::Combination comb) const {
  // Single straights (5 to 13)
  return (comb >= Move::Combination::kStraight5 &&
          comb <= Move::Combination::kStraight13);
}

template <>
inline bool
LastTrickIsStraightFeature<2>::checkIsStraight(Move::Combination comb) const {
  // Double straights (2 to 8)
  return (comb >= Move::Combination::kDoubleStraight2 &&
          comb <= Move::Combination::kDoubleStraight8);
}

template <>
inline bool
LastTrickIsStraightFeature<3>::checkIsStraight(Move::Combination comb) const {
  // Triple straights (2 to 5)
  return (comb >= Move::Combination::kTripleStraight2 &&
          comb <= Move::Combination::kTripleStraight5);
}

#endif // LAST_TRICK_IS_STRAIGHT_FEATURE_H