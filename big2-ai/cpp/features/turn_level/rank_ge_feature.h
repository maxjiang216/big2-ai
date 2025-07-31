#ifndef RANK_GE_FEATURE_H
#define RANK_GE_FEATURE_H

#include "../../game_record.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

/**
 * @brief Counts the number of cards with rank >= MinRankIdx in the player's
 * hand. Indices: 0 = 3, ..., 11 = Ace, 12 = Two
 */
template <int MinRankIdx>
class RankGreaterEqualFeature : public FeatureExtractor {
public:
  RankGreaterEqualFeature(const std::string &name) : _name(name) {}
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return _name; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        const auto &hand = view.player_hand();
        int sum = 0;
        for (int i = MinRankIdx; i < 13; ++i) {
          sum += hand[i];
        }
        features.push_back(sum);
      }
    }
    return features;
  }

private:
  std::string _name;
};

#endif // RANK_GE_FEATURE_H
