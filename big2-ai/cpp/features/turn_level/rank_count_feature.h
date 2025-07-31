// rank_count_feature.h

#ifndef RANK_COUNT_FEATURE_H
#define RANK_COUNT_FEATURE_H

#include "../../game_record.h"
#include "../feature_extractor.h"
#include <sstream>
#include <string>
#include <vector>

// Template parameter RankIdx: index in hand array (0 = 3, 1 = 4, ..., 11 = Ace,
// 12 = 2)
template <int RankIdx> class RankCountFeature : public FeatureExtractor {
public:
  RankCountFeature(const std::string &name) : _name(name) {}
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return _name; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        const auto &hand = view.player_hand();
        features.push_back(hand[RankIdx]);
      }
    }
    return features;
  }

private:
  std::string _name;
};

// Helper macro for common names
#define DEFINE_RANK_COUNT_FEATURE(RANK_IDX, NAME)                              \
  using NAME##Feature = RankCountFeature<RANK_IDX>;

#endif // RANK_COUNT_FEATURE_H
