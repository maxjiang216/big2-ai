// rank_le_feature.h

#ifndef RANK_LE_FEATURE_H
#define RANK_LE_FEATURE_H

#include "../../game_record.h"
#include "../feature_extractor.h"
#include <array>
#include <sstream>
#include <string>
#include <vector>

template <int N> class RankLessEqualFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }

  std::string name() const override {
    static const char *names[] = {"three", "four", "five", "six",  "seven",
                                  "eight", "nine", "ten",  "jack", "queen",
                                  "king",  "ace",  "two"};
    std::ostringstream ss;
    ss << "n_le_" << names[N];
    return ss.str();
  }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        const auto &hand = view.player_hand();
        int sum = 0;
        for (int i = 0; i <= N; ++i)
          sum += hand[i];
        features.push_back(sum);
      }
    }
    return features;
  }
};

#endif // RANK_LE_FEATURE_H
