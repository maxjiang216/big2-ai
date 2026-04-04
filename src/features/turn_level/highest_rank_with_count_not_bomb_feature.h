#ifndef HIGHEST_RANK_WITH_COUNT_NOT_BOMB_FEATURE_H
#define HIGHEST_RANK_WITH_COUNT_NOT_BOMB_FEATURE_H

#include "feature_extractor.h"
#include <stdexcept>

// Turn-level feature: highest rank index (0-12) with at least `count` copies,
// excluding bomb hands (4-of-a-kind, or 3 Aces). Returns -1 if none.
class HighestRankWithCountNotBombFeature : public FeatureExtractor {
public:
  explicit HighestRankWithCountNotBombFeature(int count, const std::string &feature_name)
      : count_(count), feature_name_(feature_name) {
    if (count < 1 || count > 3)
      throw std::invalid_argument("Count must be in [1, 3]");
  }

  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "highest_" + feature_name_; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        const auto &hand = turn.views[p].player_hand();
        int highest = -1;
        for (int i = 12; i >= 0; --i) {
          bool is_bomb = (i == 11 && hand[i] == 3) || (i != 11 && hand[i] == 4);
          if (hand[i] >= count_ && !is_bomb) { highest = i; break; }
        }
        features.push_back(highest);
      }
    }
    return features;
  }

private:
  int count_;
  std::string feature_name_;
};

#endif
