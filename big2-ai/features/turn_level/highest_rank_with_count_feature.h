// highest_rank_with_count_feature.h

#ifndef HIGHEST_RANK_WITH_COUNT_FEATURE_H
#define HIGHEST_RANK_WITH_COUNT_FEATURE_H

#include "../../core/game_record.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

/**
 * @brief Turn-level feature: highest rank with at least N copies.
 * Parameterized by count (1=single, 2=double, 3=triple, 4=bomb).
 * Returns rank index (0-12) or -1 if no such rank exists.
 */
class HighestRankWithCountFeature : public FeatureExtractor {
public:
  /**
   * @param count Minimum number of copies required (1-4)
   * @param feature_name Human-readable name (e.g., "single", "double", "triple", "bomb")
   */
  explicit HighestRankWithCountFeature(int count, const std::string& feature_name)
      : count_(count), feature_name_(feature_name) {
    if (count < 1 || count > 4) {
      throw std::invalid_argument("Count must be in range [1, 4]");
    }
  }

  Type type() const override { return Type::TurnLevel; }
  
  std::string name() const override { 
    return "highest_" + feature_name_; 
  }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        const auto &hand = view.player_hand();
        
        int highest_rank = -1; // -1 means no such rank exists
        
        // Search from highest rank (2) down to lowest (3)
        for (int i = 12; i >= 0; --i) {
          if (hand[i] >= count_) {
            highest_rank = i;
            break;
          }
        }
        
        features.push_back(highest_rank);
      }
    }
    return features;
  }

private:
  int count_;
  std::string feature_name_;
};

#endif // HIGHEST_RANK_WITH_COUNT_FEATURE_H