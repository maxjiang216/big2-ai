// highest_rank_with_count_not_bomb_feature.h

#ifndef HIGHEST_RANK_WITH_COUNT_NOT_BOMB_FEATURE_H
#define HIGHEST_RANK_WITH_COUNT_NOT_BOMB_FEATURE_H

#include "../../core/game_record.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

/**
 * @brief Turn-level feature: highest rank with at least N copies, excluding bombs.
 * Parameterized by count (2=double, 3=triple).
 * A bomb is 4 of a kind (or 3 Aces).
 * Returns rank index (0-12) or -1 if no such rank exists.
 */
class HighestRankWithCountNotBombFeature : public FeatureExtractor {
public:
  /**
   * @param count Minimum number of copies required (2-3)
   * @param feature_name Human-readable name (e.g., "double_not_bomb", "triple_not_bomb")
   */
  explicit HighestRankWithCountNotBombFeature(int count, const std::string& feature_name)
      : count_(count), feature_name_(feature_name) {
    if (count < 2 || count > 3) {
      throw std::invalid_argument("Count must be 2 or 3 (1 doesn't need bomb exclusion, 4 is a bomb)");
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
          bool is_bomb = (i == 11 && hand[i] == 3) || // Ace bomb (3 copies)
                         (i != 11 && hand[i] == 4);    // Regular bomb (4 copies)
          
          if (hand[i] >= count_ && !is_bomb) {
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

#endif // HIGHEST_RANK_WITH_COUNT_NOT_BOMB_FEATURE_H