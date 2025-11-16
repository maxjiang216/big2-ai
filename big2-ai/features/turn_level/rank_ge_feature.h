// rank_ge_feature.h

#ifndef RANK_GE_FEATURE_H
#define RANK_GE_FEATURE_H

#include "../../core/game_record.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

/**
 * @brief Turn-level feature: count of cards with rank >= specified rank.
 * Parameterized by rank (0-12, representing 3-2 in Big 2).
 */
class RankGeFeature : public FeatureExtractor {
public:
  /**
   * @param rank The threshold rank (0=3, 1=4, ..., 10=K, 11=A, 12=2)
   * @param rank_name Human-readable name for the rank (e.g., "a", "2", "k")
   */
  explicit RankGeFeature(int rank, const std::string& rank_name)
      : rank_(rank), rank_name_(rank_name) {
    if (rank < 0 || rank > 12) {
      throw std::invalid_argument("Rank must be in range [0, 12]");
    }
  }

  Type type() const override { return Type::TurnLevel; }
  
  std::string name() const override { 
    return "n_ge_" + rank_name_; 
  }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      // For both player perspectives
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        const auto &hand = view.player_hand();
        
        // Count all cards with rank >= rank_
        int count = 0;
        for (int i = rank_; i <= 12; ++i) {
          count += hand[i];
        }
        
        features.push_back(count);
      }
    }
    return features;
  }

private:
  int rank_;
  std::string rank_name_;
};

#endif // RANK_GE_FEATURE_H