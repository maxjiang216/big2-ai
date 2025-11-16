// last_move_combination_feature.h

#ifndef LAST_MOVE_COMBINATION_FEATURE_H
#define LAST_MOVE_COMBINATION_FEATURE_H

#include "../../core/game_record.h"
#include "../../core/move.h"
#include "../feature_extractor.h"
#include <string>
#include <vector>

/**
 * @brief Turn-level feature: 1 if last move matches the given combination, else 0.
 * Parameterized by Move::Combination type.
 */
class LastMoveCombinationFeature : public FeatureExtractor {
public:
  /**
   * @param combo The Move::Combination to check for
   * @param feature_name Human-readable name for the feature (e.g., "bomb", "double", "straight5")
   */
  explicit LastMoveCombinationFeature(Move::Combination combo, const std::string& feature_name)
      : combo_(combo), feature_name_(feature_name) {}

  Type type() const override { return Type::TurnLevel; }
  
  std::string name() const override { 
    return "last_trick_is_" + feature_name_; 
  }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        const Move &last_move = view.last_move();
        int is_combo = (last_move.combination == combo_ ? 1 : 0);
        features.push_back(is_combo);
      }
    }
    return features;
  }

private:
  Move::Combination combo_;
  std::string feature_name_;
};

#endif // LAST_MOVE_COMBINATION_FEATURE_H