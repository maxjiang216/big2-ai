#ifndef LAST_MOVE_COMBINATION_FEATURE_H
#define LAST_MOVE_COMBINATION_FEATURE_H

#include "feature_extractor.h"
#include "move.h"

// Turn-level feature: 1 if the last move's combination matches the target, else 0.
class LastMoveCombinationFeature : public FeatureExtractor {
public:
  explicit LastMoveCombinationFeature(Move::Combination combo,
                                      const std::string &feature_name)
      : combo_(combo), feature_name_(feature_name) {}

  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "last_move_is_" + feature_name_; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        features.push_back(turn.views[p].last_move().combination == combo_ ? 1 : 0);
      }
    }
    return features;
  }

private:
  Move::Combination combo_;
  std::string feature_name_;
};

#endif
