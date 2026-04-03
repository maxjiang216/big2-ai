#ifndef TABLEBASE_HITS_FEATURE_H
#define TABLEBASE_HITS_FEATURE_H

#include "feature_extractor.h"

// Game-level feature: number of tablebase "episodes" — first turn of each
// consecutive case1 or case2 run per player (not every move in a sequence).
class TablebaseHitsFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::GameLevel; }
  std::string name() const override { return "tb_hits"; }

  int gameExtract(const GameRecord &record) const override {
    auto s = tablebase_first_hit_stats(record);
    return s.case1 + s.case2;
  }
};

#endif
