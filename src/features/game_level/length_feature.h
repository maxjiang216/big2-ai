#ifndef LENGTH_FEATURE_H
#define LENGTH_FEATURE_H

#include "feature_extractor.h"

class GameLengthExtractor : public FeatureExtractor {
public:
  Type type() const override { return Type::GameLevel; }
  std::string name() const override { return "length"; }

  int gameExtract(const GameRecord &record) const override {
    return static_cast<int>(record.turns().size());
  }
};

#endif
