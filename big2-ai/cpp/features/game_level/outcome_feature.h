// outcome_feature.h
#ifndef OUTCOME_FEATURE_H
#define OUTCOME_FEATURE_H

#include "../feature_extractor.h"

class OutcomeFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::GameLevel; }

  std::string name() const override { return "outcome"; }

  int gameExtract(const GameRecord &record) const override {
    return record.game().get_winner();
  }
};

#endif // OUTCOME_FEATURE_H
