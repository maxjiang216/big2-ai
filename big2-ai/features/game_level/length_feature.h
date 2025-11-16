#ifndef GAME_LENGTH_EXTRACTOR_H
#define GAME_LENGTH_EXTRACTOR_H

#include "feature_extractor.h"
#include "game_record.h"

class GameLengthExtractor : public FeatureExtractor {
public:
  Type type() const override { return Type::GameLevel; }

  std::string name() const override { return "game_length"; }

  int gameExtract(const GameRecord &game) const override {
    return static_cast<int>(game.turns().size());
  }
};

#endif // GAME_LENGTH_EXTRACTOR_H
