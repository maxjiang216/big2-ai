#ifndef START_LEGAL_MOVES_FEATURE_H
#define START_LEGAL_MOVES_FEATURE_H

#include "feature_extractor.h"

// Game-level feature: number of legal moves for the player who opens the game
// (size of the legal move list on turn 0, before any card is played).
class StartLegalMovesFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::GameLevel; }
  std::string name() const override { return "start_legal_moves"; }

  int gameExtract(const GameRecord &record) const override {
    const auto &turns = record.turns();
    if (turns.empty())
      return -1;
    return static_cast<int>(turns[0].legal_moves.size());
  }
};

#endif
