#ifndef TB_CASE_FEATURE_H
#define TB_CASE_FEATURE_H

#include "feature_extractor.h"

// Turn-level feature: tablebase case used by the moving player this turn.
// Values: -1 = no tablebase move (normal play), 1 = forced-win sequence,
//         2 = opponent-has-1-card endgame.
// Emitted once per (turn, perspective) pair. When next_player==1 for
// perspective p, this records p's own tb_case for the move they just chose.
// When next_player==0, it records the opponent's tb_case.
class TbCaseFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "tb_case"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      // Same value for both perspectives; Python uses next_player to interpret.
      features.push_back(turn.tb_case);
      features.push_back(turn.tb_case);
    }
    return features;
  }
};

#endif
