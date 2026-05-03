#ifndef HINT_FEATURE_H
#define HINT_FEATURE_H

#include "feature_extractor.h"
#include "hint_compute.h"
#include "move.h"

// Turn-level feature: MC estimate of the probability that the opponent can
// respond to the move played this turn, from the moving player's perspective.
//
// Value is scaled by 10000 and stored as int32 (divide by 10000.0 in Python).
// Emitted only for the mover's perspective row; 0 is emitted for the observer.
//
// Semantics:
//   0     - player passed (no trick to contest)
//   10000 - opponent is definitely locked (no max-possible hand can respond)
//   other - MC fraction × 10000
class HintFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "hint"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      int move_id = encodeMove(turn.move);
      // Compute once for the mover; emit 0 for the observer.
      float h = compute_hint(move_id, turn.views[turn.current_player].player_hand(),
                             turn.views[turn.current_player].discard_pile(),
                             turn.views[turn.current_player].opponent_hand_size());
      int h_int = static_cast<int>(h * 10000.0f + 0.5f);
      for (int p = 0; p < 2; ++p)
        features.push_back(p == turn.current_player ? h_int : 0);
    }
    return features;
  }
};

#endif
