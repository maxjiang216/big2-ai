#ifndef OPP_CANNOT_RESPOND_FEATURE_H
#define OPP_CANNOT_RESPOND_FEATURE_H

#include "feature_extractor.h"
#include "move.h"
#include "util.h"

// Turn-level feature: 1 if the opponent definitely cannot respond to the move
// played this turn (based on the mover's max-possible view of opponent's cards),
// 0 otherwise.  Emitted for the mover's perspective; 0 for the observer.
//
// Uses opponent_can_respond() which checks whether any max-possible-consistent
// opponent hand has at least one legal response to the move.  Returns 0 for
// pass moves since the concept of "opponent cannot respond" doesn't apply.
class OppCannotRespondFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "opp_cannot_respond"; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      int move_id = encodeMove(turn.move);
      int flag = 0;
      if (move_id != kPASS) {
        const auto &view = turn.views[turn.current_player];
        flag = opponent_can_respond(move_id, view.player_hand(),
                                    view.discard_pile(),
                                    view.opponent_hand_size())
                   ? 0
                   : 1;
      }
      for (int p = 0; p < 2; ++p)
        features.push_back(p == turn.current_player ? flag : 0);
    }
    return features;
  }
};

#endif
