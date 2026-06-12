#ifndef AZ_PI_PI_SOLVER_H
#define AZ_PI_PI_SOLVER_H

// Perfect-information exact endgame solver. Given a full Game state (both hands
// visible), decides whether the side to move wins or loses under optimal play
// by BOTH sides assuming each plays to win the current game.
//
// This is sound only in the perfect-information setting: there is no hidden
// information, so the minimax value is exact. Used at MCTS leaves once the total
// cards remaining is small enough to solve cheaply — a proven leaf needs no NN
// evaluation and contributes an exact 1/0 to the backup.
//
// Returns UNKNOWN when the position exceeds the card threshold or the node
// budget is exhausted (caller falls back to the NN value).

#include "game.h"

namespace az_pi {

enum class Proof { UNKNOWN, WIN, LOSS };  // for the side to move at the position

struct SolverLimits {
  int max_total_cards = 12;   // skip (UNKNOWN) when both hands sum above this
  long node_budget = 10000;   // abort (UNKNOWN) once this many nodes are visited
};

// Solve win/loss for g.current_player(). g must NOT already be terminal.
Proof solve(const Game &g, const SolverLimits &limits);

}  // namespace az_pi

#endif  // AZ_PI_PI_SOLVER_H
