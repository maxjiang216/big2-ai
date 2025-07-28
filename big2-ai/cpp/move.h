#ifndef MOVE_H
#define MOVE_H

#include <iostream>
#include <stdexcept>
class Move {
public:
  // Enumerated types for all legal move combinations.
  enum class Combination {
    kPass,
    kSingle,
    kDouble,
    kTriple,
    kFullHouse,
    kBomb,
    // Straights of singles: lengths 5..13.
    kStraight5,
    kStraight6,
    kStraight7,
    kStraight8,
    kStraight9,
    kStraight10,
    kStraight11,
    kStraight12,
    kStraight13,
    // Sisters (consecutive doubles): lengths 2..8.
    kDoubleStraight2,
    kDoubleStraight3,
    kDoubleStraight4,
    kDoubleStraight5,
    kDoubleStraight6,
    kDoubleStraight7,
    kDoubleStraight8,
    // Triple straights: lengths 2..5.
    kTripleStraight2,
    kTripleStraight3,
    kTripleStraight4,
    kTripleStraight5,
  };

  Combination combination{Combination::kPass};
  // For most moves, rank holds the “primary” card value.
  // We assume 3 = 3, 4 = 4, …, 10 = 10, J = 11, Q = 12, K = 13, A = 14, 2 = 15.
  int rank{0};
  // For full houses and bombs the auxiliary field is used.
  int auxiliary{0};

  Move(Combination comb, int r = 0, int aux = 0)
      : combination(comb), rank(r), auxiliary(aux) {}
  Move(int encoded_move);

  int numCards() const;
};

// Encode a move as an integer in the range [0, TOTAL_MOVES-1].
// (Legal moves are linear‐indexed as follows: 0 = pass, then 13 singles, 12
// doubles, 11 triples, 132 full houses, 156 bombs, 54 straights, 56 sisters and
// 38 triple straights.)
int encodeMove(const Move &move);

// Print a move
std::ostream &operator<<(std::ostream &os, const Move &move);

#endif
