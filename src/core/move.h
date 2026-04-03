#ifndef MOVE_H
#define MOVE_H

#include <iostream>
#include <stdexcept>

class Move {
public:
  enum class Combination {
    kPass,
    kSingle,
    kDouble,
    kTriple,
    kFullHouse,
    kBomb,
    kStraight5,
    kStraight6,
    kStraight7,
    kStraight8,
    kStraight9,
    kStraight10,
    kStraight11,
    kStraight12,
    kStraight13,
    kDoubleStraight2,
    kDoubleStraight3,
    kDoubleStraight4,
    kDoubleStraight5,
    kDoubleStraight6,
    kDoubleStraight7,
    kDoubleStraight8,
    kTripleStraight2,
    kTripleStraight3,
    kTripleStraight4,
    kTripleStraight5,
  };

  Combination combination{Combination::kPass};
  int rank{0};
  int auxiliary{0};

  Move(Combination comb, int r = 0, int aux = 0)
      : combination(comb), rank(r), auxiliary(aux) {}
  Move(int encoded_move);

  int numCards() const;
};

int encodeMove(const Move &move);

std::ostream &operator<<(std::ostream &os, const Move &move);

#endif
