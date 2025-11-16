#include "move.h"
#include "util.h"

#include <iostream>
#include <stdexcept>

using namespace std;

Move::Move(int encoded_move) {
  if (encoded_move == kPASS) {
    combination = Combination::kPass;
  } else if (encoded_move < kDOUBLE_START) {
    int r = encoded_move - kSINGLE_START + 3;
    combination = Combination::kSingle;
    rank = r;
  } else if (encoded_move < kTRIPLE_START) {
    int r = encoded_move - kDOUBLE_START + 3;
    combination = Combination::kDouble;
    rank = r;
  } else if (encoded_move < kFULL_HOUSE_START) {
    int r = encoded_move - kTRIPLE_START + 3;
    combination = Combination::kTriple;
    rank = r;
  } else if (encoded_move < kBOMB_START) {
    int x = encoded_move - kFULL_HOUSE_START;
    int tripleRank = x / 11 + 3;
    int rem = x % 11;
    int aux = (rem < (tripleRank - 3)) ? (rem + 3) : (rem + 4);
    combination = Combination::kFullHouse;
    rank = tripleRank;
    auxiliary = aux;
  } else if (encoded_move < kSTRAIGHT5_START) {
    int x = encoded_move - kBOMB_START;
    int bombRank = x / 13 + 3;
    int rem = x % 13;
    int aux;
    if (rem == 0)
      aux = 0; // no extra card
    else if (rem + 2 < bombRank)
      aux = rem + 2;
    else
      aux = rem + 3;
    combination = Combination::kBomb;
    rank = bombRank;
    auxiliary = aux;
  } else if (encoded_move < kSTRAIGHT6_START) {
    int r = encoded_move - kSTRAIGHT5_START + 6;
    combination = Combination::kStraight5;
    rank = r;
  } else if (encoded_move < kSTRAIGHT7_START) {
    int r = encoded_move - kSTRAIGHT6_START + 7;
    combination = Combination::kStraight6;
    rank = r;
  } else if (encoded_move < kSTRAIGHT8_START) {
    int r = encoded_move - kSTRAIGHT7_START + 8;
    combination = Combination::kStraight7;
    rank = r;
  } else if (encoded_move < kSTRAIGHT9_START) {
    int r = encoded_move - kSTRAIGHT8_START + 9;
    combination = Combination::kStraight8;
    rank = r;
  } else if (encoded_move < kSTRAIGHT10_START) {
    int r = encoded_move - kSTRAIGHT9_START + 10;
    combination = Combination::kStraight9;
    rank = r;
  } else if (encoded_move < kSTRAIGHT11_START) {
    int r = encoded_move - kSTRAIGHT10_START + 11;
    combination = Combination::kStraight10;
    rank = r;
  } else if (encoded_move < kSTRAIGHT12_START) {
    int r = encoded_move - kSTRAIGHT11_START + 12;
    combination = Combination::kStraight11;
    rank = r;
  } else if (encoded_move < kSTRAIGHT13_START) {
    int r = encoded_move - kSTRAIGHT12_START + 13;
    combination = Combination::kStraight12;
    rank = r;
  } else if (encoded_move < kDOUBLESTRAIGHT2_START) {
    // There is only one legal straight of length 13.
    // (In Big 2 this corresponds to the unique 13-card straight 3,4,...,A,2,
    // which by convention has highest card 2.)
    combination = Combination::kStraight13;
    rank = 15;
  } else if (encoded_move < kDOUBLESTRAIGHT3_START) {
    int r = encoded_move - kDOUBLESTRAIGHT2_START + 4;
    combination = Combination::kDoubleStraight2;
    rank = r;
  } else if (encoded_move < kDOUBLESTRAIGHT4_START) {
    int r = encoded_move - kDOUBLESTRAIGHT3_START + 5;
    combination = Combination::kDoubleStraight3;
    rank = r;
  } else if (encoded_move < kDOUBLESTRAIGHT5_START) {
    int r = encoded_move - kDOUBLESTRAIGHT4_START + 6;
    combination = Combination::kDoubleStraight4;
    rank = r;
  } else if (encoded_move < kDOUBLESTRAIGHT6_START) {
    int r = encoded_move - kDOUBLESTRAIGHT5_START + 7;
    combination = Combination::kDoubleStraight5;
    rank = r;
  } else if (encoded_move < kDOUBLESTRAIGHT7_START) {
    int r = encoded_move - kDOUBLESTRAIGHT6_START + 8;
    combination = Combination::kDoubleStraight6;
    rank = r;
  } else if (encoded_move < kTRIPLESTRAIGHT2_START) {
    int r = encoded_move - kDOUBLESTRAIGHT7_START + 9;
    combination = Combination::kDoubleStraight7;
    rank = r;
  } else if (encoded_move < kTRIPLESTRAIGHT3_START) {
    int r = encoded_move - kTRIPLESTRAIGHT2_START + 4;
    combination = Combination::kTripleStraight2;
    rank = r;
  } else if (encoded_move < kTRIPLESTRAIGHT4_START) {
    int r = encoded_move - kTRIPLESTRAIGHT3_START + 5;
    combination = Combination::kTripleStraight3;
    rank = r;
  } else if (encoded_move < kTRIPLESTRAIGHT5_START) {
    int r = encoded_move - kTRIPLESTRAIGHT4_START + 6;
    combination = Combination::kTripleStraight4;
    rank = r;
  } else if (encoded_move < kDOUBLESTRAIGHT8_START) {
    int r = encoded_move - kTRIPLESTRAIGHT5_START + 7;
    combination = Combination::kTripleStraight5;
    rank = r;
  } else {
    int r = encoded_move - kDOUBLESTRAIGHT8_START + 10;
    combination = Combination::kDoubleStraight8;
    rank = r;
  }
}

int Move::numCards() const {

  if (combination == Combination::kPass) {
    return 0;
  }
  if (combination == Combination::kSingle) {
    return 1;
  }
  if (combination == Combination::kDouble) {
    return 2;
  }
  if (combination == Combination::kTriple) {
    return 3;
  }
  if (combination == Combination::kFullHouse) {
    return 5;
  }
  if (combination == Combination::kBomb) {
    return (rank == 14 ? 3 : 4) + (auxiliary == 0 ? 0 : 1);
  }
  if (combination == Combination::kStraight5) {
    return 5;
  }
  if (combination == Combination::kStraight6) {
    return 6;
  }
  if (combination == Combination::kStraight7) {
    return 7;
  }
  if (combination == Combination::kStraight8) {
    return 8;
  }
  if (combination == Combination::kStraight9) {
    return 9;
  }
  if (combination == Combination::kStraight10) {
    return 10;
  }
  if (combination == Combination::kStraight11) {
    return 11;
  }
  if (combination == Combination::kStraight12) {
    return 12;
  }
  if (combination == Combination::kStraight13) {
    return 13;
  }
  if (combination == Combination::kDoubleStraight2) {
    return 4;
  }
  if (combination == Combination::kDoubleStraight3) {
    return 6;
  }
  if (combination == Combination::kDoubleStraight4) {
    return 8;
  }
  if (combination == Combination::kDoubleStraight5) {
    return 10;
  }
  if (combination == Combination::kDoubleStraight6) {
    return 12;
  }
  if (combination == Combination::kDoubleStraight7) {
    return 14;
  }
  if (combination == Combination::kTripleStraight2) {
    return 6;
  }
  if (combination == Combination::kTripleStraight3) {
    return 9;
  }
  if (combination == Combination::kTripleStraight4) {
    return 12;
  }
  if (combination == Combination::kTripleStraight5) {
    return 15;
  }
  // Double straight 8
  return 16;
}

//-----------------------------------------------------------------
// encodeMove
//-----------------------------------------------------------------
int encodeMove(const Move &move) {
  switch (move.combination) {
  case Move::Combination::kPass:
    return kPASS;
  case Move::Combination::kSingle:
    return kSINGLE_START + (move.rank - 3);
  case Move::Combination::kDouble:
    return kDOUBLE_START + (move.rank - 3);
  case Move::Combination::kTriple:
    return kTRIPLE_START + (move.rank - 3);
  case Move::Combination::kFullHouse: {
    // In a full house the move.rank is the triple’s rank.
    // There are 11 possible second ranks (the pair), but note that if auxiliary
    // < rank we use one formula.
    int base = (move.rank - 3) * 11;
    int offset = (move.auxiliary < move.rank) ? (move.auxiliary - 3)
                                              : (move.auxiliary - 4);
    return kFULL_HOUSE_START + base + offset;
  }
  case Move::Combination::kBomb: {
    // For bombs the auxiliary card is optional.
    if (move.auxiliary == 0) {
      return kBOMB_START + (move.rank - 3) * 13;
    }
    int base = (move.rank - 3) * 13;
    int offset = (move.auxiliary < move.rank) ? (move.auxiliary - 2)
                                              : (move.auxiliary - 3);
    return kBOMB_START + base + offset;
  }
  case Move::Combination::kStraight5:
    return kSTRAIGHT5_START + (move.rank - 6);
  case Move::Combination::kStraight6:
    return kSTRAIGHT6_START + (move.rank - 7);
  case Move::Combination::kStraight7:
    return kSTRAIGHT7_START + (move.rank - 8);
  case Move::Combination::kStraight8:
    return kSTRAIGHT8_START + (move.rank - 9);
  case Move::Combination::kStraight9:
    return kSTRAIGHT9_START + (move.rank - 10);
  case Move::Combination::kStraight10:
    return kSTRAIGHT10_START + (move.rank - 11);
  case Move::Combination::kStraight11:
    return kSTRAIGHT11_START + (move.rank - 12);
  case Move::Combination::kStraight12:
    return kSTRAIGHT12_START + (move.rank - 13);
  case Move::Combination::kStraight13:
    return kSTRAIGHT13_START;
  case Move::Combination::kDoubleStraight2:
    return kDOUBLESTRAIGHT2_START + (move.rank - 4);
  case Move::Combination::kDoubleStraight3:
    return kDOUBLESTRAIGHT3_START + (move.rank - 5);
  case Move::Combination::kDoubleStraight4:
    return kDOUBLESTRAIGHT4_START + (move.rank - 6);
  case Move::Combination::kDoubleStraight5:
    return kDOUBLESTRAIGHT5_START + (move.rank - 7);
  case Move::Combination::kDoubleStraight6:
    return kDOUBLESTRAIGHT6_START + (move.rank - 8);
  case Move::Combination::kDoubleStraight7:
    return kDOUBLESTRAIGHT7_START + (move.rank - 9);
  case Move::Combination::kTripleStraight2:
    return kTRIPLESTRAIGHT2_START + (move.rank - 4);
  case Move::Combination::kTripleStraight3:
    return kTRIPLESTRAIGHT3_START + (move.rank - 5);
  case Move::Combination::kTripleStraight4:
    return kTRIPLESTRAIGHT4_START + (move.rank - 6);
  case Move::Combination::kTripleStraight5:
    return kTRIPLESTRAIGHT5_START + (move.rank - 7);
  case Move::Combination::kDoubleStraight8:
    return kDOUBLESTRAIGHT8_START + (move.rank - 10);
  default:
    throw std::invalid_argument("Invalid move combination in encodeMove");
  }
}

std::ostream &operator<<(std::ostream &os, const Move &move) {
  char rank_char = rankToChar(move.rank);
  char auxiliary_char = rankToChar(move.auxiliary);
  if (move.combination == Move::Combination::kPass) {
    os << "PASS";
  } else if (move.combination == Move::Combination::kSingle) {
    os << rank_char;
  } else if (move.combination == Move::Combination::kDouble) {
    os << rank_char << rank_char;
  } else if (move.combination == Move::Combination::kTriple) {
    os << rank_char << rank_char << rank_char;
  } else if (move.combination == Move::Combination::kFullHouse) {
    os << rank_char << rank_char << rank_char << auxiliary_char
       << auxiliary_char;
  } else if (move.combination == Move::Combination::kBomb) {
    if (move.rank == 14) {
      os << "AAA";
    } else {
      os << rank_char << rank_char << rank_char << rank_char;
    }
    if (move.auxiliary != 0) {
      os << auxiliary_char;
    }
  } else if (move.combination == Move::Combination::kStraight5) {
    for (int i = 4; i >= 0; i--) {
      os << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kStraight6) {
    for (int i = 5; i >= 0; i--) {
      os << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kStraight7) {
    for (int i = 6; i >= 0; i--) {
      os << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kStraight8) {
    for (int i = 7; i >= 0; i--) {
      os << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kStraight9) {
    for (int i = 8; i >= 0; i--) {
      os << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kStraight10) {
    for (int i = 9; i >= 0; i--) {
      os << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kStraight11) {
    for (int i = 10; i >= 0; i--) {
      os << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kStraight12) {
    for (int i = 11; i >= 0; i--) {
      os << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kStraight13) {
    os << "34567890JQKA2";
  } else if (move.combination == Move::Combination::kDoubleStraight2) {
    for (int i = 1; i >= 0; i--) {
      os << rankToChar(move.rank - i) << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kDoubleStraight3) {
    for (int i = 2; i >= 0; i--) {
      os << rankToChar(move.rank - i) << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kDoubleStraight4) {
    for (int i = 3; i >= 0; i--) {
      os << rankToChar(move.rank - i) << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kDoubleStraight5) {
    for (int i = 4; i >= 0; i--) {
      os << rankToChar(move.rank - i) << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kDoubleStraight6) {
    for (int i = 5; i >= 0; i--) {
      os << rankToChar(move.rank - i) << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kDoubleStraight7) {
    for (int i = 6; i >= 0; i--) {
      os << rankToChar(move.rank - i) << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kDoubleStraight8) {
    for (int i = 7; i >= 0; i--) {
      os << rankToChar(move.rank - i) << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kTripleStraight2) {
    for (int i = 1; i >= 0; i--) {
      os << rankToChar(move.rank - i) << rankToChar(move.rank - i)
         << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kTripleStraight3) {
    for (int i = 2; i >= 0; i--) {
      os << rankToChar(move.rank - i) << rankToChar(move.rank - i)
         << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kTripleStraight4) {
    for (int i = 3; i >= 0; i--) {
      os << rankToChar(move.rank - i) << rankToChar(move.rank - i)
         << rankToChar(move.rank - i);
    }
  } else if (move.combination == Move::Combination::kTripleStraight5) {
    for (int i = 4; i >= 0; i--) {
      os << rankToChar(move.rank - i) << rankToChar(move.rank - i)
         << rankToChar(move.rank - i);
    }
  }

  return os;
}