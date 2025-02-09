#include "move.h"
#include <stdexcept>

//-----------------------------------------------------------------
// Encoding constants – note these ranges were chosen so that:
//
//   PASS:              index 0
//   Singles:           13 moves, indices 1 .. 13
//   Doubles:           12 moves, indices 14 .. 25
//   Triples:           11 moves, indices 26 .. 36
//   Full Houses:       132 moves, indices 37 .. 168
//   Bombs:             156 moves, indices 169 .. 324
//   Straights:         53 moves, indices 325 .. 377
//   Sisters:           56 moves, indices 378 .. 433
//   Triple Straights:  38 moves, indices 434 .. 471
//-----------------------------------------------------------------
namespace {
    const int PASS = 0;
    const int SINGLE_START = 1;      // 13 singles
    const int DOUBLE_START = SINGLE_START + 13;  // 14, 12 doubles
    const int TRIPLE_START = DOUBLE_START + 12;    // 26, 11 triples
    const int FULL_HOUSE_START = TRIPLE_START + 11;  // 37, 132 full houses
    const int BOMB_START = FULL_HOUSE_START + 132;   // 169, 156 bombs

    const int STRAIGHT5_START = BOMB_START + 156;     // 325, 10 straights of length 5 (6 to 2)
    const int STRAIGHT6_START = STRAIGHT5_START + 10; // 335, 9 straights of length 6 (7 to 2)
    const int STRAIGHT7_START = STRAIGHT6_START + 9; // 344, 8 straights of length 7 (8 to 2)
    const int STRAIGHT8_START = STRAIGHT7_START + 8; // 352, 7 straights of length 8 (9 to 2)
    const int STRAIGHT9_START = STRAIGHT8_START + 7; // 359, 6 straights of length 9 (10 to 2)
    const int STRAIGHT10_START = STRAIGHT9_START + 6; // 365, 5 straights of length 10 (J to 2)
    const int STRAIGHT11_START = STRAIGHT10_START + 5; // 370, 4 straights of length 11 (Q to 2)
    const int STRAIGHT12_START = STRAIGHT11_START + 4; // 374, 3 straights of length 12 (K to 2)
    const int STRAIGHT13_START = STRAIGHT12_START + 3; // 377, 1 straight of length 13 (2)

    const int DOUBLESTRAIGHT2_START = STRAIGHT13_START + 1; // 378, 11 double straights of length 2 (4 to A)
    const int DOUBLESTRAIGHT3_START = DOUBLESTRAIGHT2_START + 11; // 389, 10 double straights of length 3 (5 to A)
    const int DOUBLESTRAIGHT4_START = DOUBLESTRAIGHT3_START + 10; // 399, 9 double straights of length 4 (6 to A)
    const int DOUBLESTRAIGHT5_START = DOUBLESTRAIGHT4_START + 9; // 408, 8 double straights of length 5 (7 to A)
    const int DOUBLESTRAIGHT6_START = DOUBLESTRAIGHT5_START + 8; // 416, 7 double straights of length 6 (8 to A)
    const int DOUBLESTRAIGHT7_START = DOUBLESTRAIGHT6_START + 7; // 423, 6 double straights of length 7 (9 to A)
    const int DOUBLESTRAIGHT8_START = DOUBLESTRAIGHT7_START + 6; // 429, 5 double straights of length 8 (10 to A)

    const int TRIPLESTRAIGHT2_START = DOUBLESTRAIGHT8_START + 5; // 434, 11 triple straights of length 2 (4 to A)
    const int TRIPLESTRAIGHT3_START = TRIPLESTRAIGHT2_START + 11; // 445, 10 triple straights of length 3 (5 to A)
    const int TRIPLESTRAIGHT4_START = TRIPLESTRAIGHT3_START + 10; // 455, 9 triple straights of length 4 (6 to A)
    const int TRIPLESTRAIGHT5_START = TRIPLESTRAIGHT4_START + 9; // 464, 8 triple straights of length 5 (7 to A)
} // end anonymous namespace

//-----------------------------------------------------------------
// encodeMove
//-----------------------------------------------------------------
int encodeMove(const Move &move) {
    switch (move.combination) {
        case Move::Combination::kPass:
            return PASS;
        case Move::Combination::kSingle:
            return SINGLE_START + (move.rank - 3);
        case Move::Combination::kDouble:
            return DOUBLE_START + (move.rank - 3);
        case Move::Combination::kTriple:
            return TRIPLE_START + (move.rank - 3);
        case Move::Combination::kFullHouse: {
            // In a full house the move.rank is the triple’s rank.
            // There are 11 possible second ranks (the pair), but note that if auxiliary < rank we use one formula.
            int base = (move.rank - 3) * 11;
            int offset = (move.auxiliary < move.rank) ? (move.auxiliary - 3) : (move.auxiliary - 4);
            return FULL_HOUSE_START + base + offset;
        }
        case Move::Combination::kBomb: {
            // For bombs the auxiliary card is optional.
            if (move.auxiliary == 0) {
                return BOMB_START + (move.rank - 3) * 13;
            }
            int base = (move.rank - 3) * 13;
            int offset = (move.auxiliary < move.rank) ? (move.auxiliary - 2) : (move.auxiliary - 3);
            return BOMB_START + base + offset;
        }
        case Move::Combination::kStraight5:
            return STRAIGHT5_START + (move.rank - 6);
        case Move::Combination::kStraight6:
            return STRAIGHT6_START + (move.rank - 7);
        case Move::Combination::kStraight7:
            return STRAIGHT7_START + (move.rank - 8);
        case Move::Combination::kStraight8:
            return STRAIGHT8_START + (move.rank - 9);
        case Move::Combination::kStraight9:
            return STRAIGHT9_START + (move.rank - 10);
        case Move::Combination::kStraight10:
            return STRAIGHT10_START + (move.rank - 11);
        case Move::Combination::kStraight11:
            return STRAIGHT11_START + (move.rank - 12);
        case Move::Combination::kStraight12:
            return STRAIGHT12_START + (move.rank - 13);
        case Move::Combination::kStraight13:
            return STRAIGHT13_START;
        case Move::Combination::kDoubleStraight2:
            return DOUBLESTRAIGHT2_START + (move.rank - 4);
        case Move::Combination::kDoubleStraight3:
            return DOUBLESTRAIGHT3_START + (move.rank - 5);
        case Move::Combination::kDoubleStraight4:
            return DOUBLESTRAIGHT4_START + (move.rank - 6);
        case Move::Combination::kDoubleStraight5:
            return DOUBLESTRAIGHT5_START + (move.rank - 7);
        case Move::Combination::kDoubleStraight6:
            return DOUBLESTRAIGHT6_START + (move.rank - 8);
        case Move::Combination::kDoubleStraight7:
            return DOUBLESTRAIGHT7_START + (move.rank - 9);
        case Move::Combination::kDoubleStraight8:
            return DOUBLESTRAIGHT8_START + (move.rank - 10);
        case Move::Combination::kTripleStraight2:
            return TRIPLESTRAIGHT2_START + (move.rank - 4);
        case Move::Combination::kTripleStraight3:
            return TRIPLESTRAIGHT3_START + (move.rank - 5);
        case Move::Combination::kTripleStraight4:
            return TRIPLESTRAIGHT4_START + (move.rank - 6);
        case Move::Combination::kTripleStraight5:
            return TRIPLESTRAIGHT5_START + (move.rank - 7);
        default:
            throw std::invalid_argument("Invalid move combination in encodeMove");
    }
}

//-----------------------------------------------------------------
// decodeMove
//-----------------------------------------------------------------
Move decodeMove(int encoded_move) {
    if (encoded_move == PASS) {
        return Move(Move::Combination::kPass);
    }
    if (encoded_move < DOUBLE_START) {
        int rank = encoded_move - SINGLE_START + 3;
        return Move(Move::Combination::kSingle, rank);
    }
    if (encoded_move < TRIPLE_START) {
        int rank = encoded_move - DOUBLE_START + 3;
        return Move(Move::Combination::kDouble, rank);
    }
    if (encoded_move < FULL_HOUSE_START) {
        int rank = encoded_move - TRIPLE_START + 3;
        return Move(Move::Combination::kTriple, rank);
    }
    if (encoded_move < BOMB_START) {
        int x = encoded_move - FULL_HOUSE_START;
        int tripleRank = x / 11 + 3;
        int rem = x % 11;
        int aux = (rem < (tripleRank - 3)) ? (rem + 3) : (rem + 4);
        return Move(Move::Combination::kFullHouse, tripleRank, aux);
    }
    if (encoded_move < STRAIGHT5_START) {
        int x = encoded_move - BOMB_START;
        int bombRank = x / 13 + 3;
        int rem = x % 13;
        int aux;
        if (rem == 0)
            aux = 0;  // no extra card
        else if (rem + 2 < bombRank)
            aux = rem + 2;
        else
            aux = rem + 3;
        return Move(Move::Combination::kBomb, bombRank, aux);
    }
    if (encoded_move < STRAIGHT6_START) {
        int rank = encoded_move - STRAIGHT5_START + 6;
        return Move(Move::Combination::kStraight5, rank);
    }
    if (encoded_move < STRAIGHT7_START) {
        int rank = encoded_move - STRAIGHT6_START + 7;
        return Move(Move::Combination::kStraight6, rank);
    }
    if (encoded_move < STRAIGHT8_START) {
        int rank = encoded_move - STRAIGHT7_START + 8;
        return Move(Move::Combination::kStraight7, rank);
    }
    if (encoded_move < STRAIGHT9_START) {
        int rank = encoded_move - STRAIGHT8_START + 9;
        return Move(Move::Combination::kStraight8, rank);
    }
    if (encoded_move < STRAIGHT10_START) {
        int rank = encoded_move - STRAIGHT9_START + 10;
        return Move(Move::Combination::kStraight9, rank);
    }
    if (encoded_move < STRAIGHT11_START) {
        int rank = encoded_move - STRAIGHT10_START + 11;
        return Move(Move::Combination::kStraight10, rank);
    }
    if (encoded_move < STRAIGHT12_START) {
        int rank = encoded_move - STRAIGHT11_START + 12;
        return Move(Move::Combination::kStraight11, rank);
    }
    if (encoded_move < STRAIGHT13_START) {
        int rank = encoded_move - STRAIGHT12_START + 13;
        return Move(Move::Combination::kStraight12, rank);
    }
    if (encoded_move < DOUBLESTRAIGHT2_START) {
        // There is only one legal straight of length 13.
        // (In Big 2 this corresponds to the unique 13-card straight 3,4,...,A,2,
        // which by convention has highest card 2.)
        return Move(Move::Combination::kStraight13, 2);
    }
    if (encoded_move < DOUBLESTRAIGHT3_START) {
        int rank = encoded_move - DOUBLESTRAIGHT2_START + 4;
        return Move(Move::Combination::kDoubleStraight2, rank);
    }
    if (encoded_move < DOUBLESTRAIGHT4_START) {
        int rank = encoded_move - DOUBLESTRAIGHT3_START + 5;
        return Move(Move::Combination::kDoubleStraight3, rank);
    }
    if (encoded_move < DOUBLESTRAIGHT5_START) {
        int rank = encoded_move - DOUBLESTRAIGHT4_START + 6;
        return Move(Move::Combination::kDoubleStraight4, rank);
    }
    if (encoded_move < DOUBLESTRAIGHT6_START) {
        int rank = encoded_move - DOUBLESTRAIGHT5_START + 7;
        return Move(Move::Combination::kDoubleStraight5, rank);
    }
    if (encoded_move < DOUBLESTRAIGHT7_START) {
        int rank = encoded_move - DOUBLESTRAIGHT6_START + 8;
        return Move(Move::Combination::kDoubleStraight6, rank);
    }
    if (encoded_move < DOUBLESTRAIGHT8_START) {
        int rank = encoded_move - DOUBLESTRAIGHT7_START + 9;
        return Move(Move::Combination::kDoubleStraight7, rank);
    }
    if (encoded_move < TRIPLESTRAIGHT2_START) {
        int rank = encoded_move - DOUBLESTRAIGHT8_START + 10;
        return Move(Move::Combination::kDoubleStraight8, rank);
    }
    if (encoded_move < TRIPLESTRAIGHT3_START) {
        int rank = encoded_move - TRIPLESTRAIGHT2_START + 4;
        return Move(Move::Combination::kTripleStraight2, rank);
    }
    if (encoded_move < TRIPLESTRAIGHT4_START) {
        int rank = encoded_move - TRIPLESTRAIGHT3_START + 5;
        return Move(Move::Combination::kTripleStraight3, rank);
    }
    if (encoded_move < TRIPLESTRAIGHT5_START) {
        int rank = encoded_move - TRIPLESTRAIGHT4_START + 6;
        return Move(Move::Combination::kTripleStraight4, rank);
    }
    if (encoded_move < TRIPLESTRAIGHT5_START + 8) {
        int rank = encoded_move - TRIPLESTRAIGHT5_START + 7;
        return Move(Move::Combination::kTripleStraight5, rank);
    }
    throw std::invalid_argument("Encoded move out of range in decodeMove");
}
