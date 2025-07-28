#ifndef UTIL_H
#define UTIL_H

#include "game.h"
#include "move.h"

#include <bitset>

using std::bitset;

#define NEURAL_NETWORK_MOVES_SIZE 467
#define LEGAL_MOVES_SIZE 472

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
const int kPASS = 0;
const int kSINGLE_START = 1;      // 13 singles
const int kDOUBLE_START = kSINGLE_START + 13;  // 14, 12 doubles
const int kTRIPLE_START = kDOUBLE_START + 12;    // 26, 11 triples
const int kFULL_HOUSE_START = kTRIPLE_START + 11;  // 37, 132 full houses
const int kBOMB_START = kFULL_HOUSE_START + 132;   // 169, 156 bombs

const int kSTRAIGHT5_START = kBOMB_START + 156;     // 325, 10 straights of length 5 (6 to 2)
const int kSTRAIGHT6_START = kSTRAIGHT5_START + 10; // 335, 9 straights of length 6 (7 to 2)
const int kSTRAIGHT7_START = kSTRAIGHT6_START + 9; // 344, 8 straights of length 7 (8 to 2)
const int kSTRAIGHT8_START = kSTRAIGHT7_START + 8; // 352, 7 straights of length 8 (9 to 2)
const int kSTRAIGHT9_START = kSTRAIGHT8_START + 7; // 359, 6 straights of length 9 (10 to 2)
const int kSTRAIGHT10_START = kSTRAIGHT9_START + 6; // 365, 5 straights of length 10 (J to 2)
const int kSTRAIGHT11_START = kSTRAIGHT10_START + 5; // 370, 4 straights of length 11 (Q to 2)
const int kSTRAIGHT12_START = kSTRAIGHT11_START + 4; // 374, 3 straights of length 12 (K to 2)
const int kSTRAIGHT13_START = kSTRAIGHT12_START + 3; // 377, 1 straight of length 13 (2)

const int kDOUBLESTRAIGHT2_START = kSTRAIGHT13_START + 1; // 378, 11 double straights of length 2 (4 to A)
const int kDOUBLESTRAIGHT3_START = kDOUBLESTRAIGHT2_START + 11; // 389, 10 double straights of length 3 (5 to A)
const int kDOUBLESTRAIGHT4_START = kDOUBLESTRAIGHT3_START + 10; // 399, 9 double straights of length 4 (6 to A)
const int kDOUBLESTRAIGHT5_START = kDOUBLESTRAIGHT4_START + 9; // 408, 8 double straights of length 5 (7 to A)
const int kDOUBLESTRAIGHT6_START = kDOUBLESTRAIGHT5_START + 8; // 416, 7 double straights of length 6 (8 to A)
const int kDOUBLESTRAIGHT7_START = kDOUBLESTRAIGHT6_START + 7; // 423, 6 double straights of length 7 (9 to A)

const int kTRIPLESTRAIGHT2_START = kDOUBLESTRAIGHT7_START + 6; // 429, 11 triple straights of length 2 (4 to A)
const int kTRIPLESTRAIGHT3_START = kTRIPLESTRAIGHT2_START + 11; // 440, 10 triple straights of length 3 (5 to A)
const int kTRIPLESTRAIGHT4_START = kTRIPLESTRAIGHT3_START + 10; // 450, 9 triple straights of length 4 (6 to A)
const int kTRIPLESTRAIGHT5_START = kTRIPLESTRAIGHT4_START + 9; // 459, 8 triple straights of length 5 (7 to A)

// This is last as it is not in the neural network
const int kDOUBLESTRAIGHT8_START = kTRIPLESTRAIGHT5_START + 8; // 467, 5 double straights of length 8 (10 to A)

class Sample {
    public:
        Sample(int player_turn, int perspective, const Game& game, const Move& move, const bitset<LEGAL_MOVES_SIZE>& legal_moves);
        void writeSample(float *stateBuffer, int *turnBuffer, int *legalMaskBuffer, float *valueBuffer, int *moveBuffer) const;
        void writeSampleComplex(float *stateBuffer, int *turnBuffer, int *legalMaskBuffer, float *valueBuffer, int *moveBuffer) const;
        int result{0};
        // Player turn is used to deduce result after the game is over
        int player_turn{0};
        // Perspective is used to determine which head to use
        int perspective{0};
        Game game;
        Move move;
        bitset<LEGAL_MOVES_SIZE> legal_moves;
};

char rankToChar(int rank);

#endif