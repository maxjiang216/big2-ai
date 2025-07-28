#include "util.h"

#include <iostream>

using namespace std;

Sample::Sample(int player_turn, int perspective, const Game& game, const Move& move, const bitset<LEGAL_MOVES_SIZE>& legal_moves):
player_turn(player_turn), perspective(perspective), game(game), move(move), legal_moves(legal_moves) {}

void Sample::writeSample(float *stateBuffer, int *turnBuffer, int *legalMaskBuffer, float *valueBuffer, int *moveBuffer) const {
    // State is encoded as 26 floats
    // 1 for player turn
    stateBuffer[0] = (float)perspective;
    // 13 for player cards
    // Normalize between 0 and 1
    for (int i = 0; i < 12; i++) {
        stateBuffer[i + 1] = (float)game.getPlayerCardRank(i) / 4;
    }
    stateBuffer[12] = (float)game.getPlayerCardRank(11) / 3;
    stateBuffer[13] = (float)game.getPlayerCardRank(12);
    // Last combination type played
    for (int i = 0; i < 8; i++) {
        stateBuffer[14 + i] = 0;
    }

    if (move.combination == Move::Combination::kSingle) {
        stateBuffer[14] = 1;
    }
    else if (move.combination == Move::Combination::kDouble) {
        stateBuffer[15] = 1;
    }
    else if (move.combination == Move::Combination::kTriple) {
        stateBuffer[16] = 1;
    }
    else if (move.combination == Move::Combination::kFullHouse) {
        stateBuffer[17] = 1;
    }
    else if (move.combination == Move::Combination::kBomb) {
        stateBuffer[18] = 1;
    }
    else if (move.combination == Move::Combination::kStraight5 ||
             move.combination == Move::Combination::kStraight6 ||
             move.combination == Move::Combination::kStraight7 ||
             move.combination == Move::Combination::kStraight8 ||
             move.combination == Move::Combination::kStraight9 ||
             move.combination == Move::Combination::kStraight10 ||
             move.combination == Move::Combination::kStraight11 ||
             move.combination == Move::Combination::kStraight12 ||
             move.combination == Move::Combination::kStraight13) {
        stateBuffer[19] = 1;
    }
    else if (move.combination == Move::Combination::kDoubleStraight2 ||
             move.combination == Move::Combination::kDoubleStraight3 ||
             move.combination == Move::Combination::kDoubleStraight4 ||
             move.combination == Move::Combination::kDoubleStraight5 ||
             move.combination == Move::Combination::kDoubleStraight6 ||
             move.combination == Move::Combination::kDoubleStraight7) {
        stateBuffer[20] = 1;
    }
    else if (move.combination == Move::Combination::kTripleStraight2 ||
             move.combination == Move::Combination::kTripleStraight3 ||
             move.combination == Move::Combination::kTripleStraight4 ||
             move.combination == Move::Combination::kTripleStraight5) {
        stateBuffer[21] = 1;
    }
    // Last rank played
    stateBuffer[22] = 0;
    Move last_move = game.getLastMove();
    if (last_move.combination != Move::Combination::kPass) {
        stateBuffer[22] = ((float)last_move.rank - 3) / 12;
    }
    // Straight length
    // Normalize between 0 and 1
    stateBuffer[23] = 0;
    if (last_move.combination == Move::Combination::kDoubleStraight2 ||
        last_move.combination == Move::Combination::kTripleStraight2) {
        stateBuffer[23] = (2.0 - 1.0) / 12.0;
    }
    else if (last_move.combination == Move::Combination::kDoubleStraight3 ||
             last_move.combination == Move::Combination::kTripleStraight3) {
        stateBuffer[23] = (3.0 - 1.0) / 12.0;
    }
    else if (last_move.combination == Move::Combination::kDoubleStraight4 ||
             last_move.combination == Move::Combination::kTripleStraight4) {
        stateBuffer[23] = (4.0 - 1.0) / 12.0;
    }
    else if (last_move.combination == Move::Combination::kStraight5 ||
             last_move.combination == Move::Combination::kDoubleStraight5 ||
             last_move.combination == Move::Combination::kTripleStraight5) {
        stateBuffer[23] = (5.0 - 1.0) / 12.0;
    }
    else if (last_move.combination == Move::Combination::kStraight6 ||
             last_move.combination == Move::Combination::kDoubleStraight6) {
        stateBuffer[23] = (6.0 - 1.0) / 12.0;
    }
    else if (last_move.combination == Move::Combination::kStraight7 ||
             last_move.combination == Move::Combination::kDoubleStraight7) {
        stateBuffer[23] = (7.0 - 1.0) / 12.0;
    }
    else if (last_move.combination == Move::Combination::kStraight8) {
        stateBuffer[23] = (8.0 - 1.0) / 12.0;
    }
    else if (last_move.combination == Move::Combination::kStraight9) {
        stateBuffer[23] = (9.0 - 1.0) / 12.0;
    }
    else if (last_move.combination == Move::Combination::kStraight10) {
        stateBuffer[23] = (10.0 - 1.0) / 12.0;
    }
    else if (last_move.combination == Move::Combination::kStraight11) {
        stateBuffer[23] = (11.0 - 1.0) / 12.0;
    }
    else if (last_move.combination == Move::Combination::kStraight12) {
        stateBuffer[23] = (12.0 - 1.0) / 12.0;
    }
    else if (last_move.combination == Move::Combination::kStraight13) {
        stateBuffer[23] = (13.0 - 1.0) / 12.0;
    }

    // Auxiliary rank
    stateBuffer[24] = 0;
    if (last_move.combination == Move::Combination::kFullHouse ||
        last_move.combination == Move::Combination::kBomb) {
        stateBuffer[24] = ((float)last_move.auxiliary - 3) / 12;
    }

    // Opponent hand size
    stateBuffer[25] = (float)game.getOpponentCardNum() / 16;

    // Player Turn
    turnBuffer[0] = perspective;

    // Legal move mask
    for (int i = 0; i < NEURAL_NETWORK_MOVES_SIZE; i++) {
        legalMaskBuffer[i] = legal_moves[i];
    }

    // Value
    // Normalize between -1 and 1
    valueBuffer[0] = 2 * result - 1;

    // Move
    moveBuffer[0] = encodeMove(move);
}

void Sample::writeSampleComplex(float *stateBuffer, int *turnBuffer, int *legalMaskBuffer, float *valueBuffer, int *moveBuffer) const {
    // State is encoded as 26 floats
    // 1 for player turn
    stateBuffer[0] = (float)perspective;
    // 13 for player cards
    // Normalize between 0 and 1
    for (int i = 0; i < 12; i++) {
        stateBuffer[i + 1] = (float)game.getPlayerCardRank(i) / 4;
    }
    stateBuffer[12] = (float)game.getPlayerCardRank(11) / 3;
    stateBuffer[13] = (float)game.getPlayerCardRank(12);
    // Last combination type played
    for (int i = 0; i < 8; i++) {
        stateBuffer[14 + i] = 0;
    }

    if (move.combination == Move::Combination::kSingle) {
        stateBuffer[14] = 1;
    }
    else if (move.combination == Move::Combination::kDouble) {
        stateBuffer[15] = 1;
    }
    else if (move.combination == Move::Combination::kTriple) {
        stateBuffer[16] = 1;
    }
    else if (move.combination == Move::Combination::kFullHouse) {
        stateBuffer[17] = 1;
    }
    else if (move.combination == Move::Combination::kBomb) {
        stateBuffer[18] = 1;
    }
    else if (move.combination == Move::Combination::kStraight5) {
        stateBuffer[19] = 1;
    }
    else if (move.combination == Move::Combination::kStraight6) {
        stateBuffer[20] = 1;
    }
    else if (move.combination == Move::Combination::kStraight7) {
        stateBuffer[21] = 1;
    }
    else if (move.combination == Move::Combination::kStraight8) {
        stateBuffer[22] = 1;
    }
    else if (move.combination == Move::Combination::kStraight9) {
        stateBuffer[23] = 1;
    }
    else if (move.combination == Move::Combination::kStraight10) {
        stateBuffer[24] = 1;
    }
    else if (move.combination == Move::Combination::kStraight11) {
        stateBuffer[25] = 1;
    }
    else if (move.combination == Move::Combination::kStraight12) {
        stateBuffer[26] = 1;
    }
    else if (move.combination == Move::Combination::kStraight13) {
        stateBuffer[27] = 1;
    }
    else if (move.combination == Move::Combination::kDoubleStraight2) {
        stateBuffer[28] = 1;
    }
    else if (move.combination == Move::Combination::kDoubleStraight3) {
        stateBuffer[29] = 1;
    }
    else if (move.combination == Move::Combination::kDoubleStraight4) {
        stateBuffer[30] = 1;
    }
    else if (move.combination == Move::Combination::kDoubleStraight5) {
        stateBuffer[31] = 1;
    }
    else if (move.combination == Move::Combination::kDoubleStraight6) {
        stateBuffer[32] = 1;
    }
    else if (move.combination == Move::Combination::kDoubleStraight7) {
        stateBuffer[33] = 1;
    }
    else if (move.combination == Move::Combination::kTripleStraight2) {
        stateBuffer[34] = 1;
    }
    else if (move.combination == Move::Combination::kTripleStraight3) {
        stateBuffer[35] = 1;
    }
    else if (move.combination == Move::Combination::kTripleStraight4) {
        stateBuffer[36] = 1;
    }
    else if (move.combination == Move::Combination::kTripleStraight5) {
        stateBuffer[37] = 1;
    }

    // Last rank played
    stateBuffer[38] = 0;
    Move last_move = game.getLastMove();
    if (last_move.combination != Move::Combination::kPass) {
        stateBuffer[38] = ((float)last_move.rank - 3) / 12;
    }

    // Table cards
    for (int i = 0; i < 13; i++) {
        stateBuffer[39 + i] = 0;
    }
    for (int i = 0; i < 12; i++) {
        stateBuffer[39 + i] = (float)game.getTableCardRank(i) / 4;
    }
    stateBuffer[51] = (float)game.getTableCardRank(11) / 3;
    stateBuffer[52] = (float)game.getTableCardRank(12);

    // Player Turn
    turnBuffer[0] = perspective;

    // Legal move mask
    for (int i = 0; i < NEURAL_NETWORK_MOVES_SIZE; i++) {
        legalMaskBuffer[i] = legal_moves[i];
    }

    // Value
    // Normalize between -1 and 1
    valueBuffer[0] = 2 * result - 1;

    // Move
    moveBuffer[0] = encodeMove(move);
}

char rankToChar(int rank) {
    if (rank == 2 || rank == 15) {
        return '2';
    }
    if (rank < 10) {
        return '0' + rank;
    }
    if (rank == 10) {
        return '0';
    }
    if (rank == 11) {
        return 'J';
    }
    if (rank == 12) {
        return 'Q';
    }
    if (rank == 13) {
        return 'K';
    }
    return 'A';
}