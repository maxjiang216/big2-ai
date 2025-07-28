#include "game.h"
#include "util.h"
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

#include <iostream>

using namespace std;

Game::Game(bool player_turn, int player_cards[16]): player_turn(player_turn) {
    for (int i = 0; i < 13; ++i) {
        this->player_cards[i] = 0;
    }
    for (int i = 0; i < 16; ++i) {
        ++this->player_cards[player_cards[i] - 3];
    }
    opponent_card_num = 16;
    for (int i = 0; i < 13; ++i) {
        table_cards[i] = 0;
    }
}

int Game::getPlayerTurn() const {
    return player_turn;
}

int Game::getPlayerCardRank(int i) const {
    return player_cards[i];
}

int Game::getTableCardRank(int i) const {
    return table_cards[i];
}

int Game::getOpponentCardNum() const {
    return opponent_card_num;
}

Move Game::getLastMove() const {
    return last_move;
}

void Game::doMove(Move move) {

    // Single
    if (move.combination == Move::Combination::kSingle) {
        if (player_turn) {
            player_cards[move.rank - 3]--;
        } else {
            opponent_card_num--;
        }
        table_cards[move.rank - 3]++;
    }

    // Double
    if (move.combination == Move::Combination::kDouble) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 2;
        } else {
            opponent_card_num -= 2;
        }
        table_cards[move.rank - 3] += 2;
    }

    // Triple
    if (move.combination == Move::Combination::kTriple) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 3;
        } else {
            opponent_card_num -= 3;
        }
        table_cards[move.rank - 3] += 3;
    }

    // Full house
    if (move.combination == Move::Combination::kFullHouse) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 3;
            player_cards[move.auxiliary - 3] -= 2;
        } else {
            opponent_card_num -= 5;
        }
        table_cards[move.rank - 3] += 3;
        table_cards[move.auxiliary - 3] += 2;
    }

    // Bomb
    if (move.combination == Move::Combination::kBomb) {
        if (player_turn) {
            player_cards[move.rank - 3] = 0;
        } else {
            // Ace bomb has 3 cards
            if (move.rank == 14) {
                opponent_card_num -= 3;
            } else {
                opponent_card_num -= 4;
            }
        }
        if (move.rank == 14) {
            table_cards[move.rank - 3] = 3;
        } else {
            table_cards[move.rank - 3] = 4;
        }
        if (move.auxiliary != 0) {
            if (player_turn) {
                player_cards[move.auxiliary - 3] -= 1;
            } else {
                opponent_card_num -= 1;
            }
            table_cards[move.auxiliary - 3] += 1;
        }
    }

    // Straight of length 5
    if (move.combination == Move::Combination::kStraight5) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 1;
            player_cards[move.rank - 4] -= 1;
            player_cards[move.rank - 5] -= 1;
            player_cards[move.rank - 6] -= 1;
            player_cards[(move.rank + 6) % 13] -= 1;
        } else {
            opponent_card_num -= 5;
        }
        table_cards[move.rank - 3] += 1;
        table_cards[move.rank - 4] += 1;
        table_cards[move.rank - 5] += 1;
        table_cards[move.rank - 6] += 1;
        table_cards[(move.rank + 6) % 13] += 1;
    }

    // Straight of length 6
    if (move.combination == Move::Combination::kStraight6) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 1;
            player_cards[move.rank - 4] -= 1;
            player_cards[move.rank - 5] -= 1;
            player_cards[move.rank - 6] -= 1;
            player_cards[move.rank - 7] -= 1;
            player_cards[(move.rank + 5) % 13] -= 1;
        } else {
            opponent_card_num -= 6;
        }
        table_cards[move.rank - 3] += 1;
        table_cards[move.rank - 4] += 1;
        table_cards[move.rank - 5] += 1;
        table_cards[move.rank - 6] += 1;
        table_cards[move.rank - 7] += 1;
        table_cards[(move.rank + 5) % 13] += 1;
    }

    // Straight of length 7
    if (move.combination == Move::Combination::kStraight7) {
        if (player_turn) {  
            player_cards[move.rank - 3] -= 1;
            player_cards[move.rank - 4] -= 1;
            player_cards[move.rank - 5] -= 1;
            player_cards[move.rank - 6] -= 1;
            player_cards[move.rank - 7] -= 1;
            player_cards[move.rank - 8] -= 1;
            player_cards[(move.rank + 4) % 13] -= 1;
        } else {
            opponent_card_num -= 7;
        }
        table_cards[move.rank - 3] += 1;
        table_cards[move.rank - 4] += 1;
        table_cards[move.rank - 5] += 1;
        table_cards[move.rank - 6] += 1;
        table_cards[move.rank - 7] += 1;
        table_cards[move.rank - 8] += 1;
        table_cards[(move.rank + 4) % 13] += 1;
    }

    // Straight of length 8
    if (move.combination == Move::Combination::kStraight8) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 1;
            player_cards[move.rank - 4] -= 1;
            player_cards[move.rank - 5] -= 1;
            player_cards[move.rank - 6] -= 1;
            player_cards[move.rank - 7] -= 1;
            player_cards[move.rank - 8] -= 1;
            player_cards[move.rank - 9] -= 1;
            player_cards[(move.rank + 3) % 13] -= 1;
        } else {
            opponent_card_num -= 8;
        }
        table_cards[move.rank - 3] += 1;
        table_cards[move.rank - 4] += 1;
        table_cards[move.rank - 5] += 1;
        table_cards[move.rank - 6] += 1;
        table_cards[move.rank - 7] += 1;
        table_cards[move.rank - 8] += 1;
        table_cards[move.rank - 9] += 1;
        table_cards[(move.rank + 3) % 13] += 1;
    }

    // Straight of length 9
    if (move.combination == Move::Combination::kStraight9) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 1;
            player_cards[move.rank - 4] -= 1;
            player_cards[move.rank - 5] -= 1;
            player_cards[move.rank - 6] -= 1;
            player_cards[move.rank - 7] -= 1;
            player_cards[move.rank - 8] -= 1;
            player_cards[move.rank - 9] -= 1;
            player_cards[move.rank - 10] -= 1;
            player_cards[(move.rank + 2) % 13] -= 1;
        } else {
            opponent_card_num -= 9;
        }
        table_cards[move.rank - 3] += 1;
        table_cards[move.rank - 4] += 1;
        table_cards[move.rank - 5] += 1;
        table_cards[move.rank - 6] += 1;
        table_cards[move.rank - 7] += 1;
        table_cards[move.rank - 8] += 1;
        table_cards[move.rank - 9] += 1;
        table_cards[move.rank - 10] += 1;
        table_cards[(move.rank + 2) % 13] += 1;
    }

    // Straight of length 10
    if (move.combination == Move::Combination::kStraight10) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 1;
            player_cards[move.rank - 4] -= 1;
            player_cards[move.rank - 5] -= 1;
            player_cards[move.rank - 6] -= 1;
            player_cards[move.rank - 7] -= 1;
            player_cards[move.rank - 8] -= 1;
            player_cards[move.rank - 9] -= 1;
            player_cards[move.rank - 10] -= 1;
            player_cards[move.rank - 11] -= 1;
            player_cards[(move.rank + 1) % 13] -= 1;
        } else {
            opponent_card_num -= 10;
        }
        table_cards[move.rank - 3] += 1;
        table_cards[move.rank - 4] += 1;
        table_cards[move.rank - 5] += 1;
        table_cards[move.rank - 6] += 1;
        table_cards[move.rank - 7] += 1;
        table_cards[move.rank - 8] += 1;
        table_cards[move.rank - 9] += 1;
        table_cards[move.rank - 10] += 1;
        table_cards[move.rank - 11] += 1;
        table_cards[(move.rank + 1) % 13] += 1;
    }

    // Straight of length 11
    if (move.combination == Move::Combination::kStraight11) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 1;
            player_cards[move.rank - 4] -= 1;
            player_cards[move.rank - 5] -= 1;
            player_cards[move.rank - 6] -= 1;
            player_cards[move.rank - 7] -= 1;
            player_cards[move.rank - 8] -= 1;
            player_cards[move.rank - 9] -= 1;
            player_cards[move.rank - 10] -= 1;
            player_cards[move.rank - 11] -= 1;
            player_cards[move.rank - 12] -= 1;
            player_cards[move.rank % 13] -= 1;
        } else {
            opponent_card_num -= 11;
        }
        table_cards[move.rank - 3] += 1;
        table_cards[move.rank - 4] += 1;
        table_cards[move.rank - 5] += 1;
        table_cards[move.rank - 6] += 1;
        table_cards[move.rank - 7] += 1;
        table_cards[move.rank - 8] += 1;
        table_cards[move.rank - 9] += 1;
        table_cards[move.rank - 10] += 1;
        table_cards[move.rank - 11] += 1;
        table_cards[move.rank - 12] += 1;
        table_cards[move.rank % 13] += 1;
    }

    // Straight of length 12
    if (move.combination == Move::Combination::kStraight12) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 1;
            player_cards[move.rank - 4] -= 1;
            player_cards[move.rank - 5] -= 1;
            player_cards[move.rank - 6] -= 1;
            player_cards[move.rank - 7] -= 1;
            player_cards[move.rank - 8] -= 1;
            player_cards[move.rank - 9] -= 1;
            player_cards[move.rank - 10] -= 1;
            player_cards[move.rank - 11] -= 1;
            player_cards[move.rank - 12] -= 1;
            player_cards[move.rank - 13] -= 1;
            player_cards[(move.rank - 1) % 13] -= 1;
        } else {
            opponent_card_num -= 12;
        }
        table_cards[move.rank - 3] += 1;
        table_cards[move.rank - 4] += 1;
        table_cards[move.rank - 5] += 1;
        table_cards[move.rank - 6] += 1;
        table_cards[move.rank - 7] += 1;
        table_cards[move.rank - 8] += 1;
        table_cards[move.rank - 9] += 1;
        table_cards[move.rank - 10] += 1;
        table_cards[move.rank - 11] += 1;
        table_cards[move.rank - 12] += 1;
        table_cards[move.rank - 13] += 1;
        table_cards[(move.rank - 1) % 13] += 1;
    }

    // Straight of length 13
    if (move.combination == Move::Combination::kStraight13) {
        if (player_turn) {
            for (int i = 0; i < 13; ++i) {
                player_cards[i] -= 1;
                table_cards[i] += 1;
            }
        } else {
            opponent_card_num -= 13;
            for (int i = 0; i < 13; ++i) {
                table_cards[i] += 1;
            }
        }
    }
        
    // Sisters of length 2
    if (move.combination == Move::Combination::kDoubleStraight2) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 2;
            player_cards[move.rank - 4] -= 2;
        } else {
            opponent_card_num -= 4;
        }
        table_cards[move.rank - 3] += 2;
        table_cards[move.rank - 4] += 2;
    }

    // Sisters of length 3
    if (move.combination == Move::Combination::kDoubleStraight3) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 2;
            player_cards[move.rank - 4] -= 2;
            player_cards[move.rank - 5] -= 2;
        } else {
            opponent_card_num -= 6;
        }
        table_cards[move.rank - 3] += 2;
        table_cards[move.rank - 4] += 2;
        table_cards[move.rank - 5] += 2;
    }

    // Sisters of length 4
    if (move.combination == Move::Combination::kDoubleStraight4) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 2;
            player_cards[move.rank - 4] -= 2;
            player_cards[move.rank - 5] -= 2;
            player_cards[move.rank - 6] -= 2;
        } else {
            opponent_card_num -= 8;
        }
        table_cards[move.rank - 3] += 2;
        table_cards[move.rank - 4] += 2;
        table_cards[move.rank - 5] += 2;
        table_cards[move.rank - 6] += 2;
    }

    // Sisters of length 5
    if (move.combination == Move::Combination::kDoubleStraight5) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 2;
            player_cards[move.rank - 4] -= 2;
            player_cards[move.rank - 5] -= 2;
            player_cards[move.rank - 6] -= 2;
            player_cards[move.rank - 7] -= 2;
        } else {
            opponent_card_num -= 10;
        }
        table_cards[move.rank - 3] += 2;
        table_cards[move.rank - 4] += 2;
        table_cards[move.rank - 5] += 2;
        table_cards[move.rank - 6] += 2;
        table_cards[move.rank - 7] += 2;
    }

    // Sisters of length 6
    if (move.combination == Move::Combination::kDoubleStraight6) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 2;
            player_cards[move.rank - 4] -= 2;
            player_cards[move.rank - 5] -= 2;
            player_cards[move.rank - 6] -= 2;
            player_cards[move.rank - 7] -= 2;
            player_cards[move.rank - 8] -= 2;
        } else {
            opponent_card_num -= 12;
        }
        table_cards[move.rank - 3] += 2;
        table_cards[move.rank - 4] += 2;
        table_cards[move.rank - 5] += 2;
        table_cards[move.rank - 6] += 2;
        table_cards[move.rank - 7] += 2;
    }

    // Sisters of length 7
    if (move.combination == Move::Combination::kDoubleStraight7) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 2;
            player_cards[move.rank - 4] -= 2;
            player_cards[move.rank - 5] -= 2;
            player_cards[move.rank - 6] -= 2;
            player_cards[move.rank - 7] -= 2;
            player_cards[move.rank - 8] -= 2;
            player_cards[move.rank - 9] -= 2;
        } else {
            opponent_card_num -= 14;
        }
        table_cards[move.rank - 3] += 2;
        table_cards[move.rank - 4] += 2;
        table_cards[move.rank - 5] += 2;
        table_cards[move.rank - 6] += 2;
        table_cards[move.rank - 7] += 2;
        table_cards[move.rank - 8] += 2;
        table_cards[move.rank - 9] += 2;
    }

    // Sisters of length 8
    if (move.combination == Move::Combination::kDoubleStraight8) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 2;
            player_cards[move.rank - 4] -= 2;
            player_cards[move.rank - 5] -= 2;
            player_cards[move.rank - 6] -= 2;
            player_cards[move.rank - 7] -= 2;
            player_cards[move.rank - 8] -= 2;
            player_cards[move.rank - 9] -= 2;
            player_cards[move.rank - 10] -= 2;
        } else {
            opponent_card_num = 0;
        }
        table_cards[move.rank - 3] += 2;
        table_cards[move.rank - 4] += 2;
        table_cards[move.rank - 5] += 2;
        table_cards[move.rank - 6] += 2;
        table_cards[move.rank - 7] += 2;
        table_cards[move.rank - 8] += 2;
        table_cards[move.rank - 9] += 2;
        table_cards[move.rank - 10] += 2;
    }

    // Triple straights of length 2
    if (move.combination == Move::Combination::kTripleStraight2) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 3;
            player_cards[move.rank - 4] -= 3;
        } else {
            opponent_card_num -= 6;
        }
        table_cards[move.rank - 3] += 3;
        table_cards[move.rank - 4] += 3;
    }

    // Triple straights of length 3
    if (move.combination == Move::Combination::kTripleStraight3) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 3;
            player_cards[move.rank - 4] -= 3;
            player_cards[move.rank - 5] -= 3;
        } else {
            opponent_card_num -= 9;
        }
        table_cards[move.rank - 3] += 3;
        table_cards[move.rank - 4] += 3;
        table_cards[move.rank - 5] += 3;
    }

    // Triple straights of length 4
    if (move.combination == Move::Combination::kTripleStraight4) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 3;
            player_cards[move.rank - 4] -= 3;
            player_cards[move.rank - 5] -= 3;
            player_cards[move.rank - 6] -= 3;
        } else {
            opponent_card_num -= 12;
        }
        table_cards[move.rank - 3] += 3;
        table_cards[move.rank - 4] += 3;
        table_cards[move.rank - 5] += 3;
        table_cards[move.rank - 6] += 3;
    }

    // Triple straights of length 5
    if (move.combination == Move::Combination::kTripleStraight5) {
        if (player_turn) {
            player_cards[move.rank - 3] -= 3;
            player_cards[move.rank - 4] -= 3;
            player_cards[move.rank - 5] -= 3;
            player_cards[move.rank - 6] -= 3;
            player_cards[move.rank - 7] -= 3;
        } else {
            opponent_card_num -= 15;
        }
        table_cards[move.rank - 3] += 3;
        table_cards[move.rank - 4] += 3;
        table_cards[move.rank - 5] += 3;
        table_cards[move.rank - 6] += 3;
        table_cards[move.rank - 7] += 3;
    }

    // Switch turn
    player_turn = !player_turn;
    last_move = move;
    
}

vector<int> Game::getLegalMoves() const {

    vector<int> legal_moves;

    // Pass as last move means new trick
    // Cannot pass for new trick
    if (last_move.combination != Move::Combination::kPass) {
        legal_moves.push_back(0);
    }

    // If it is the player's turn
    // We output all legal moves that can be played by the player
    if (player_turn) {    

        // Singles
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kSingle) {
            int start = last_move.combination == Move::Combination::kPass ? 3 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (player_cards[i - 3] > 0) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kSingle, i)));
                }
            }
        }

        // Doubles
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDouble) {
            int start = last_move.combination == Move::Combination::kPass ? 3 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (player_cards[i - 3] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDouble, i)));
                }
            }
        }

        // Triples
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kTriple) {
            int start = last_move.combination == Move::Combination::kPass ? 3 : last_move.rank + 1;
            for (int i = start; i <= 13; ++i) {
                if (player_cards[i - 3] >= 3) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kTriple, i)));
                }
            }
        }

        // Full houses
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kFullHouse) {
            int start = last_move.combination == Move::Combination::kPass ? 3 : last_move.rank + 1;
            for (int i = start; i <= 13; ++i) {
                if (player_cards[i - 3] >= 3) {
                    for (int j = 3; j <= 14; ++j) {
                        if (i != j && player_cards[j - 3] >= 2) {
                            legal_moves.push_back(encodeMove(Move(Move::Combination::kFullHouse, i, j)));
                        }
                    }
                }
            }
        }

        // Bombs
        if (last_move.combination == Move::Combination::kBomb) {
            for (int i = last_move.rank + 1; i <= 14; ++i) {
                if (player_cards[i - 3] == 4 || (i == 14 && player_cards[i - 3] == 3)) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kBomb, i)));
                    for (int j = 3; j <= 15; ++j) {
                        if (i != j && player_cards[j - 3] >= 1) {
                            legal_moves.push_back(encodeMove(Move(Move::Combination::kBomb, i, j)));
                        }
                    }
                }
            }
        }
        // Use bombs as burn
        else {
            for (int i = 3; i <= 14; ++i) {
                if (player_cards[i - 3] == 4 || (i == 14 && player_cards[i - 3] == 3)) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kBomb, i)));
                    for (int j = 3; j <= 15; ++j) {
                        if (i != j && player_cards[j - 3] >= 1) {
                            legal_moves.push_back(encodeMove(Move(Move::Combination::kBomb, i, j)));
                        }
                    }
                }
            }
        }

        // Straight of length 5
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight5) {
            int start = last_move.combination == Move::Combination::kPass ? 6 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[(i + 6) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight5, i)));
                }
            }
        }

        // Straight of length 6
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight6) {
            int start = last_move.combination == Move::Combination::kPass ? 7 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
                player_cards[(i + 5) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight6, i)));
                }
            }
        }

        // Straight of length 7
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight7) {
            int start = last_move.combination == Move::Combination::kPass ? 8 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
                player_cards[i - 8] >= 1 && player_cards[(i + 4) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight7, i)));
                }
            }
        }

        // Straight of length 8
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight8) {
            int start = last_move.combination == Move::Combination::kPass ? 9 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
                player_cards[i - 8] >= 1 && player_cards[i - 9] >= 1 && player_cards[(i + 3) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight8, i)));
                }
            }
        }

        // Straight of length 9
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight9) {
            int start = last_move.combination == Move::Combination::kPass ? 10 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
                player_cards[i - 8] >= 1 && player_cards[i - 9] >= 1 && player_cards[i - 10] >= 1 && player_cards[(i + 2) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight9, i)));
                }
            }
        }

        // Straight of length 10
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight10) {
            int start = last_move.combination == Move::Combination::kPass ? 11 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
                player_cards[i - 8] >= 1 && player_cards[i - 9] >= 1 && player_cards[i - 10] >= 1 && player_cards[i - 11] >= 1 && player_cards[(i + 1) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight10, i)));
                }
            }
        }

        // Straight of length 11
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight11) {
            int start = last_move.combination == Move::Combination::kPass ? 12 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
                player_cards[i - 8] >= 1 && player_cards[i - 9] >= 1 && player_cards[i - 10] >= 1 && player_cards[i - 11] >= 1 && player_cards[i - 12] >= 1 &&
                player_cards[i % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight11, i)));
                }
            }
        }

        // Straight of length 12
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight12) {
            int start = last_move.combination == Move::Combination::kPass ? 13 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
                player_cards[i - 8] >= 1 && player_cards[i - 9] >= 1 && player_cards[i - 10] >= 1 && player_cards[i - 11] >= 1 && player_cards[i - 12] >= 1 &&
                player_cards[i - 13] >= 1 && player_cards[(i - 1) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight12, i)));
                }
            }
        }

        // Straight of length 13
        if (last_move.combination == Move::Combination::kPass) {
            bool possible = true;
            for (int i = 0; i < 13; ++i) {
                if (player_cards[i] == 0) {
                    possible = false;
                    break;
                }
            }
            if (possible) {
                legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight13, 15)));
            }
        }

        // Double straight of length 2
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight2) {
            int start = last_move.combination == Move::Combination::kPass ? 4 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (player_cards[i - 3] >= 2 && player_cards[i - 4] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight2, i)));
                }
            }
        }

        // Double straight of length 3
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight3) {
            int start = last_move.combination == Move::Combination::kPass ? 5 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (player_cards[i - 3] >= 2 && player_cards[i - 4] >= 2 && player_cards[i - 5] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight3, i)));
                }
            }
        }

        // Double straight of length 4
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight4) {
            int start = last_move.combination == Move::Combination::kPass ? 6 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (player_cards[i - 3] >= 2 && player_cards[i - 4] >= 2 && player_cards[i - 5] >= 2 && player_cards[i - 6] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight4, i)));
                }
            }
        }

        // Double straight of length 5
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight5) {
            int start = last_move.combination == Move::Combination::kPass ? 7 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (player_cards[i - 3] >= 2 && player_cards[i - 4] >= 2 && player_cards[i - 5] >= 2 && player_cards[i - 6] >= 2 && player_cards[i - 7] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight5, i)));
                }
            }
        }

        // Double straight of length 6
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight6) {
            int start = last_move.combination == Move::Combination::kPass ? 8 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (player_cards[i - 3] >= 2 && player_cards[i - 4] >= 2 && player_cards[i - 5] >= 2 && player_cards[i - 6] >= 2 && player_cards[i - 7] >= 2 &&
                player_cards[i - 8] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight6, i)));
                }
            }
        }

        // Double straight of length 7
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight7) {
            int start = last_move.combination == Move::Combination::kPass ? 9 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (player_cards[i - 3] >= 2 && player_cards[i - 4] >= 2 && player_cards[i - 5] >= 2 && player_cards[i - 6] >= 2 && player_cards[i - 7] >= 2 &&
                player_cards[i - 8] >= 2 && player_cards[i - 9] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight7, i)));
                }
            }
        }

        // Double straight of length 8
        if (last_move.combination == Move::Combination::kPass) {
            for (int i = 10; i <= 14; ++i) {
                if (player_cards[i - 3] >= 2 && player_cards[i - 4] >= 2 && player_cards[i - 5] >= 2 && player_cards[i - 6] >= 2 && player_cards[i - 7] >= 2 &&
                player_cards[i - 8] >= 2 && player_cards[i - 9] >= 2 && player_cards[i - 10] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight8, i)));
                }
            }
        }

        // Triple straight of length 2
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kTripleStraight2) {
            int start = last_move.combination == Move::Combination::kPass ? 4 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (player_cards[i - 3] >= 3 && player_cards[i - 4] >= 3) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kTripleStraight2, i)));
                }
            }
        }

        // Triple straight of length 3
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kTripleStraight3) {
            int start = last_move.combination == Move::Combination::kPass ? 5 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (player_cards[i - 3] >= 3 && player_cards[i - 4] >= 3 && player_cards[i - 5] >= 3) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kTripleStraight3, i)));
                }
            }
        }

        // Triple straight of length 4
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kTripleStraight4) {
            int start = last_move.combination == Move::Combination::kPass ? 6 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (player_cards[i - 3] >= 3 && player_cards[i - 4] >= 3 && player_cards[i - 5] >= 3 && player_cards[i - 6] >= 3) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kTripleStraight4, i)));
                }
            }
        }

        // Triple straight of length 5
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kTripleStraight5) {
            int start = last_move.combination == Move::Combination::kPass ? 7 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (player_cards[i - 3] >= 3 && player_cards[i - 4] >= 3 && player_cards[i - 5] >= 3 && player_cards[i - 6] >= 3 && player_cards[i - 7] >= 3) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kTripleStraight5, i)));
                }
            }
        }
    }
    // Otherwise, we output all moves that could be played by the opponent

    else {

        // This is the maximum number of cards the opponent could have
        int max_cards[13];
        for (int i = 0; i < 11; ++i) {
            max_cards[i] = 4 - player_cards[i] - table_cards[i];
        }
        max_cards[11] = 3 - player_cards[11] - table_cards[11];
        max_cards[12] = 1 - player_cards[12] - table_cards[12];

        // Singles
        if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kSingle) {
            int start = last_move.combination == Move::Combination::kPass ? 3 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (max_cards[i - 3] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kSingle, i)));
                }
            }
        }

        // Doubles
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDouble) && opponent_card_num >= 2) {
            int start = last_move.combination == Move::Combination::kPass ? 3 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDouble, i)));
                }
            }
        }

        // Triples
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kTriple) && opponent_card_num >= 3) {
            int start = last_move.combination == Move::Combination::kPass ? 3 : last_move.rank + 1;
            for (int i = start; i <= 13; ++i) {
                if (max_cards[i - 3] >= 3) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kTriple, i)));
                }
            }
        }

        // Full houses
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kFullHouse) && opponent_card_num >= 5) {
            int start = last_move.combination == Move::Combination::kPass ? 3 : last_move.rank + 1;
            for (int i = start; i <= 13; ++i) {
                if (max_cards[i - 3] >= 3) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kFullHouse, i)));
                }
            }
        }

        // Bombs
        if (last_move.combination == Move::Combination::kBomb && opponent_card_num >= 4) {
            int start = last_move.combination == Move::Combination::kPass ? 3 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] == 4 || (i == 14 && max_cards[11] == 3)) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kBomb, i)));
                    for (int j = 3; j <= 15; ++j) {
                        if (i != j && max_cards[j - 3] >= 1) {
                            legal_moves.push_back(encodeMove(Move(Move::Combination::kBomb, i, j)));
                        }
                    }
                }
            }
        }
        // Use bombs burn
        else if (opponent_card_num >= 4) {
            for (int i = 3; i <= 14; ++i) {
                if (max_cards[i - 3] == 4 || (i == 14 && max_cards[11] == 3)) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kBomb, i)));
                    for (int j = 3; j <= 15; ++j) {
                        if (i != j && max_cards[j - 3] >= 1) {
                            legal_moves.push_back(encodeMove(Move(Move::Combination::kBomb, i, j)));
                        }
                    }
                }
            }
        }

        // Straight of length 5
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight5) && opponent_card_num >= 5) {
            int start = last_move.combination == Move::Combination::kPass ? 6 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (max_cards[i - 3] >= 1 && max_cards[i - 4] >= 1 && max_cards[i - 5] >= 1 && max_cards[i - 6] >= 1 && max_cards[(i + 6) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight5, i)));
                }
            }
        }

        // Straight of length 6
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight6) && opponent_card_num >= 6) {
            int start = last_move.combination == Move::Combination::kPass ? 7 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (max_cards[i - 3] >= 1 && max_cards[i - 4] >= 1 && max_cards[i - 5] >= 1 && max_cards[i - 6] >= 1 && max_cards[i - 7] >= 1 &&
                max_cards[(i + 5) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight6, i)));
                }
            }
        }

        // Straight of length 7
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight7) && opponent_card_num >= 7) {
            int start = last_move.combination == Move::Combination::kPass ? 8 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (max_cards[i - 3] >= 1 && max_cards[i - 4] >= 1 && max_cards[i - 5] >= 1 && max_cards[i - 6] >= 1 && max_cards[i - 7] >= 1 &&
                max_cards[i - 8] >= 1 && max_cards[(i + 4) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight7, i)));
                }
            }
        }

        // Straight of length 8
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight8) && opponent_card_num >= 8) {
            int start = last_move.combination == Move::Combination::kPass ? 9 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (max_cards[i - 3] >= 1 && max_cards[i - 4] >= 1 && max_cards[i - 5] >= 1 && max_cards[i - 6] >= 1 && max_cards[i - 7] >= 1 &&
                max_cards[i - 8] >= 1 && max_cards[i - 9] >= 1 && max_cards[(i + 3) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight8, i)));
                }
            }
        }

        // Straight of length 9
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight9) && opponent_card_num >= 9) {
            int start = last_move.combination == Move::Combination::kPass ? 10 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (max_cards[i - 3] >= 1 && max_cards[i - 4] >= 1 && max_cards[i - 5] >= 1 && max_cards[i - 6] >= 1 && max_cards[i - 7] >= 1 &&
                max_cards[i - 8] >= 1 && max_cards[i - 9] >= 1 && max_cards[i - 10] >= 1 && max_cards[(i + 2) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight9, i)));
                }
            }
        }

        // Straight of length 10
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight10) && opponent_card_num >= 10) {
            int start = last_move.combination == Move::Combination::kPass ? 11 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (max_cards[i - 3] >= 1 && max_cards[i - 4] >= 1 && max_cards[i - 5] >= 1 && max_cards[i - 6] >= 1 && max_cards[i - 7] >= 1 &&
                max_cards[i - 8] >= 1 && max_cards[i - 9] >= 1 && max_cards[i - 10] >= 1 && max_cards[i - 11] >= 1 && max_cards[(i + 1) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight10, i)));
                }
            }
        }

        // Straight of length 11
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight11) && opponent_card_num >= 11) {
            int start = last_move.combination == Move::Combination::kPass ? 12 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (max_cards[i - 3] >= 1 && max_cards[i - 4] >= 1 && max_cards[i - 5] >= 1 && max_cards[i - 6] >= 1 && max_cards[i - 7] >= 1 &&
                max_cards[i - 8] >= 1 && max_cards[i - 9] >= 1 && max_cards[i - 10] >= 1 && max_cards[i - 11] >= 1 && max_cards[i - 12] >= 1 &&
                max_cards[i % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight11, i)));
                }
            }
        }

        // Straight of length 12
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight12) && opponent_card_num >= 12) {
            int start = last_move.combination == Move::Combination::kPass ? 13 : last_move.rank + 1;
            for (int i = start; i <= 15; ++i) {
                if (max_cards[i - 3] >= 1 && max_cards[i - 4] >= 1 && max_cards[i - 5] >= 1 && max_cards[i - 6] >= 1 && max_cards[i - 7] >= 1 &&
                max_cards[i - 8] >= 1 && max_cards[i - 9] >= 1 && max_cards[i - 10] >= 1 && max_cards[i - 11] >= 1 && max_cards[i - 12] >= 1 &&
                max_cards[i - 13] >= 1 && max_cards[(i - 1) % 13] >= 1) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight12, i)));
                }
            }
        }

        // Straight of length 13
        if (last_move.combination == Move::Combination::kPass && opponent_card_num >= 13) {
            bool possible = true;
            for (int i = 0; i < 13; ++i) {
                if (max_cards[i] == 0) {
                    possible = false;
                    break;
                }
            }
            if (possible) {
                legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight13, 15)));
            }
            
        }

        // Double straight of length 2
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight2) && opponent_card_num >= 4) {
            int start = last_move.combination == Move::Combination::kPass ? 4 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] >= 2 && max_cards[i - 4] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight2, i)));
                }
            }
        }

        // Double straight of length 3
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight3) && opponent_card_num >= 6) {
            int start = last_move.combination == Move::Combination::kPass ? 5 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] >= 2 && max_cards[i - 4] >= 2 && max_cards[i - 5] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight3, i)));
                }
            }
        }

        // Double straight of length 4
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight4) && opponent_card_num >= 8) {
            int start = last_move.combination == Move::Combination::kPass ? 6 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] >= 2 && max_cards[i - 4] >= 2 && max_cards[i - 5] >= 2 && max_cards[i - 6] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight4, i)));
                }
            }
        }

        // Double straight of length 5
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight5) && opponent_card_num >= 10) {
            int start = last_move.combination == Move::Combination::kPass ? 7 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] >= 2 && max_cards[i - 4] >= 2 && max_cards[i - 5] >= 2 && max_cards[i - 6] >= 2 && max_cards[i - 7] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight5, i)));
                }
            }
        }

        // Double straight of length 6
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight6) && opponent_card_num >= 12) {
            int start = last_move.combination == Move::Combination::kPass ? 8 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] >= 2 && max_cards[i - 4] >= 2 && max_cards[i - 5] >= 2 && max_cards[i - 6] >= 2 && max_cards[i - 7] >= 2 && max_cards[i - 8] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight6, i)));
                }
            }
        }

        // Double straight of length 7
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight7) && opponent_card_num >= 14) {
            int start = last_move.combination == Move::Combination::kPass ? 9 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] >= 2 && max_cards[i - 4] >= 2 && max_cards[i - 5] >= 2 && max_cards[i - 6] >= 2 && max_cards[i - 7] >= 2 && max_cards[i - 8] >= 2 && max_cards[i - 9] >= 2) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kDoubleStraight7, i)));
                }
            }
        }

        // Triple straight of length 2
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kTripleStraight2) && opponent_card_num >= 6) {
            int start = last_move.combination == Move::Combination::kPass ? 4 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] >= 3 && max_cards[i - 4] >= 3) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kTripleStraight2, i)));
                }
            }
        }

        // Triple straight of length 3
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kTripleStraight3) && opponent_card_num >= 9) {
            int start = last_move.combination == Move::Combination::kPass ? 5 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] >= 3 && max_cards[i - 4] >= 3 && max_cards[i - 5] >= 3) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kTripleStraight3, i)));
                }
            }
        }

        // Triple straight of length 4
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kTripleStraight4) && opponent_card_num >= 12) {
            int start = last_move.combination == Move::Combination::kPass ? 6 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] >= 3 && max_cards[i - 4] >= 3 && max_cards[i - 5] >= 3 && max_cards[i - 6] >= 3) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kTripleStraight4, i)));
                }
            }
        }

        // Triple straight of length 5
        if ((last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kTripleStraight5) && opponent_card_num >= 15) {
            int start = last_move.combination == Move::Combination::kPass ? 7 : last_move.rank + 1;
            for (int i = start; i <= 14; ++i) {
                if (max_cards[i - 3] >= 3 && max_cards[i - 4] >= 3 && max_cards[i - 5] >= 3 && max_cards[i - 6] >= 3 && max_cards[i - 7] >= 3) {
                    legal_moves.push_back(encodeMove(Move(Move::Combination::kTripleStraight5, i)));
                }
            }
        }

    }
        
    return legal_moves;
    
}

std::ostream& operator<<(std::ostream& os, const Game& game) {
    os << "Player turn: " << game.player_turn << '\n';
    os << "Player cards: ";
    int sum = 0;
    for (int i = 0; i < 13; ++i) {
        char rankChar = rankToChar(i + 3);
        if (game.player_cards[i] < 0 || game.player_cards[i] > 4) {
            os << "Error: Player cards out of range: " << i << " " << game.player_cards[i] << endl;
            std::cout << "Error: Player cards out of range: " << i << " " << game.player_cards[i] << endl;
        }
        for (int j = 0; j < game.player_cards[i]; ++j) {
            os << rankChar;
        }
        sum += game.player_cards[i];
    }
    os << " (" << sum << ")\nOpponent cards: " << game.opponent_card_num;
    if (sum < 0 || sum > 16) {
        os << "Error: Too many cards in player's hand: " << sum << endl;
        std::cout << "Error: Too many cards in player's hand: " << sum << endl;
    }
    if (game.opponent_card_num < 0 || game.opponent_card_num > 16) {
        os << "Error: Too many cards in opponent's hand: " << game.opponent_card_num << endl;
        std::cout << "Error: Too many cards in opponent's hand: " << game.opponent_card_num << endl;
    }
    os << "\nTable cards: ";
    sum = 0;
    for (int i = 0; i < 13; ++i) {
        char rankChar = rankToChar(i + 3);
        for (int j = 0; j < game.table_cards[i]; ++j) {
            os << rankChar;
        }
        sum += game.table_cards[i];
        if (game.table_cards[i] < 0 || game.table_cards[i] > 4) {
            os << "Error: Table cards out of range: " << game.table_cards[i] << endl;
            std::cout << "Error: Table cards out of range: " << game.table_cards[i] << endl;
        }
    }
    if (sum > 0) os << ' ';
    if (sum < 0 || sum > 32) {
        os << "Error: Too many cards on table: " << sum << endl;
        std::cout << "Error: Too many cards on table: " << sum << endl;
    }
    os << "(" << sum << ")\nLast move: " << game.last_move;

    return os;
}