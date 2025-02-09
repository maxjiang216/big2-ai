#include "game.h"
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

Game::Game(int seed) {
    // Create a deck of 48 card indices (0 to 47)
    std::vector<int> deck(48);
    std::iota(deck.begin(), deck.end(), 0);  // Fill deck with 0, 1, ..., 47

    // Initialize a random engine with the provided seed
    std::mt19937 rng(seed);

    // Randomly choose who goes first
    player_turn = rng() % 2 == 0;

    // Shuffle the deck using std::shuffle (Fisher-Yates algorithm)
    std::shuffle(deck.begin(), deck.end(), rng);

    // Clear any pre-existing bits (just to be safe)
    player_cards.reset();
    opponent_cards.reset();
    played_cards.reset();

    // Deal 16 cards to the player
    for (int i = 0; i < 16; ++i) {
        player_cards.set(deck[i]);
    }

    // Deal 16 cards to the opponent
    for (int i = 16; i < 32; ++i) {
        opponent_cards.set(deck[i]);
    }
    // The remaining 16 cards (indices 32 to 47) remain unused.
}

void Game::do_move(int move) {
    // Check if the move is valid
    if (!player_cards.test(move)) {
        throw std::invalid_argument("Invalid move: card not in player's hand");
    }
}

vector<int> Game::get_legal_moves() const {
    vector<int> legal_moves;

    // Pass as last move means new trick
    // Cannot pass for new trick
    if (last_move.combination != Move::Combination::kPass) {
        legal_moves.push_back(0);
    }

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
    // Use bomb as burn
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
            if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[(i + 8) % 15] >= 1) {
                legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight5, i)));
            }
        }
    }

    // Straight of length 6
    if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight6) {
        int start = last_move.combination == Move::Combination::kPass ? 7 : last_move.rank + 1;
        for (int i = start; i <= 15; ++i) {
            if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
            player_cards[(i + 7) % 15] >= 1) {
                legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight6, i)));
            }
        }
    }

    // Straight of length 7
    if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight7) {
        int start = last_move.combination == Move::Combination::kPass ? 8 : last_move.rank + 1;
        for (int i = start; i <= 15; ++i) {
            if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
            player_cards[i - 8] >= 1 && player_cards[(i + 6) % 15] >= 1) {
                legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight7, i)));
            }
        }
    }

    // Straight of length 8
    if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight8) {
        int start = last_move.combination == Move::Combination::kPass ? 9 : last_move.rank + 1;
        for (int i = start; i <= 15; ++i) {
            if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
            player_cards[i - 8] >= 1 && player_cards[i - 9] >= 1 && player_cards[(i + 5) % 15] >= 1) {
                legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight8, i)));
            }
        }
    }

    // Straight of length 9
    if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight9) {
        int start = last_move.combination == Move::Combination::kPass ? 10 : last_move.rank + 1;
        for (int i = start; i <= 15; ++i) {
            if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
            player_cards[i - 8] >= 1 && player_cards[i - 9] >= 1 && player_cards[i - 10] >= 1 && player_cards[(i + 4) % 15] >= 1) {
                legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight9, i)));
            }
        }
    }

    // Straight of length 10
    if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kStraight10) {
        int start = last_move.combination == Move::Combination::kPass ? 11 : last_move.rank + 1;
        for (int i = start; i <= 15; ++i) {
            if (player_cards[i - 3] >= 1 && player_cards[i - 4] >= 1 && player_cards[i - 5] >= 1 && player_cards[i - 6] >= 1 && player_cards[i - 7] >= 1 &&
            player_cards[i - 8] >= 1 && player_cards[i - 9] >= 1 && player_cards[i - 10] >= 1 && player_cards[i - 11] >= 1 && player_cards[(i + 3) % 15] >= 1) {
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
            player_cards[(i + 2) % 15] >= 1) {
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
            player_cards[i - 13] >= 1 && player_cards[(i + 1) % 15] >= 1) {
                legal_moves.push_back(encodeMove(Move(Move::Combination::kStraight12, i)));
            }
        }
    }

    // Straight of length 13
    if (last_move.combination == Move::Combination::kPass) {
        bool possible = true;
        for (int i = 0; i < 15; ++i) {
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
    if (last_move.combination == Move::Combination::kPass || last_move.combination == Move::Combination::kDoubleStraight8) {
        int start = last_move.combination == Move::Combination::kPass ? 10 : last_move.rank + 1;
        for (int i = start; i <= 14; ++i) {
            if (player_cards[i - 3] >= 2 && player_cards[i - 4] >= 2 && player_cards[i - 5] >= 2 && player_cards[i - 6] >= 2 && player_cards[i - 7] >= 2 &&
            player_cards[i - 8] >= 2 && player_cards[i - 9] >= 2) {
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

    return legal_moves;
    
}

