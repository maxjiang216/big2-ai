#include "game.h"
#include "util.h"
#include "move.h"

#include <array>
#include <iostream>

using namespace std;

int main() {
    for (int i = 0; i < 472; ++i) {
        try {
            Move move(i);
            cout << move << ' ';
            for (int j = 0; j < 13; ++j) {
                cout << MOVE_TO_CARDS.at(i)[j] << ",";
            }
            cout << MOVE_TO_CARDS.at(i)[13] << '\n';
            
        } catch (const std::exception& e) {
            // Skip invalid moves
            continue;
        }
    }
    return 0;
}