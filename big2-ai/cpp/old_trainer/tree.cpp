#include "tree.h"
#include "node.h"
#include "game.h"
#include <random>

Tree::Tree(bool player_turn, int player_cards[13], int max_searches, float c_puct): max_searches_(max_searches), c_puct_(c_puct) {
    root_ = new Node(Game(player_turn, player_cards));
    cur_ = root_;
}
