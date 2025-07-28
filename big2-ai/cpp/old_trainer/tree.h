#ifndef TREE_H
#define TREE_H

#include "node.h"

// Monte Carlo Search Tree
class Tree {

    public:
        Tree() = default;
        ~Tree() = default;
        Tree(const Tree &) = default;
        Tree(Tree &&) = default;
        Tree &operator=(const Tree &) = default;
        Tree &operator=(Tree &&) = default;

        Tree(bool player_turn, int player_cards[13], int max_searches, float c_puct);

    private:

        Node *root_{nullptr};

        Node *cur_{nullptr};

        int max_searches_;

        float c_puct_{1.0};

};

#endif
