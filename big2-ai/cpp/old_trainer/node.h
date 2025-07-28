#ifndef NODE_H
#define NODE_H

#include "game.h"
#include "move.h"

#include <vector>

using std::vector;

class Node {
    public:
        Node() = default;
        Node(const Node &) = delete;
        Node(Node &&) = delete;
        ~Node() = default;

        Node(const Game &game);

        Node(const Game &game, Node *parent);
        
    private:
        Game game_;

        Node *parent_{nullptr};

        Node *next_sibling_{nullptr};

        Node *first_child_{nullptr};

        double value_{1.0};

        vector<Move> moves_;

        vector<double> probabilities_;

        // Nodes where the move is forced have different behavior
        bool forced_move_{false};

        int visits_{0};

};

#endif
