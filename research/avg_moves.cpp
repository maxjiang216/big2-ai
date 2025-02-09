#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
using namespace std;

// --- Global Definitions ---
// We have 13 ranks (indices 0..12): 
//   3..13 represent 3..K (each 4 copies), Ace (14) has 3 copies, 2 (15) has 1 copy.
static const array<int, 13> LIMITS = {4,4,4,4,4,4,4,4,4,4,4,3,1};

// Convert a card's rank (3..15) to an index (0..12). (2 is represented as 15.)
inline int r2i(int r) {
    return (r == 15 ? 12 : r - 3);
}

// --- Legal Move Check Functions ---
inline bool can_play_single(const array<int, 13>& comp, int r) {
    return comp[r2i(r)] >= 1;
}
inline bool can_play_double(const array<int, 13>& comp, int r) {
    return comp[r2i(r)] >= 2;
}
inline bool can_play_triple(const array<int, 13>& comp, int r) {
    return comp[r2i(r)] >= 3;
}
inline bool can_play_full_house(const array<int, 13>& comp, int triple, int pair) {
    return comp[r2i(triple)] >= 3 && comp[r2i(pair)] >= 2;
}
inline bool can_play_bomb(const array<int, 13>& comp, int r, int aux) {
    bool ok = (r == 14 ? (comp[r2i(r)] >= 3) : (comp[r2i(r)] >= 4));
    if (aux != 0)
        ok = ok && (comp[r2i(aux)] >= 1);
    return ok;
}
// A straight of singles: length L (5<=L<=13) with highest card H.
// Special case: if H == L+1 then the straight is interpreted as using 2 as the low card.
inline bool can_play_straight(const array<int, 13>& comp, int L, int H) {
    if (H == L + 1) {
        if (comp[r2i(15)] < 1) return false;
        for (int r = 3; r <= H; r++) {
            if (comp[r2i(r)] < 1)
                return false;
        }
    } else {
        for (int r = H - L + 1; r <= H; r++) {
            if (comp[r2i(r)] < 1)
                return false;
        }
    }
    return true;
}
// Double straight ("sister"): length L (2<=L<=8) with highest pair H.
inline bool can_play_sister(const array<int, 13>& comp, int L, int H) {
    for (int r = H - L + 1; r <= H; r++) {
        if (comp[r2i(r)] < 2)
            return false;
    }
    return true;
}
// Triple straight: length L (2<=L<=5) with highest triple H.
inline bool can_play_triple_straight(const array<int, 13>& comp, int L, int H) {
    for (int r = H - L + 1; r <= H; r++) {
        if (comp[r2i(r)] < 3)
            return false;
    }
    return true;
}

// --- Count the Number of Legal Moves for a Given Hand Composition ---
int legal_moves_count(const array<int, 13>& comp) {
    int count = 0;
    // 1. Singles: for each rank r in 3..15.
    for (int r = 3; r <= 15; r++) {
        if (can_play_single(comp, r))
            count++;
    }
    // 2. Doubles: for each rank r in 3..14.
    for (int r = 3; r <= 14; r++) {
        if (can_play_double(comp, r))
            count++;
    }
    // 3. Triples: for each rank r in 3..13.
    for (int r = 3; r <= 13; r++) {
        if (can_play_triple(comp, r))
            count++;
    }
    // 4. Full houses: for triple in 3..15 and pair in 3..15 (different).
    for (int triple = 3; triple <= 15; triple++) {
        for (int pair = 3; pair <= 15; pair++) {
            if (pair == triple) continue;
            if (can_play_full_house(comp, triple, pair))
                count++;
        }
    }
    // 5. Bombs: for r in 3..14.
    for (int r = 3; r <= 14; r++) {
        if (can_play_bomb(comp, r, 0))
            count++;
        // With extra card: valid aux values: all a in 3..15 except a == r.
        for (int a = 3; a < r; a++) {
            if (can_play_bomb(comp, r, a))
                count++;
        }
        for (int a = r + 1; a <= 15; a++) {
            if (can_play_bomb(comp, r, a))
                count++;
        }
    }
    // 6. Straights: for each length L from 5 to 12, for H in [L+1, 15].
    for (int L = 5; L <= 12; L++) {
        for (int H = L + 1; H <= 15; H++) {
            if (can_play_straight(comp, L, H))
                count++;
        }
    }
    if (can_play_straight(comp, 13, 15))
        count++;
    // 7. Sisters (Double straights): for L from 2 to 8, for H in [L+2, 14].
    for (int L = 2; L <= 8; L++) {
        for (int H = L + 2; H <= 14; H++) {
            if (can_play_sister(comp, L, H))
                count++;
        }
    }
    // 8. Triple straights: for L from 2 to 5, for H in [L+2, 14].
    for (int L = 2; L <= 5; L++) {
        for (int H = L + 2; H <= 14; H++) {
            if (can_play_triple_straight(comp, L, H))
                count++;
        }
    }
    return count;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // --- Build the deck ---
    // The deck is 48 cards: ranks 3..13 (4 copies each), Ace (14) (3 copies), and 2 (15) (1 copy).
    vector<int> deck;
    for (int r = 3; r <= 13; r++) {
        for (int i = 0; i < 4; i++) {
            deck.push_back(r);
        }
    }
    for (int i = 0; i < 3; i++) {
        deck.push_back(14);
    }
    deck.push_back(15);
    const int deckSize = deck.size();  // should be 48

    // --- Setup random generator ---
    random_device rd;
    mt19937 gen(rd());

    // --- Monte Carlo Simulation ---
    const long long iterations = 1000000;  // 1e6 iterations
    long long total_moves = 0;

    for (long long iter = 0; iter < iterations; iter++) {
        // Shuffle the deck to simulate a random deal.
        shuffle(deck.begin(), deck.end(), gen);

        // Build the composition for the first 16 cards.
        array<int, 13> comp = {0};
        for (int i = 0; i < 16; i++) {
            int card = deck[i];
            comp[r2i(card)]++;
        }

        int moves = legal_moves_count(comp);
        total_moves += moves;
    }

    double avg_moves = static_cast<double>(total_moves) / iterations;
    cout << "Average number of legal moves in a starting hand: " << avg_moves << "\n";

    return 0;
}
