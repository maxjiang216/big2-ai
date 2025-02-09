#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
using namespace std;

// --- Global Definitions ---
// We have 13 ranks: indices 0..12 correspond to: 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2.
// Limits: Ranks 3–K (indices 0..10) have 4 copies; Ace (index 11) has 3 copies; 2 (index 12) has 1 copy.
static const array<int, 13> LIMITS = {4,4,4,4,4,4,4,4,4,4,4,3,1};

// --- Helper: n Choose k (for small values) ---
long double comb(int n, int k) {
    if (k > n) return 0;
    if (k > n - k) k = n - k;
    long double result = 1;
    for (int i = 1; i <= k; i++) {
        result = result * (n - i + 1) / i;
    }
    return result;
}

// --- Enumerate All Starting Hands ---
// Each hand is an array<int, 13> (counts per rank) with total count = 16.
// We also compute a "weight" (the product of binomial coefficients) though it’s not used in move counting.
void gen_compositions(int i, int remaining, array<int, 13>& current, long double weight,
                      vector<pair<array<int, 13>, long double>>& allHands) {
    if (i == 13) {
        if (remaining == 0)
            allHands.push_back({current, weight});
        return;
    }
    int limit = LIMITS[i];
    for (int x = 0; x <= min(limit, remaining); x++) {
        current[i] = x;
        long double new_weight = weight * comb(limit, x);
        gen_compositions(i + 1, remaining - x, current, new_weight, allHands);
    }
}

// --- Mapping: Convert a rank value (3..15) to index (0..12) ---
// We represent 3..13 as 3..K, 14 as Ace, and 15 as 2.
inline int r2i(int r) {
    return (r == 15 ? 12 : r - 3);
}

// --- Legal Move Check Functions ---

// A single is legal if at least one copy is present.
bool can_play_single(const array<int, 13>& comp, int r) {
    return comp[r2i(r)] >= 1;
}

// A double requires at least 2 copies.
bool can_play_double(const array<int, 13>& comp, int r) {
    return comp[r2i(r)] >= 2;
}

// A triple requires at least 3 copies.
bool can_play_triple(const array<int, 13>& comp, int r) {
    return comp[r2i(r)] >= 3;
}

// A full house: a triple of one rank and a pair of another.
bool can_play_full_house(const array<int, 13>& comp, int triple, int pair) {
    return comp[r2i(triple)] >= 3 && comp[r2i(pair)] >= 2;
}

// A bomb: if r != Ace (14) requires 4 copies; if Ace requires 3 copies.
// Optionally, an extra card (aux ≠ 0) is required.
bool can_play_bomb(const array<int, 13>& comp, int r, int aux) {
    bool ok = (r == 14 ? (comp[r2i(r)] >= 3) : (comp[r2i(r)] >= 4));
    if (aux != 0)
        ok = ok && (comp[r2i(aux)] >= 1);
    return ok;
}

// A straight (of singles) of length L (5 <= L <= 13) with highest card H.
// Special case: if H == L+1, then the straight uses 2 as the low card
// (i.e. required cards are {2} ∪ {3,4,…,H}). Otherwise, required cards are {H-L+1, …, H}.
bool can_play_straight(const array<int, 13>& comp, int L, int H) {
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

// A double straight (sister): length L (2<=L<=8) with highest pair H.
// Requires at least 2 copies in each rank in the range [H-L+1, H].
bool can_play_sister(const array<int, 13>& comp, int L, int H) {
    for (int r = H - L + 1; r <= H; r++) {
        if (comp[r2i(r)] < 2)
            return false;
    }
    return true;
}

// A triple straight: length L (2<=L<=5) with highest triple H.
// Requires at least 3 copies in each rank in the range [H-L+1, H].
bool can_play_triple_straight(const array<int, 13>& comp, int L, int H) {
    for (int r = H - L + 1; r <= H; r++) {
        if (comp[r2i(r)] < 3)
            return false;
    }
    return true;
}

// --- Count the Number of Legal Moves for a Given Hand Composition ---
int legal_moves_count(const array<int, 13>& comp) {
    int count = 0;
    // 1. Singles: ranks 3..15
    for (int r = 3; r <= 15; r++) {
        if (can_play_single(comp, r))
            count++;
    }
    // 2. Doubles: ranks 3..14
    for (int r = 3; r <= 14; r++) {
        if (can_play_double(comp, r))
            count++;
    }
    // 3. Triples: ranks 3..13
    for (int r = 3; r <= 13; r++) {
        if (can_play_triple(comp, r))
            count++;
    }
    // 4. Full houses: triple in 3..15, pair in 3..15 (must be different)
    for (int triple = 3; triple <= 15; triple++) {
        for (int pair = 3; pair <= 15; pair++) {
            if (pair == triple) continue;
            if (can_play_full_house(comp, triple, pair))
                count++;
        }
    }
    // 5. Bombs: for r in 3..14
    for (int r = 3; r <= 14; r++) {
        if (can_play_bomb(comp, r, 0))
            count++;
        // With extra card: valid aux values are all a in 3..15 except a == r.
        for (int a = 3; a < r; a++) {
            if (can_play_bomb(comp, r, a))
                count++;
        }
        for (int a = r + 1; a <= 15; a++) {
            if (can_play_bomb(comp, r, a))
                count++;
        }
    }
    // 6. Straights: for each length L from 5 to 12, for H in [L+1, 15]
    for (int L = 5; L <= 12; L++) {
        for (int H = L + 1; H <= 15; H++) {
            if (can_play_straight(comp, L, H))
                count++;
        }
    }
    if (can_play_straight(comp, 13, 15))
        count++;
    // 7. Sisters (double straights): for L from 2 to 8, for H in [L+2, 14]
    for (int L = 2; L <= 8; L++) {
        for (int H = L + 2; H <= 14; H++) {
            if (can_play_sister(comp, L, H))
                count++;
        }
    }
    // 8. Triple straights: for L from 2 to 5, for H in [L+2, 14]
    for (int L = 2; L <= 5; L++) {
        for (int H = L + 2; H <= 14; H++) {
            if (can_play_triple_straight(comp, L, H))
                count++;
        }
    }
    return count;
}

// --- Main Function ---
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<pair<array<int, 13>, long double>> allHands;
    array<int, 13> current = {0};
    gen_compositions(0, 16, current, 1.0L, allHands);
    cout << "Total number of compositions: " << allHands.size() << "\n";

    int max_moves = -1;
    array<int, 13> best_comp;
    for (const auto &entry : allHands) {
        const array<int, 13>& comp = entry.first;
        int moves = legal_moves_count(comp);
        if (moves > max_moves) {
            max_moves = moves;
            best_comp = comp;
        }
    }

    cout << "Maximum legal moves count: " << max_moves << "\n";
    cout << "Starting hand composition with most legal moves:\n";

    // Mapping indices to rank labels:
    const array<string, 13> rank_labels = {"3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"};
    for (int i = 0; i < 13; i++) {
        cout << rank_labels[i] << ": " << best_comp[i] << "  ";
    }
    cout << "\n";

    return 0;
}
