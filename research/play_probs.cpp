#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <map>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;

// --- Helper: Compute n choose k (for small n, k) ---
long double comb(int n, int k) {
    if (k > n)
        return 0;
    if (k > n - k)
        k = n - k;
    long double result = 1;
    for (int i = 1; i <= k; i++) {
        result = result * (n - i + 1) / i;
    }
    return result;
}

// --- Global definitions ---
// There are 13 ranks; we use indices 0..12 corresponding to:
 // 3,4,5,6,7,8,9,10,J,Q,K,A,2.
 // We represent the card’s “rank value” as follows:
 //   3..13 correspond to 3..K, 14 represents Ace, and 15 represents 2.
static const array<int, 13> LIMITS = {4,4,4,4,4,4,4,4,4,4,4,3,1};

// r2i: convert a card’s rank (3..15) to index 0..12.
inline int r2i(int r) {
    return (r == 15 ? 12 : r - 3);
}

// --- Enumerate all starting hands by rank composition ---
// Each hand is represented as an array<int,13> (counts for each rank).
// "weight" is the product over ranks of comb(LIMITS[i], count).
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

// --- Move eligibility functions ---
// Returns true if the composition (hand) can play a single of rank r.
bool can_play_single(const array<int, 13>& comp, int r) {
    return comp[r2i(r)] >= 1;
}

bool can_play_double(const array<int, 13>& comp, int r) {
    return comp[r2i(r)] >= 2;
}

bool can_play_triple(const array<int, 13>& comp, int r) {
    return comp[r2i(r)] >= 3;
}

bool can_play_full_house(const array<int, 13>& comp, int triple, int pair) {
    return comp[r2i(triple)] >= 3 && comp[r2i(pair)] >= 2;
}

bool can_play_bomb(const array<int, 13>& comp, int r, int aux) {
    // For a bomb: if r==14 (Ace) we require at least 3 copies; otherwise, 4 copies.
    bool ok = (r == 14 ? (comp[r2i(r)] >= 3) : (comp[r2i(r)] >= 4));
    // If aux != 0 then an extra card is required.
    if (aux != 0)
        ok = ok && (comp[r2i(aux)] >= 1);
    return ok;
}

// For a straight of length L (5<=L<=13) with highest card H.
// Special twist: if H == L+1, we interpret that as the case where 2 is used as the low card.
// In that case the required set is {2} (i.e. rank 15) and {3,...,H}.
// Otherwise the required set is {H-L+1, ..., H}.
bool can_play_straight(const array<int, 13>& comp, int L, int H) {
    if (H == L + 1) {
        if (comp[r2i(15)] < 1)
            return false;
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

// Double straight (“sister”): for length L (2<=L<=8) with highest pair H,
// required: for each rank in [H-L+1, H] the hand has at least 2 copies.
bool can_play_sister(const array<int, 13>& comp, int L, int H) {
    for (int r = H - L + 1; r <= H; r++) {
        if (comp[r2i(r)] < 2)
            return false;
    }
    return true;
}

// Triple straight: for length L (2<=L<=5) with highest triple H,
// required: for each rank in [H-L+1, H] the hand has at least 3 copies.
bool can_play_triple_straight(const array<int, 13>& comp, int L, int H) {
    for (int r = H - L + 1; r <= H; r++) {
        if (comp[r2i(r)] < 3)
            return false;
    }
    return true;
}

// For pretty printing: convert rank value to label.
string rank_label(int r) {
    if (r == 11)
        return "J";
    if (r == 12)
        return "Q";
    if (r == 13)
        return "K";
    if (r == 14)
        return "A";
    if (r == 15)
        return "2";
    return to_string(r);
}

int main() {
    // --- Enumerate all compositions ---
    vector<pair<array<int, 13>, long double>> allHands;
    array<int, 13> current = {0};
    gen_compositions(0, 16, current, 1.0L, allHands);
    long double total_weight = 0;
    for (auto &p : allHands) {
        total_weight += p.second;
    }
    cout << "Total weighted hands = " << total_weight << "\n";

    // --- For each legal move, accumulate the weight of hands that support it ---
    map<string, long double> results;

    // 0. Pass: always playable.
    results["Pass"] = total_weight;

    // 1. Singles: for each rank r in 3..15.
    cout << "Processing singles...\n";
    for (int r = 3; r <= 15; r++) {
        string desc = "Single " + rank_label(r);
        long double s = 0;
        for (auto &hand : allHands) {
            if (can_play_single(hand.first, r))
                s += hand.second;
        }
        results[desc] = s;
    }

    // 2. Doubles: for each rank r in 3..14 (2 is not allowed).
    cout << "Processing doubles...\n";
    for (int r = 3; r <= 14; r++) {
        string desc = "Double " + rank_label(r);
        long double s = 0;
        for (auto &hand : allHands) {
            if (can_play_double(hand.first, r))
                s += hand.second;
        }
        results[desc] = s;
    }

    // 3. Triples: for each rank r in 3..13.
    cout << "Processing triples...\n";
    for (int r = 3; r <= 13; r++) {
        string desc = "Triple " + to_string(r);
        long double s = 0;
        for (auto &hand : allHands) {
            if (can_play_triple(hand.first, r))
                s += hand.second;
        }
        results[desc] = s;
    }

    // 4. Full Houses: for triple rank in 3..15 and pair rank in 3..15 (pair != triple).
    cout << "Processing full houses...\n";
    for (int triple = 3; triple <= 15; triple++) {
        for (int pair = 3; pair <= 15; pair++) {
            if (pair == triple)
                continue;
            string desc = "FullHouse (" + rank_label(triple) + " over " + rank_label(pair) + ")";
            long double s = 0;
            for (auto &hand : allHands) {
                if (can_play_full_house(hand.first, triple, pair))
                    s += hand.second;
            }
            results[desc] = s;
        }
    }

    // 5. Bombs: for bomb rank r in 3..14; first, bomb with no extra card,
    // then bomb with an extra card (aux) for each valid aux.
    cout << "Processing bombs...\n";
    for (int r = 3; r <= 14; r++) {
        {
            string desc = "Bomb " + rank_label(r) + " (no extra)";
            long double s = 0;
            for (auto &hand : allHands) {
                if (can_play_bomb(hand.first, r, 0))
                    s += hand.second;
            }
            results[desc] = s;
        }
        // with extra card: valid aux values: from 3 to r-1 and from r+1 to 15.
        for (int a = 3; a < r; a++) {
            string desc = "Bomb " + rank_label(r) + " + extra " + rank_label(a);
            long double s = 0;
            for (auto &hand : allHands) {
                if (can_play_bomb(hand.first, r, a))
                    s += hand.second;
            }
            results[desc] = s;
        }
        for (int a = r + 1; a <= 15; a++) {
            string desc = "Bomb " + rank_label(r) + " + extra " + rank_label(a);
            long double s = 0;
            for (auto &hand : allHands) {
                if (can_play_bomb(hand.first, r, a))
                    s += hand.second;
            }
            results[desc] = s;
        }
    }

    // 6. Straights: for each length L from 5 to 12, for H in [L+1,15];
    // and for L==13, only possibility H==15.
    cout << "Processing straights...\n";
    for (int L = 5; L <= 12; L++) {
        for (int H = L + 1; H <= 15; H++) {
            string desc = "Straight " + to_string(L) + " (highest " + rank_label(H) + ")";
            long double s = 0;
            for (auto &hand : allHands) {
                if (can_play_straight(hand.first, L, H))
                    s += hand.second;
            }
            results[desc] = s;
        }
    }
    {
        int L = 13, H = 15;
        string desc = "Straight 13 (unique)";
        long double s = 0;
        for (auto &hand : allHands) {
            if (can_play_straight(hand.first, L, H))
                s += hand.second;
        }
        results[desc] = s;
    }

    // 7. Sisters (Double Straights): for L from 2 to 8, for H in [L+2,14].
    cout << "Processing double straights...\n";
    for (int L = 2; L <= 8; L++) {
        for (int H = L + 2; H <= 14; H++) {
            string desc = "DoubleStraight " + to_string(L) + " (highest " + rank_label(H) + ")";
            long double s = 0;
            for (auto &hand : allHands) {
                if (can_play_sister(hand.first, L, H))
                    s += hand.second;
            }
            results[desc] = s;
        }
    }

    // 8. Triple Straights: for L from 2 to 5, for H in [L+2,14].
    cout << "Processing triple straights...\n";
    for (int L = 2; L <= 5; L++) {
        for (int H = L + 2; H <= 14; H++) {
            string desc = "TripleStraight " + to_string(L) + " (highest " + rank_label(H) + ")";
            long double s = 0;
            for (auto &hand : allHands) {
                if (can_play_triple_straight(hand.first, L, H))
                    s += hand.second;
            }
            results[desc] = s;
        }
    }

    // --- Finally, print the percentage for each move ---
    cout << "\nPercentage of starting hands that can play each move (approx.):\n\n";
    for (const auto &entry : results) {
        long double perc = entry.second / total_weight * 100.0L;
        cout << setw(35) << left << entry.first << ": " 
             << fixed << setprecision(4) << perc << "%" << "\n";
    }

    return 0;
}
