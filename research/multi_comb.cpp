#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std;

// Function to check if the given hand has at least two doubles of different ranks.
// This function counts the frequency of each rank (excluding the rank for 2, represented as 15)
// and returns true if at least two distinct ranks appear at least twice.
bool hasAtLeastTwoDoubles(const vector<int>& hand) {
    // Frequency vector for ranks 0..15 (we only use 3..15)
    vector<int> freq(16, 0);
    for (int card : hand) {
        freq[card]++;
    }

    bool flag = false;
    for (int r = 3; r <= 13; ++r) {
        if (freq[r] >= 3) {
            flag = true;
            break;
        }
    }
    if (!flag) return false;
    
    int doubleCount = 0;
    // Check ranks 3 through Ace (14). Note: Rank 15 (the 2) is excluded.
    for (int r = 3; r <= 14; ++r) {
        if (freq[r] >= 2) {
            doubleCount++;
        }
        if (doubleCount == 2) return true;
    }
    
    return false;
}

#include <vector>
#include <algorithm>

using namespace std;

bool hasStraightOfLengthX(const vector<int>& hand, int x) {
   // if (x < 5) return false; // Straights must be at least length 5

    vector<int> freq(16, 0); // Frequency count for ranks 3 to 15 (Ace = 14, 2 = 15)
    
    // Count occurrences of each rank in the hand
    for (int card : hand) {
        freq[card]++; // Only presence matters, not duplicates
    }

    // Standard check for consecutive numbers forming a straight
    int maxStraight = 0, currentStraight = 0, straits = 0;
    for (int r = 3; r <= 14; ++r) {
        if (freq[r] >= 3) {
            currentStraight++;
            maxStraight = max(maxStraight, currentStraight);
        } else {
            if (currentStraight >= x) straits++;
            currentStraight = 0;
        }
    }

    // // Special check for A-2-3 wrap-around
    // if (freq[15] && freq[3] && freq[4] && freq[5]) { // 2-3-4-5 must be present
    //     int wrapStraight = 4; // 2-3-4-5
    //     for (int r = 6; r <= 14; ++r) { // Continue from 6 to Ace
    //         if (freq[r]) wrapStraight++;
    //         else break;
    //     }
    //     maxStraight = max(maxStraight, wrapStraight);
    // }

    return maxStraight >= x;
}


int main() {
    // Number of Monte Carlo trials. Increase this number for a more accurate probability.
    const int numTrials = 10000000;
    int success = 0;

    // Build the deck according to the rules:
    // Ranks 3 to King (3 to 13): 4 copies each.
    // Ace (14): 3 copies.
    // 2 (15): 1 copy.
    vector<int> deck;
    for (int r = 3; r <= 13; ++r) {
        for (int i = 0; i < 4; ++i) {
            deck.push_back(r);
        }
    }
    for (int i = 0; i < 3; ++i) {
        deck.push_back(14); // Ace
    }
    deck.push_back(15); // 2
    
    // Setup random number generator.
    random_device rd;
    mt19937 gen(rd());
    
    // To measure simulation time (optional).
    auto start = chrono::high_resolution_clock::now();
    
    // Monte Carlo simulation loop.
    for (int t = 0; t < numTrials; ++t) {
        // Create a local copy of the deck and shuffle it.
        vector<int> currentDeck = deck;
        shuffle(currentDeck.begin(), currentDeck.end(), gen);
        
        // Deal the first 16 cards as the player's hand.
        vector<int> hand(currentDeck.begin(), currentDeck.begin() + 16);
        vector<int> hand2(currentDeck.begin() + 16, currentDeck.begin() + 32);
        
        // Use the function to check for at least two doubles.
        if (hasAtLeastTwoDoubles(hand)) {
            ++success;
        }
    }
    
    double probability = static_cast<double>(success) / numTrials;
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    cout << "Probability: " << probability << "\n";
    cout << "Trials: " << numTrials << ", Elapsed time: " << elapsed.count() << " seconds.\n";
    
    return 0;
}
