#include <bits/stdc++.h>
#include <algorithm>  // for std::all_of
using namespace std;

// === Configuration ===
static constexpr bool ALLOW_VOLUNTARY_PASSES = false;
static constexpr bool ALLOW_SINGLE_START = false;

// === Combination definitions ===
enum ComboType {
    NONE,
    SINGLE, DOUBLE, TRIPLE,
    FULLHOUSE,
    STRAIGHT, SISTERS, TRIPLESTRAIGHT,
    BOMB,
    PASS
};

static const vector<string> ComboNames = {
    "NONE",
    "SINGLE", "DOUBLE", "TRIPLE",
    "FULLHOUSE",
    "STRAIGHT", "SISTERS", "TRIPLESTRAIGHT",
    "BOMB",
    "PASS"
};

struct Combo {
    ComboType type;
    int rank;    // 0=3, 1=4, ..., 9=Q, 10=K, 11=A, 12=2
    int length;  // for STRAIGHT/SISTERS/TRIPLESTRAIGHT
};

// Generate all legal combos in 'hand'
vector<Combo> generate_combos(const array<int,13>& hand) {
    vector<Combo> combos;
    for(int r=0; r<13; ++r) {
        int cnt = hand[r];
        if(cnt >= 1) combos.push_back({SINGLE, r, 1});
        if(cnt >= 2) combos.push_back({DOUBLE, r, 1});
        if(cnt >= 3 && r != 11) combos.push_back({TRIPLE, r, 1});
    }
    vector<int> triples, doubles;
    for(int r=0; r<13; ++r) {
        if(hand[r]>=3 && r!=11) triples.push_back(r);
        if(hand[r]>=2)          doubles.push_back(r);
    }
    for(int tr: triples)
        for(int dr: doubles)
            if(dr!=tr)
                combos.push_back({FULLHOUSE, tr, 1});
    array<bool,13> p1{}, p2{}, p3{};
    for(int r=0;r<13;++r) {
        p1[r] = hand[r]>=1;
        p2[r] = hand[r]>=2;
        p3[r] = (hand[r]>=3 && r!=11);
    }
    for(int i=0; i<13; ) {
        if(p1[i]) {
            int j=i; while(j<13 && p1[j]) ++j;
            int run = j - i;
            if(run>=5){
                for(int len=5; len<=run; ++len)
                    for(int s=i; s+len-1<j; ++s)
                        combos.push_back({STRAIGHT, s+len-1, len});
            }
            i = j;
        } else { ++i; }
    }
    for(int i=0; i<13; ) {
        if(p2[i]) {
            int j=i; while(j<13 && p2[j]) ++j;
            int run = j - i;
            if(run>=2){
                for(int len=2; len<=run; ++len)
                    for(int s=i; s+len-1<j; ++s)
                        combos.push_back({SISTERS, s+len-1, len});
            }
            i = j;
        } else { ++i; }
    }
    for(int i=0; i<13; ) {
        if(p3[i]) {
            int j=i; while(j<13 && p3[j]) ++j;
            int run = j - i;
            if(run>=2){
                for(int len=2; len<=run; ++len)
                    for(int s=i; s+len-1<j; ++s)
                        combos.push_back({TRIPLESTRAIGHT, s+len-1, len});
            }
            i = j;
        } else { ++i; }
    }
    for(int r=0; r<13; ++r)
        if(hand[r]==4)
            combos.push_back({BOMB, r, 1});
    if(hand[11]>=3)
        combos.push_back({BOMB, 11, 1});
    return combos;
}

bool combo_beats(const Combo &c, const Combo &base) {
    if(base.type == BOMB) return (c.type == BOMB && c.rank > base.rank);
    if(c.type == BOMB) return true;
    if(c.type != base.type || c.length != base.length) return false;
    return c.rank > base.rank;
}

bool hand_is_empty(const array<int,13>& hand) {
    return all_of(hand.begin(), hand.end(), [](int cnt){ return cnt == 0; });
}

int simulate_game(
    mt19937 &rng,
    long long &out_passes,
    array<long long, 10> &out_tricks,
    long long &out_two_moves,
    long long &out_three_moves,
    long long &out_big_choice_total_start,
    long long &out_big_choice_count_start,
    long long &out_big_choice_total_resp,
    long long &out_big_choice_count_resp
)
{
    out_passes = 0;
    out_tricks.fill(0);
    out_two_moves = 0;
    out_three_moves = 0;
    out_big_choice_total_start = out_big_choice_count_start = 0;
    out_big_choice_total_resp  = out_big_choice_count_resp  = 0;
    vector<int> deck;
    for(int r=0;r<13;++r){
        int copies = (r==11?3:(r==12?1:4));
        for(int i=0;i<copies;++i) deck.push_back(r);
    }
    shuffle(deck.begin(), deck.end(), rng);
    array<int,13> h1{}, h2{};
    for(int i=0;i<16;++i) h1[deck[i]]++;
    for(int i=16;i<32;++i) h2[deck[i]]++;
    array<array<int,13>,2> hands = {h1, h2};
    int current = 0, last_player = 0, moves = 0;
    Combo current_combo{NONE,0,0};
    while(!hand_is_empty(hands[0]) && !hand_is_empty(hands[1])) {
        auto &hand = hands[current];
        auto combos = generate_combos(hand);
        vector<Combo> legal;
        if(current_combo.type == NONE) {
            legal = combos;
        } else {
            for(auto &c: combos)
                if(combo_beats(c, current_combo))
                    legal.push_back(c);
        }
        vector<Combo> choices = legal;
        bool mustPass = legal.empty();
        bool canVoluntary = ALLOW_VOLUNTARY_PASSES && current_combo.type!=NONE && !legal.empty();
        if(mustPass || canVoluntary) choices.push_back({PASS,0,0});
        if(current_combo.type == NONE && !ALLOW_SINGLE_START) {
            vector<Combo> non_single_choices;
            for (const auto& c : choices)
                if (c.type != SINGLE) non_single_choices.push_back(c);
            if (!non_single_choices.empty()) choices = non_single_choices;
        }
        if(choices.size() == 2) out_two_moves++;
        if(choices.size() == 3) out_three_moves++;
        if(choices.size() >= 4) {
            if(current_combo.type == NONE) {
                out_big_choice_total_start += choices.size();
                out_big_choice_count_start++;
            } else {
                out_big_choice_total_resp += choices.size();
                out_big_choice_count_resp++;
            }
        }
        uniform_int_distribution<int> dist(0, (int)choices.size()-1);
        Combo pick = choices[dist(rng)];
        moves++;
        if(pick.type == PASS) {
            out_passes++;
            current = last_player;
            current_combo = {NONE,0,0};
        } else {
            if(current_combo.type == NONE) out_tricks[pick.type]++;
            int r = pick.rank, L = pick.length;
            switch(pick.type) {
              case SINGLE:    hand[r]--; break;
              case DOUBLE:    hand[r]-=2; break;
              case TRIPLE:    hand[r]-=3; break;
              case FULLHOUSE:
                  hand[r]-=3;
                  for(int pr=0; pr<13; ++pr)
                      if(pr!=r && hand[pr]>=2) { hand[pr]-=2; break; }
                  break;
              case STRAIGHT:
              case SISTERS:
              case TRIPLESTRAIGHT: {
                  int cnt = (pick.type==STRAIGHT?1:(pick.type==SISTERS?2:3));
                  for(int x=r-L+1; x<=r; ++x) hand[x]-=cnt;
                  break;
              }
              case BOMB:
                  hand[r] -= (r==11?3:4);
                  break;
              default: break;
            }
            current_combo = pick;
            last_player     = current;
            current         = 1 - current;
        }
    }
    return moves;
}

int main(){
    const int TRIALS = 50000;
    mt19937 rng(random_device{}());
    long long sum_moves = 0, sum_passes = 0;
    array<long long,10> sum_tricks{};
    sum_tricks.fill(0);
    long long sum_two_moves = 0, sum_three_moves = 0;
    long long sum_big_choice_total_start = 0, sum_big_choice_count_start = 0;
    long long sum_big_choice_total_resp  = 0, sum_big_choice_count_resp  = 0;
    for(int t=1; t<=TRIALS; ++t){
        long long passes;
        array<long long,10> tricks;
        long long two_moves, three_moves, big_choice_total_start, big_choice_count_start, big_choice_total_resp, big_choice_count_resp;
        int moves = simulate_game(
            rng, passes, tricks, two_moves, three_moves,
            big_choice_total_start, big_choice_count_start,
            big_choice_total_resp, big_choice_count_resp
        );
        sum_moves   += moves;
        sum_passes  += passes;
        sum_two_moves += two_moves;
        sum_three_moves += three_moves;
        sum_big_choice_total_start += big_choice_total_start;
        sum_big_choice_count_start += big_choice_count_start;
        sum_big_choice_total_resp  += big_choice_total_resp;
        sum_big_choice_count_resp  += big_choice_count_resp;
        for(int i=0;i<10;++i) sum_tricks[i] += tricks[i];
        if(t % 1000 == 0)
            cout << "Completed " << t << " / " << TRIALS << " games\n";
    }
    cout << fixed << setprecision(2);
    cout << "Average moves:  " << (double)sum_moves/TRIALS << "\n";
    cout << "Average passes: " << (double)sum_passes/TRIALS << "\n";
    cout << "Avg. tricks by type:\n";
    for(int i=1; i<=8; ++i) {
        cout << "  " << ComboNames[i] << ": "
             << (double)sum_tricks[i]/TRIALS << "\n";
    }
    cout << "Average moves with exactly 2 choices: " << (double)sum_two_moves/TRIALS << "\n";
    cout << "Average moves with exactly 3 choices: " << (double)sum_three_moves/TRIALS << "\n";
    if(sum_big_choice_count_start)
        cout << "Average legal moves per 'big choice' (≥4) turn, start-of-trick: "
             << (double)sum_big_choice_total_start/sum_big_choice_count_start << "\n";
    else
        cout << "No 'big choice' start-of-trick turns found.\n";
    cout << "Average number of 'big choice' start-of-trick turns per game: "
         << (double)sum_big_choice_count_start/TRIALS << "\n";
    if(sum_big_choice_count_resp)
        cout << "Average legal moves per 'big choice' (≥4) turn, response: "
             << (double)sum_big_choice_total_resp/sum_big_choice_count_resp << "\n";
    else
        cout << "No 'big choice' response turns found.\n";
    cout << "Average number of 'big choice' response turns per game: "
         << (double)sum_big_choice_count_resp/TRIALS << "\n";
    return 0;
}
