// build_memo.cpp (revised)
// Compile: g++ -O3 -std=c++17 -o build_memo build_memo.cpp
// -----------------------------------------------------------------------------
// Generates two files:
//   1. memo.jsonl  –  every hand (≤15 cards) whose best_move is **non-trivial**
//                     ("h" = hand array, "s" = score, "m" = [len,endRank]).
//   2. sample.txt  –  up to 1 000 non-trivial hands with only hand & score for
//                     quick sanity-checking.
// After the run it prints the distribution of scores for both trivial and
// non-trivial positions as well as the counts of each category.
// -----------------------------------------------------------------------------
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────── Constants ───────────────────────────────
constexpr int RANKS      = 13;                      // 0 = 3 … 12 = 2
constexpr int MAX_CARDS  = 15;
constexpr std::array<uint8_t, RANKS> MAX_COUNTS{4,4,4,4,4,4,4,4,4,4,4,3,1};

// ────────────────────────── Hand representation & hashing ────────────────────
using Hand = std::array<uint8_t, RANKS>;            // small, trivially copyable
struct HandHash {
    std::size_t operator()(const Hand& h) const noexcept {
        std::size_t hash = 14695981039346656037ull; // FNV-1a
        for (uint8_t c : h) {
            hash ^= c;
            hash *= 1099511628211ull;
        }
        return hash;
    }
};
struct Result {
    uint8_t                                       score;   // best score (0-15)
    std::optional<std::pair<uint8_t,uint8_t>>     move;    // (length, endRank)
};
static std::unordered_map<Hand, Result, HandHash> MEMO;

// ─────────────────────────────────── Forward decl. ───────────────────────────
Result fun(const Hand& hand);

// ───────────────────────── Hand generator (depth-first) ──────────────────────
void gen_rec(int idx, int total, Hand& hand, std::vector<Hand>& out) {
    if (idx == RANKS) { out.push_back(hand); return; }
    const uint8_t cap = MAX_COUNTS[idx];
    for (uint8_t c = 0; c <= cap; ++c) {
        int new_total = total + c;
        if (new_total > MAX_CARDS) break;           // prune
        hand[idx] = c;
        gen_rec(idx+1, new_total, hand, out);
    }
    hand[idx] = 0;                                  // restore
}

// ───────────────────────────── Utility helpers ───────────────────────────────
inline bool all_positive(const Hand& h) {
    for (uint8_t x : h) if (!x) return false;
    return true;
}

// ─────────────────────────── DP core (heuristic) ─────────────────────────────
Result fun(const Hand& hand) {
    if (auto it = MEMO.find(hand); it != MEMO.end()) return it->second;

    uint8_t out = 15;                                      // worst/best sentinel
    std::optional<std::pair<uint8_t,uint8_t>> best_move;   // none yet

    // numExtras = 4-of-kind or Ace-triple bonus cards +1 baseline
    uint8_t numExtras = 1;
    for (int i = 0; i < RANKS; ++i)
        if (hand[i] == 4 || (i == 11 && hand[i] == 3)) ++numExtras;

    // simple discard heuristic
    for (int rank = 0; rank < RANKS; ++rank) {
        if (hand[rank] == 1) {
            if (numExtras) --numExtras;
            else { out = rank + 3; break; }
        }
    }
    if (out == 15) {
        Result res{15, std::nullopt};
        MEMO.emplace(hand, res);
        return res;
    }

    // working copy
    Hand tmp = hand;

    // straight lengths 13 .. 5
    for (int len = 13; len >= 5; --len) {
        int max_start = 13 - len;
        for (int s = 0; s <= max_start; ++s) {
            bool legal = true;
            for (int j = 0; j < len; ++j) if (!hand[s+j]) { legal = false; break; }
            if (!legal) continue;
            for (int j = 0; j < len; ++j) tmp[s+j]--;
            auto sub = fun(tmp);
            for (int j = 0; j < len; ++j) tmp[s+j]++;
            if (sub.score > out) {
                out = sub.score;
                best_move = {static_cast<uint8_t>(len), static_cast<uint8_t>(s+len-1)};
                if (out == 15) goto finish;               // perfect
            }
        }
        // bottom-2 special straight
        if (hand[12] && len < 13) {
            bool legal = true;
            for (int j = 0; j < len-1; ++j) if (!hand[j]) { legal = false; break; }
            if (legal) {
                tmp[12]--; for (int j = 0; j < len-1; ++j) tmp[j]--;
                auto sub = fun(tmp);
                tmp[12]++; for (int j = 0; j < len-1; ++j) tmp[j]++;
                if (sub.score > out) {
                    out       = sub.score;
                    best_move = {static_cast<uint8_t>(len), static_cast<uint8_t>(len-2)};
                    if (out == 15) goto finish;
                }
            }
        }
    }

finish:
    Result res{out, best_move};
    MEMO.emplace(hand, res);
    return res;
}

// ───────────────────────────────────── main ──────────────────────────────────
int main() {
    // Generate universe of hands
    std::vector<Hand> hands; Hand h{}; gen_rec(0,0,h,hands);

    std::ofstream jsonl("memo.jsonl");
    std::ofstream sample("sample.txt");

    std::array<std::uint64_t,16> trivDist{};      // score freq for trivial
    std::array<std::uint64_t,16> nonTrivDist{};   // score freq for non-trivial

    std::size_t nonTrivialCnt = 0;
    const std::size_t SAMPLE_CAP = 1000;

    std::size_t processed = 0;
    for (const auto& hd : hands) {
        auto res = fun(hd);
        ++processed;
        if (res.move) {                           // non-trivial
            ++nonTrivialCnt;
            ++nonTrivDist[res.score];
            // write full JSON-line
            jsonl << "{\"h\":[";
            for (int i = 0; i < RANKS; ++i) {
                jsonl << int(hd[i]);
                if (i+1 != RANKS) jsonl << ',';
            }
            jsonl << "],\"s\":" << int(res.score)
                  << ",\"m\":[" << int(res.move->first)
                  << ',' << int(res.move->second) << "]}\n";
            // sampled plain hand+score
            if (nonTrivialCnt <= SAMPLE_CAP) {
                for (int i = 0; i < RANKS; ++i) {
                    sample << int(hd[i]) << (i+1==RANKS ? ' ' : ',');
                }
                sample << int(res.score) << '\n';
            }
        } else {
            ++trivDist[res.score];
        }
        if (processed % 100000 == 0)
            std::cerr << "processed " << processed << " hands...\n";
    }

    jsonl.close(); sample.close();

    // ───────────── print distributions ─────────────
    std::cout << "\nScore distribution (trivial vs non-trivial)\n";
    std::cout << "score\ttrivial\tnon_trivial\n";
    for (int s = 0; s <= 15; ++s)
        std::cout << s << '\t' << trivDist[s] << '\t' << nonTrivDist[s] << '\n';

    std::cout << "\nTotal hands        : " << hands.size() << '\n';
    std::cout << "Non-trivial hands  : " << nonTrivialCnt << '\n';
    std::cout << "memo.jsonl lines   : " << nonTrivialCnt << '\n';
    std::cout << "sample.txt lines   : " << std::min<std::size_t>(nonTrivialCnt,SAMPLE_CAP) << '\n';
    return 0;
}