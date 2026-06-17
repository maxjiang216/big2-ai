#ifndef AZ_PI_PI_PRUNE_H
#define AZ_PI_PI_PRUNE_H

// Weakly dominated attachment pruning (bomb kickers, full-house pairs).
//
// If two variants of the same base combo differ only in a LOOSE attachment
// (a card/pair that can never participate in any larger combo from this
// hand), attaching the LOWER rank weakly dominates: the kept higher card
// beats a superset of what the kept lower card beats, and is legal to play
// whenever the lower one is. Hands only ever lose cards, so "loose" is
// monotone — safe to decide locally. The argument is pure dominance and
// holds in both perfect- and imperfect-information play.
//
// Used in the solver's move loop and in search expansion (callers exempt the
// root so policy targets keep the full move set).

#include "game.h"
#include "move.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <vector>

namespace az_pi {

// Loose single: exactly one held, and no fully-held 5-rank straight window
// contains it (so it can never join a straight; pairs/triples impossible).
inline bool loose_single(const std::array<int, 13> &h, int r) {
  if (h[r] != 1) return false;
  for (int lo = std::max(0, r - 4); lo <= r && lo + 4 <= 12; ++lo) {
    bool full = true;
    for (int k = lo; k < lo + 5; ++k)
      if (h[k] == 0) { full = false; break; }
    if (full) return false;
  }
  return true;
}

// Loose pair: exactly two held (no triple/bomb potential) and no adjacent
// rank holds a pair (so it can never join a double straight).
inline bool loose_pair(const std::array<int, 13> &h, int r) {
  if (h[r] != 2) return false;
  if (r > 0 && h[r - 1] >= 2) return false;
  if (r < 12 && h[r + 1] >= 2) return false;
  return true;
}

// Remove dominated attachment variants from `legal` in place. For each base
// combo (bomb of rank b / full house on triple b), among variants whose
// attachment is loose only the lowest-ranked attachment survives; variants
// with non-loose attachments are always kept (they change combo structure).
inline void prune_dominated_attachments(const std::array<int, 13> &h,
                                        std::vector<int> &legal) {
  // best loose attachment rank per base rank, -1 = none seen
  std::array<int8_t, 13> best_bomb, best_fh;
  best_bomb.fill(-1);
  best_fh.fill(-1);

  auto classify = [](int mid, int &base, int &att, bool &is_fh) -> bool {
    const Move m(mid);
    if (m.combination == Move::Combination::kBomb) {
      base = att = -1;
      for (int r = 0; r < 13; ++r) {
        const int c = MOVE_TO_CARDS[mid][r];
        if (c >= 3) base = r;        // 4-of-a-kind, or AAA (3)
        else if (c == 1) att = r;    // kicker
      }
      is_fh = false;
      return base >= 0 && att >= 0;  // kickerless bombs are left alone
    }
    if (m.combination == Move::Combination::kFullHouse) {
      base = att = -1;
      for (int r = 0; r < 13; ++r) {
        const int c = MOVE_TO_CARDS[mid][r];
        if (c == 3) base = r;
        else if (c == 2) att = r;
      }
      is_fh = true;
      return base >= 0 && att >= 0;
    }
    return false;
  };

  bool any = false;
  for (int mid : legal) {
    int base, att;
    bool is_fh;
    if (!classify(mid, base, att, is_fh)) continue;
    if (is_fh ? !loose_pair(h, att) : !loose_single(h, att)) continue;
    auto &best = is_fh ? best_fh : best_bomb;
    if (best[base] < 0 || att < best[base]) best[base] = static_cast<int8_t>(att);
    any = true;
  }
  if (!any) return;

  legal.erase(std::remove_if(legal.begin(), legal.end(),
                             [&](int mid) {
                               int base, att;
                               bool is_fh;
                               if (!classify(mid, base, att, is_fh))
                                 return false;
                               if (is_fh ? !loose_pair(h, att)
                                         : !loose_single(h, att))
                                 return false;
                               const auto &best = is_fh ? best_fh : best_bomb;
                               return att != best[base];
                             }),
              legal.end());
}

}  // namespace az_pi

#endif  // AZ_PI_PI_PRUNE_H
