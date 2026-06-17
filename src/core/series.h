#ifndef SERIES_H
#define SERIES_H

#include <array>
#include <istream>
#include <random>
#include <string>

// ---------------------------------------------------------------------------
// Series scoring + win-probability tables for the multi-game match objective.
//
// A series is played to kSeriesTarget points. After each game the winner scores
// points equal to the loser's remaining cards, with flat replacements for large
// counts (see points_for_cards). The game winner leads the next game.
//
// A series STATE is (a, b) = (points of the player leading the next game,
// points of the follower). The leader is the previous game's winner; at (0,0)
// it is the holder of the 3 of spades (sampled, see sample_first_player_3s).
// ---------------------------------------------------------------------------

constexpr int kSeriesTarget = 50;

// Points the winner scores given the loser's remaining card count.
// 1..12 -> identity; 13 -> 20; 14 -> 30; 15 -> 40; 16 -> 50.
int points_for_cards(int cards_remaining);

// Series win-probability + natural-frequency tables, leader-perspective.
//   v[a][b]       = P(the player leading a game at state (a, b) wins the series)
//   natural[a][b] = expected number of games started at state (a, b) per series
// Indices are series points in [0, kSeriesTarget); states with a >= 50 or
// b >= 50 are terminal and not stored.
struct SeriesTable {
  std::array<std::array<float, kSeriesTarget>, kSeriesTarget> v{};
  std::array<std::array<float, kSeriesTarget>, kSeriesTarget> natural{};
  bool loaded = false;
};

// Parse a `a,b,v,natural_freq` CSV (lines beginning with '#' are comments).
bool load_series_table(const std::string &csv_path, SeriesTable &out);
bool load_series_table(std::istream &in, SeriesTable &out);

// P(the game's winner ultimately wins the series) after a game in which the
// winner held `winner_pts` and the loser `loser_pts` BEFORE the game, and the
// loser finished with `loser_cards_remaining` cards. Returns 1.0 if the win
// ends the series, else table.v[winner_pts + p][loser_pts] (winner leads next).
float series_value_after_win(const SeriesTable &t, int winner_pts, int loser_pts,
                             int loser_cards_remaining);

// Sample which seat (0 or 1) holds the lowest spade (then heart, ...) for the
// 3-of-spades opening rule, given the rank-count hands. The deck is rank-only;
// suit assignment within each rank is uniform, so the holder of a specific
// suited card is an exact conditional draw. rng advances.
int sample_first_player_3s(const std::array<int, 13> &hand0,
                           const std::array<int, 13> &hand1, std::mt19937 &rng);

// True if any straight-family move (single/double/triple straight) is a legal
// lead from this hand. opp1_series_move is only valid when this is false.
bool hand_has_straight_lead(const std::array<int, 13> &hand);

// Series-optimal next move when WE lead, the opponent holds exactly 1 card, and
// the hand has no legal straight lead (caller must check hand_has_straight_lead).
// Sheds all multi-card combos (unbeatable by a 1-card opponent), attaching the
// smallest loose single as a bomb auxiliary, then plays singles highest-first.
// Returns an encoded move id. Re-query after each move.
int opp1_series_move(const std::array<int, 13> &hand);

#endif
