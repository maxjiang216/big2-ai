#ifndef SAMPLES_MD_H
#define SAMPLES_MD_H

#include "game_record.h"

#include <string>
#include <utility>
#include <vector>

// Writes Markdown with starting hands + turn-by-turn history for anomaly games
// (longest, shortest, most/fewest opening legal moves).
bool write_samples_md(const std::vector<std::pair<int, GameRecord>> &indexed,
                      const std::string &path);

#endif
